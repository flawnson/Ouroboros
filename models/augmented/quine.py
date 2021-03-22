import torch
import numpy as np

from typing import *
from copy import deepcopy
from logzero import logger
from abc import ABC, abstractmethod

from utils.utilities import get_example_size
from utils.reduction import Reduction
from torch.utils.data import Dataset
from utils.utilities import timed


class Quine(ABC):
    @abstractmethod
    def __init__(self, config: Dict, model: torch.nn.Module, device: torch.device):
        super(Quine, self).__init__()
        self.model_aug_config = config["model_aug_config"]
        self.model = model
        self.device = device
        self.param_list = [] + self.model.param_list  # Combine the parameters from the main model
        self.param_names = []
        self.cum_params_arr = np.cumsum(np.array([np.prod(p.shape) for p in self.param_list]))
        self.num_params = int(self.cum_params_arr[-1])

    def indexer(self) -> List:
        """
        Function that reaches into the model parameters to return their coordinates

        Returns:
            A list of the coordinates of the model parameters
        """
        coordinates = []
        counter = 0
        for i, params in enumerate(self.param_list):
            try:
                for n, param in enumerate(params):
                    try:
                        for d, p in enumerate(param):
                            coordinates.append([i, n, d])
                    except TypeError:
                        coordinates.append([0, 0, 0])  # Sacrificing the first param
                        counter += 1
            except TypeError:
                coordinates.append([0, 0, 0])  # Sacrificing the first param
                counter += 1

        logger.info(f"Regeneration will fail for {counter} parameters")

        return coordinates

    @abstractmethod
    def regenerate(self):
        pass

    def reduction(self, data_size) -> Reduction:
        """
        Select the reduction method

        Returns:
            Output of the reduction method
        """
        return Reduction(self.model_aug_config, data_size).reduce()

    def get_param(self, idx: int) -> torch.tensor:
        assert idx < self.num_params
        subtract = 0
        param = None
        normalized_idx = None
        for i, n_params in enumerate(self.cum_params_arr):
            if idx < n_params:
                param = self.param_list[i]
                normalized_idx = idx - subtract
                break
            else:
                subtract = n_params
        return param.view(-1)[normalized_idx]

    @abstractmethod
    def forward(self, x: torch.tensor, y: torch.tensor = None):
        pass


class Vanilla(Quine, torch.nn.Module):
    def __init__(self, config: Dict, model: torch.nn.Module, device: torch.device):
        super(Vanilla, self).__init__(config, model, device)
        self.model_aug_config = config["model_aug_config"]
        self.model = model
        self.device = device
        self.van_input = self.van_input()
        self.van_output = self.van_output()

    def van_input(self) -> torch.nn.Sequential:
        rand_proj_layer = torch.nn.Linear(self.num_params,
                                          self.model_aug_config["n_hidden"] // self.model_aug_config["n_inputs"],
                                          bias=False)  # Modify so there are half as many hidden units
        rand_proj_layer.weight.data = torch.tensor(self.reduction(self.num_params), dtype=torch.float32)
        for p in rand_proj_layer.parameters():
            p.requires_grad_(False)
        return torch.nn.Sequential(rand_proj_layer)

    def van_output(self) -> torch.nn.Sequential:
        # TODO: Make cleaner
        weight_predictor_layers = []
        current_layer = torch.nn.Linear(self.model_aug_config["n_hidden"], 1, bias=True)
        weight_predictor_layers.append(current_layer)
        self.param_list.append(current_layer.weight)
        self.param_names.append("wp_layer{}_weight".format(0))
        self.param_list.append(current_layer.bias)
        self.param_names.append("wp_layer{}_bias".format(0))
        return torch.nn.Sequential(*weight_predictor_layers)

    def forward(self, x: torch.tensor, y: torch.tensor = None) -> Dict:
        x = self.van_input()(x)
        x = self.model(x)
        x = self.van_output()(x)
        return {"param": x}

    @timed
    @torch.no_grad()
    def regenerate(self):
        """
        The regenerate, implemented by following the original Quine paper.
        Model parameters are kept in self.param_list and used for training and inference
        Due to the iteration, the model uses the regenerated version of itself to regenerate the next parameter.
        # TODO: Regenerate takes way too long on cpu; refactor to make faster
        """
        params_data = torch.eye(self.num_params, device=self.device)
        index_list = list(range(self.num_params))
        coordinates = self.indexer()
        logger.info(f"Regenerating {len(coordinates)} parameters")
        for param_idx, coo in zip(index_list, coordinates):
            logger.info(f"Regenerating parameter {param_idx}")
            with torch.no_grad():
                idx_vector = torch.squeeze(params_data[param_idx])  # Pulling out the nested tensor
                predicted_param = self.forward(idx_vector, None).values()  # Forward output is dict
                new_params = deepcopy(self.param_list)
                new_params[coo[0]][coo[1]][coo[2]] = predicted_param
                self.param_list = new_params
        logger.info(f"Successfully regenerated weights")


class Auxiliary(Vanilla, torch.nn.Module):
    def __init__(self, config: Dict, model: torch.nn.Module, dataset: Dataset, device):
        super(Auxiliary, self).__init__(config, model, device)
        super(torch.nn.Module)
        self.config_aug_config = config["model_aug_config"]
        self.model = model
        self.dataset = dataset
        self.device = device
        self.aux_input = self.aux_input()
        self.aux_output = self.aux_output()

    def aux_input(self) -> torch.nn.Sequential:
        rand_proj_layer = torch.nn.Linear(get_example_size(self.dataset),
                                          self.model_aug_config["n_hidden"] // self.model_aug_config["n_inputs"],
                                          bias=False)  # Modify so there are half as many hidden units
        rand_proj_layer.weight.data = torch.tensor(self.reduction(get_example_size(self.dataset)), dtype=torch.float32)
        for p in rand_proj_layer.parameters():
            p.requires_grad_(False)
        return torch.nn.Sequential(rand_proj_layer)

    def aux_output(self) -> torch.nn.Sequential:
        # TODO: Make cleaner
        digit_predictor_layers = []
        current_layer = torch.nn.Linear(self.model_aug_config["n_hidden"], 10, bias=True)
        logsoftmax = torch.nn.LogSoftmax(dim=0) #should have no learnable weights
        digit_predictor_layers.append(current_layer)
        digit_predictor_layers.append(logsoftmax)
        self.param_list.append(current_layer.weight)
        self.param_names.append("dp_layer{}_weight".format(0))
        self.param_list.append(current_layer.bias)
        self.param_names.append("dp_layer{}_bias".format(0))
        return torch.nn.Sequential(*digit_predictor_layers)

    def forward(self, x: torch.tensor, y: torch.tensor = None) -> Dict:
        """
        Forward method of augmented model

        Args:
            x: The one hot coordinate for the model parameter
            y: Auxiliary input data (MNIST)

        Returns:
            torch.tensor output
        """
        new_output = self.van_input()(x)
        if y is not None:
            y = y.reshape(-1)  # Flatten MNIST input in place
            output2 = self.aux_input()(y)
            new_output = torch.cat((new_output, output2))
        else:
            new_output = torch.cat((new_output, torch.rand(self.model_aug_config["n_hidden"])))

        # run_logging.info("Input 1: ", output1)
        # run_logging.info("Input 2: ", output2)

        #concatenate and feed both into main Network
        output3 = self.model(new_output)

        weight = self.van_output()(output3)  # Weight prediction network
        aux_output = self.aux_output()(output3)  # Auxiliary prediction network

        return {"param": weight, "aux": aux_output}

    @timed
    @torch.no_grad()
    def regenerate(self):
        """
        The regenerate, implemented by following the original Quine paper.
        Model parameters are kept in self.param_list and used for training and inference
        Due to the iteration, the model uses the regenerated version of itself to regenerate the next parameter.
        # TODO: Regenerate takes way too long on cpu; refactor to make faster
        """
        params_data = torch.eye(self.num_params, device=self.device)
        index_list = list(range(self.num_params))
        coordinates = self.indexer()
        logger.info(f"Regenerating {len(coordinates)} parameters")
        for param_idx, coo in zip(index_list, coordinates):
            logger.info(f"Regenerating parameter {param_idx}")
            with torch.no_grad():
                idx_vector = torch.squeeze(params_data[param_idx])  # Pulling out the nested tensor
                predicted_param, predicted_aux = self.forward(idx_vector, None).values()  # Forward output is dict
                new_params = deepcopy(self.param_list)
                new_params[coo[0]][coo[1]][coo[2]] = predicted_param
                self.param_list = new_params
        logger.info(f"Successfully regenerated weights")

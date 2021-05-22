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
        self.cum_params_arr = np.cumsum(np.array([np.prod(p.shape) for p in self.param_list]))
        self.num_params = int(self.cum_params_arr[-1])
        #if we specify a smaller subset, use that amount instead
        subset = config["data_config"]["subset"]
        if isinstance(subset, int):
            self.num_params = subset

    def to_onehot(self, idxs: torch.tensor):
        idxs = idxs.item() if type(idxs) == torch.tensor else idxs
        onehot = torch.zeros(self.num_params, device=self.device)
        onehot[idxs] = 1
        # onehots = [torch.zeros(self.num_params, device=self.device)[idx.item()] for idx in idxs]  # Was testing different batch sizes
        return onehot

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
                if len(list(params.size())) == 0:
                    params = torch.unsqueeze(params, dim=0)
                for n, param in enumerate(params):
                    try:
                        if len(list(param.size())) == 0:
                            param = torch.unsqueeze(param, dim=0)
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
        self.van_input = self.build_van_input()
        self.van_output = self.build_van_output()

    def build_van_input(self) -> torch.nn.Sequential:
        rand_proj_layer = torch.nn.Linear(self.num_params,
                                          self.model_aug_config["n_hidden"] // self.model_aug_config["n_inputs"],
                                          bias=False)  # Modify so there are half as many hidden units
        rand_proj_layer.weight.data = torch.tensor(self.reduction(self.num_params), dtype=torch.float32)
        for p in rand_proj_layer.parameters():
            p.requires_grad_(False)
        return torch.nn.Sequential(rand_proj_layer)

    def build_van_output(self) -> torch.nn.Sequential:
        weight_predictor_layers = []
        for in_size, out_size in zip(self.model_aug_config["van_output_layers"], self.model_aug_config["van_output_layers"][1:]):
            layer = torch.nn.Linear(in_size, out_size, bias=True)
            weight_predictor_layers.append(layer)
            self.param_list.append(layer.weight)
            self.param_list.append(layer.bias)
        return torch.nn.Sequential(*weight_predictor_layers)

    def forward(self, x: torch.tensor, y: torch.tensor = None) -> Dict:
        x = self.van_input(x)
        x = self.model(x)
        x = self.van_output(x)
        return x

    @timed
    @torch.no_grad()
    def regenerate(self):
        """
        The regenerate, implemented by following the original Quine paper.
        Model parameters are kept in self.param_list and used for training and inference
        Due to the iteration, the model uses the regenerated version of itself to regenerate the next parameter.
        """
        index_list = list(range(self.num_params))
        coordinates = self.indexer()
        logger.info(f"Regenerating {len(coordinates)} parameters")
        for param_idx, coo in zip(index_list, coordinates):
            logger.info(f"Regenerating parameter {param_idx}")
            with torch.no_grad():
                idx_vector = torch.squeeze(self.to_onehot(param_idx))  # Pulling out the nested tensor
                predicted_param = self.forward(idx_vector, None)
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
        self.aux_input = self.build_aux_input()
        self.aux_output = self.build_aux_output()

    def build_aux_input(self) -> torch.nn.Sequential:
        rand_proj_layer = torch.nn.Linear(get_example_size(self.dataset),
                                          self.model_aug_config["n_hidden"] // self.model_aug_config["n_inputs"],
                                          bias=False)  # Modify so there are half as many hidden units
        rand_proj_layer.weight.data = torch.tensor(self.reduction(get_example_size(self.dataset)), dtype=torch.float32)
        for p in rand_proj_layer.parameters():
            p.requires_grad_(False)
        return torch.nn.Sequential(rand_proj_layer)

    def build_aux_output(self) -> torch.nn.Sequential:
        # TODO: Make cleaner
        aux_predictor_layers = []
        for in_size, out_size in zip(self.model_aug_config["aux_output_layers"], self.model_aug_config["aux_output_layers"][1:]):
            layer = torch.nn.Linear(in_size, out_size, bias=True)
            aux_predictor_layers.append(layer)
            self.param_list.append(layer.weight)
            self.param_list.append(layer.bias)
        logsoftmax = torch.nn.LogSoftmax(dim=0) #should have no learnable weights
        aux_predictor_layers.append(logsoftmax)
        return torch.nn.Sequential(*aux_predictor_layers)

    def forward(self, x: torch.tensor, y: torch.tensor = None) -> Tuple[torch.tensor, torch.tensor]:
        """
        Forward method of augmented model
        Args:
            x: The one hot coordinate for the model parameter
            y: Auxiliary input data (MNIST)
        Returns:
            torch.tensor output
        """
        new_output = self.van_input(x)
        if y is not None:
            y = y.reshape(-1)  # Flatten MNIST input in place
            output2 = self.aux_input(y)
            new_output = torch.cat((new_output, output2))
        else:
            # Substitutes data with random matrix during regeneration... Could probably do better
            new_output = torch.cat((new_output, torch.rand(self.model_aug_config["n_hidden"]).to(self.device)))

        # run_logging.info("Input 1: ", output1)
        # run_logging.info("Input 2: ", output2)

        #concatenate and feed both into main Network
        output3 = self.model(new_output)

        weight = self.van_output(output3)  # Weight prediction network
        aux_output = self.aux_output(output3)  # Auxiliary prediction network

        return weight, aux_output

    @timed
    @torch.no_grad()
    def regenerate(self):
        """
        The regenerate, implemented by following the original Quine paper.
        Model parameters are kept in self.param_list and used for training and inference
        Due to the iteration, the model uses the regenerated version of itself to regenerate the next parameter.
        """
        index_list = list(range(self.num_params))
        coordinates = self.indexer()
        logger.info(f"Regenerating {len(coordinates)} parameters")
        for param_idx, coo in zip(index_list, coordinates):
            logger.info(f"Regenerating parameter {param_idx}")
            with torch.no_grad():
                idx_vector = torch.squeeze(self.to_onehot(param_idx))  # Pulling out the nested tensor
                predicted_param, predicted_aux = self.forward(idx_vector, None)
                new_params = deepcopy(self.param_list)
                try:
                    # To catch ases where the parameter tensor is of size 0
                    new_params[coo[0]][coo[1]][coo[2]] = predicted_param
                except Exception as e:
                    logger.exception(e)
                    new_params[coo[0]] = predicted_param
                self.param_list = new_params
        logger.info(f"Successfully regenerated weights")

    def multiregenerate(self):
        """
        Regenerate, inspired by hypernetworks and quines.
        This method is meant to regenerate entire layers or models at a time rather than individual weights.
        """
        pass
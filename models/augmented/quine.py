import torch
import numpy as np

from typing import *
from copy import deepcopy
from logzero import logger
from abc import ABC, abstractmethod
from sklearn import random_projection

from utils.utilities import get_example_size


class Quine(ABC):
    @abstractmethod
    def __init__(self, config, model, device):
        super(Quine, self).__init__()
        self.model_aug_config = config["model_aug_config"]
        self.model = model
        self.device = device
        self.param_list = []
        self.param_names = []
        self.num_params = self.num_params()

    def projection(self):
        X = np.random.rand(1, self.num_params)
        transformer = random_projection.GaussianRandomProjection(n_components=self.model_aug_config["n_hidden"])
        transformer.fit(X)
        rand_proj_matrix = transformer.components_

        return rand_proj_matrix

    def num_params(self, params=[]):  # To account for the input and output parameters not part of the main model
        # Create the parameter counting function
        # TODO: Check if function as expected
        num_params_arr = np.array([np.prod(p.shape) for p in list(self.model.parameters()) + self.param_list])
        cum_params_arr = np.cumsum(num_params_arr)
        num_params = int(cum_params_arr[-1])

        return num_params

    def get_param(self, idx):
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
    def forward(self):
        pass


class Vanilla(Quine, torch.nn.Module):
    def __init__(self, config, model, device):
        super(Vanilla, self).__init__(config, model, device)
        self.model_aug_config = config["model_aug_config"]
        self.model = model
        self.device = device

    def van_input(self):
        rand_proj_layer = torch.nn.Linear(self.num_params, self.model_aug_config["n_hidden"] // self.model_aug_config["n_inputs"],
                                          bias=False)  # Modify so there are half as many hidden units
        rand_proj_layer.weight.data = torch.tensor(self.projection(), dtype=torch.float32)
        for p in rand_proj_layer.parameters():
            p.requires_grad_(False)
        return torch.nn.Sequential(rand_proj_layer)

    def van_output(self):
        # TODO: Make cleaner
        weight_predictor_layers = []
        current_layer = torch.nn.Linear(self.model_aug_config["n_hidden"], 1, bias=True)
        weight_predictor_layers.append(current_layer)
        self.param_list.append(current_layer.weight)
        self.param_names.append("wp_layer{}_weight".format(0))
        self.param_list.append(current_layer.bias)
        self.param_names.append("wp_layer{}_bias".format(0))
        return torch.nn.Sequential(*weight_predictor_layers)

    def forward(self, x):
        x = self.van_input(x)
        x = self.model(x)
        x = self.van_output(x)
        return x

    def get_van(self):
        pass
        # self.van_input()


class Auxiliary(Vanilla, torch.nn.Module):
    def __init__(self, config, model, dataset, device):
        super(Auxiliary, self).__init__(config, model, device)
        super(torch.nn.Module)
        self.config_aug_config = config["model_aug_config"]
        self.model = model
        self.dataset = dataset
        self.device = device
        self.van_model = self.get_van()
        self.aux_model = self.get_aux()

    @staticmethod
    def indexer(model):
        coordinates = []
        counter = 0
        for i, params in enumerate(model.param_list):
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

        logger.info(f"Regeneration failed for {counter} parameters")

        return coordinates

    def regenerate(self):
        # Taken from the training pipeline
        # TODO: Regenerate takes way too long on cpu; refactor to make faster
        params_data = torch.eye(self.van_model.num_params, device=self.device)
        index_list = list(range(self.van_model.num_params))
        coordinates = self.indexer(self.van_model)
        for param_idx, coo in zip(index_list, coordinates):
            with torch.no_grad():
                idx_vector = torch.squeeze(params_data[param_idx])  # Pulling out the nested tensor
                predicted_param, predicted_aux = self.van_model(idx_vector, None)
                new_params = deepcopy(self.van_model.param_list)
                new_params[coo[0]][coo[1]][coo[2]] = predicted_param
                self.van_model.param_list = new_params
        logger.info(f"Successfully regenerated weights")

    def aux_input(self):
        rand_proj_layer = torch.nn.Linear(get_example_size(self.dataset),
                                          self.model_aug_config["n_hidden"] // self.model_aug_config["n_inputs"],
                                          bias=False)  # Modify so there are half as many hidden units
        rand_proj_layer.weight.data = torch.tensor(self.projection(), dtype=torch.float32)
        for p in rand_proj_layer.parameters():
            p.requires_grad_(False)
        return torch.nn.Sequential(rand_proj_layer)

    def aux_output(self):
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

    def forward(self, x, y=None):
        #x = one hot coordinate
        #y = auxiliary input
        new_output = self.van_input()(x)
        if y is not None:
            y = y.reshape(-1) #Flatten MNIST input
            output2 = self.aux_input()(y)
            new_output = torch.cat((new_output, output2))
        else:
            new_output = torch.cat((new_output, torch.rand(20)))

        # run_logging.info("Input 1: ", output1)
        # run_logging.info("Input 2: ", output2)

        #concatenate and feed both into main Network
        output3 = self.model(new_output)

        weight = self.van_output()(output3)  # Weight prediction network
        aux_output = self.aux_output()(output3)  # Auxiliary prediction network

        return weight, aux_output

    def get_aux(self):
        pass
        # self.aux_input()
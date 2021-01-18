import torch
import numpy as np

from abc import ABC, abstractmethod
from sklearn import random_projection


class Quine(ABC):
    @abstractmethod
    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.num_params = self.num_params()

    def num_params(self):
        # Create the parameter counting function
        # TODO: Check if function as as expected
        num_params_arr = np.array([np.prod(p.shape) for p in self.model.parameters()])
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


class Vanilla(Quine):
    def __init__(self, config, model):
        super(Vanilla, self).__init__(config, model)
        self.config = config
        self.model = model
        self.van_model = self.get_van()

    def projection(self):
        X = np.random.rand(1, self.num_params)
        transformer = random_projection.GaussianRandomProjection(n_components=self.config["n_hidden"])
        transformer.fit(X)
        rand_proj_matrix = transformer.components_
        rand_proj_layer = torch.nn.Linear(self.num_params, self.config["n_hidden"], bias=False)
        rand_proj_layer.weight.data = torch.tensor(rand_proj_matrix, dtype=torch.float32)
        for p in rand_proj_layer.parameters():
            p.requires_grad_(False)
        self.model.layers.insert(0, rand_proj_layer)

    def get_van(self):

        return model


class Auxiliary(Vanilla):
    def __init__(self, config, model):
        super(Auxiliary, self).__init__(config, model)
        self.config = config
        self.model = model
        self.aux_model = self.get_aux()

    def get_aux(self):
        self.van_model
        exit()

        return model

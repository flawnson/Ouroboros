import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from abc import ABC
from typing import *
from torch.nn import Linear


class AbstractMLPModel(torch.nn.Module, ABC):
    def __init__(self, config: Dict, layer_dict: List, pooling: torch.nn.functional, device: torch.device):
        """ Class for training and testing loops
        Args:
            config: Model config file (Python dictionary from JSON)
            layer_dict: Dictionary containing layer information including sizes, caching, etc.
            pooling: Either None or torch pooling objects used between layers in forward propagation
            device: torch device object defined in main.py
        Returns:
            torch.tensor
        """
        super(AbstractMLPModel, self).__init__()
        self.config = config
        self.layers = torch.nn.ModuleList([self.factory(info) for info in layer_dict])
        self.pool = pooling if pooling else [None] * len(self.layers)
        self.device = device

    @staticmethod
    def factory(sizes: Dict) -> eval:
        name = sizes["name"]
        sizes_copy = sizes.copy()
        sizes_copy.pop("name", None)
        return eval(name)(**sizes_copy)

    def forward(self, x: torch.tensor) -> torch.tensor:
        z = x
        for layer, pooling in zip(self.layers, self.pool):
            x = layer(x)
            z = x
            x = pooling(x) if pooling else x
            x = F.relu(x)
            x = F.dropout(x, p=self.config["dropout"], training=self.training)
        x = z
        return x


class MLPModel(AbstractMLPModel, ABC):
    # Provide pooling arguments as kwargs (only needed for GlobalAttentionPooling and Set2Set (forward parameters should
    # be provided in the forward function of the model)
    # TODO: implement and test pooling
    def __init__(self, config: Dict, device: torch.device, pooling: str = None, **kwargs):
        self.model_config = config["model_config"]
        self.layer_sizes = [self.model_config["input_layer_size_2*model_aug_n_hidden"]] + \
                            self.model_config["layer_sizes"] + \
                           [self.model_config["output_layer_size_model_aug_n_hidden"]]
        super(MLPModel, self).__init__(
            config=self.model_config,
            layer_dict=[dict(name=nn.Linear.__name__,
                             in_features=in_size,
                             out_features=out_size,
                             bias=True)  # Dictionary keys must be exactly as written in the documentation for the layer
                        for in_size, out_size in zip(self.layer_sizes, self.layer_sizes[1:])],
            pooling=[eval(pooling)(kwargs).to(device) for size in self.layer_sizes[1:]] if pooling else None,
            device=device)
        self.param_list = [layer.weight for layer in self.layers] + [layer.bias for layer in self.layers]  # Keeping track of params for Quine


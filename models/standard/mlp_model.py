import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from abc import ABC
from typing import *
from torch.nn import Linear


class AbstractMLPModel(torch.nn.Module, ABC):
    def __init__(self, config: Dict, layer_dict: List, pooling: F, normalize: F, device: torch.device):
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
        self.norm = normalize if normalize else [None] * len(self.layers)
        self.device = device

    @staticmethod
    def factory(sizes: Dict) -> eval:
        name = sizes["name"]
        sizes_copy = sizes.copy()
        sizes_copy.pop("name", None)
        return eval(name)(**sizes_copy)

    def forward(self, x: torch.tensor) -> torch.tensor:
        z = x
        for layer, pooling, normalization in zip(self.layers, self.pool, self.norm):
            x = layer(x)
            z = x
            x = pooling(x, **self.config["pooling_kwargs"]) if pooling else x
            x = F.relu(x)
            x = normalization(x, normalized_shape=list(x.size()), **self.config["normalize_kwargs"]) if normalization else x
            x = F.dropout(x, p=self.config["dropout"], training=self.training)
        x = torch.nn.functional.log_softmax(z, dim=0)
        return x


class MLPModel(AbstractMLPModel, ABC):
    # Provide pooling arguments as kwargs (only needed for GlobalAttentionPooling and Set2Set (forward parameters should
    # be provided in the forward function of the model)
    def __init__(self, config: Dict, device: torch.device):
        self.model_config = config["model_config"]
        self.layer_sizes = self.model_config["layer_sizes"]
        super(MLPModel, self).__init__(
            config=self.model_config,
            layer_dict=[dict(name=nn.Linear.__name__,
                             in_features=in_size,
                             out_features=out_size,
                             bias=True)  # Dictionary keys must be exactly as written in the documentation for the layer
                        for in_size, out_size in zip(self.layer_sizes, self.layer_sizes[1:])],
            pooling=[eval(self.model_config["pooling"]) for size in self.layer_sizes[1:]] if self.model_config["pooling"] else None,
            normalize=[eval(self.model_config["normalize"]) for size in self.layer_sizes[1:]] if self.model_config["normalize"] else None,
            device=device)
        self.param_list = [layer.weight for layer in self.layers] + [layer.bias for layer in self.layers]  # Keeping track of params for Quine


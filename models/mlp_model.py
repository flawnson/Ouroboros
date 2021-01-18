import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from abc import ABC
from torch.nn import Linear


class AbstractMLPModel(torch.nn.Module, ABC):
    def __init__(self, config: dict, layer_dict: list, pooling: torch.nn.functional, device: torch.device):
        """ Class for training and testing loops
        Args:
            config: Model config file (Python dictionary from JSON)
            linear_model: Either None or Linear model stored as torch Module object (only implemented for GCN model)
            layer_dict: Dicionary containing layer information including sizes, cacheing, etc.
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
    def factory(sizes: dict) -> eval:
        name = sizes["name"]
        sizes_copy = sizes.copy()
        sizes_copy.pop("name", None)
        return eval(name)(**sizes_copy)

    def forward(self, data_obj, x: torch.tensor) -> torch.tensor:
        z = x
        for layer, pooling in zip(self.layers, self.pool):
            x = layer(data_obj, x)
            z = x
            x = pooling(data_obj, x) if pooling else x
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)
        x = z
        return x


class MLPModel(AbstractMLPModel, ABC):
    # Provide pooling arguments as kwargs (only needed for GlobalAttentionPooling and Set2Set (forward parameters should
    # be provided in the forward function of the model)
    # TODO: implement and test pooling
    def __init__(self, config: dict, data: torch.tensor, device: torch.device, pooling: str = None, **kwargs):
        self.data = data
        self.config = config["model_config"]
        self.layer_sizes = [self.config["aux_input_size"] + self.config["weight_input_size"]] +\
                            self.config["layer_sizes"] + \
                           [self.config["aux_output_size"] + self.config["weight_output_size"]]
        super(MLPModel, self).__init__(
            config=self.config,
            layer_dict=[dict(name=nn.Linear.__name__,
                             in_features=in_size,
                             out_features=out_size,
                             bias=True)  # Dictionary keys must be exactly as written in the documentation for the layer
                        for in_size, out_size in zip(self.layer_sizes, self.layer_sizes[1:])],
            pooling=[eval(pooling)(kwargs).to(device) for size in self.layer_sizes[1:]] if pooling else None,
            device=device)


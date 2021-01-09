import dgl
import torch
import numpy as np
import torch.nn.functional as F

from abc import ABC
from nn.graph_layers import GenericGNNLayer
from dgl.nn.pytorch.conv import GraphConv, GATConv, GINConv, SAGEConv, ChebConv, EdgeConv
from dgl.nn.pytorch.glob import AvgPooling, MaxPooling, SortPooling, GlobalAttentionPooling, Set2Set


class AbstractGNNModel(torch.nn.Module, ABC):
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
        super(GenericGNNModel, self).__init__()
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

    def forward(self, graph_obj: dgl.DGLGraph, x: torch.tensor) -> torch.tensor:
        z = x
        for layer, pooling in zip(self.layers, self.pool):
            x = layer(graph_obj, x)
            z = x
            x = pooling(graph_obj, x) if pooling else x
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)
        x = z
        return x


class GNNModel(AbstractGNNModel, ABC):
    # Provide pooling arguments as kwargs (only needed for GlobalAttentionPooling and Set2Set (forward parameters should
    # be provided in the forward function of the model)
    # TODO: implement and test pooling
    def __init__(self, config: dict, data: torch.tensor, device: torch.device, pooling: str = None, **kwargs):
        self.data = data
        self.layer_sizes = [data.ndata["x"].size(1)] + config["layer_sizes"] + [len(np.unique(data.ndata["y"].numpy()))]
        super(GNNModel, self).__init__(
            config=config,
            layer_dict=[dict(name=GenericGNNLayer.__name__,
                             in_channels=in_size,
                             out_channels=out_size)
                        for in_size, out_size in zip(self.layer_sizes, self.layer_sizes[1:])],
            pooling=[eval(pooling)(kwargs).to(device) for size in self.layer_sizes[1:]] if pooling else None,
            device=device)


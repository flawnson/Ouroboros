"""For custom GNN layers"""

import dgl
import torch

import dgl.function as fn
import torch.nn as nn


class GenericGNNLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm="both", weight=True, bias=True):
        """ Class for training and testing loops
            Generic GNN layer can be modified with DGL's built in tools (currently implemented as GCN)
            Reference for self; u: source node, v: destination node, e edges among those nodes
        Args:
            in_channels: The input size of the layer
            out_channels: The output size of the layer
            norm: The type of normalization to use in the layer
            weight: Whether or not to use weights in layer
            bias: Whether or not to use biases in layer
        Returns:
            torch.tensor
        """
        super(GenericGNNLayer, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)
        self.norm = norm
        self.bias: bool = bias
        if weight:
            self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels), requires_grad=True)
        else:
            self.register_parameter('weight', None)

    def reset_parameters(self):
        # Obligatory parameter reset method
        pass

    def forward(self, graph_obj: dgl.DGLGraph, feature: torch.tensor, weight=True) -> torch.tensor:
        # local_scope needed to ensure that the stored data (in messages) doesn't accumulate
        # When implementing a layer you have a choice of initializing your own weights and matmul, or using nn.Linear
        # For performance reasons, DGL's implementation performs operations in order according to input/output size
        # It should be possible however, to use matmul(features, weights) or nn.Linear(features) anyplace anytime
        with graph_obj.local_scope():
            if self.norm == 'both':
                degs = graph_obj.out_degrees().to(feature.device).float().clamp(min=1)
                norm = torch.pow(degs, -0.5)
                shp = norm.shape + (1,) * (feature.dim() - 1)
                norm = torch.reshape(norm, shp)
                feat = feature * norm

            # feat = torch.matmul(feat, weight)
            graph_obj.srcdata['h'] = feat
            graph_obj.update_all(fn.copy_src(src='h', out='m'), fn.sum(msg='m', out='h'))
            feat = graph_obj.dstdata['h']

            if self.bias is not None:
                feat = self.bias + feat

        return self.linear(feat)


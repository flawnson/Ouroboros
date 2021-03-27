import torch
import numpy as np

from typing import *
from logzero import logger
from abc import ABC, abstractmethod

from models.standard.mlp_model import MLPModel


class AbstractHyperNetwork(ABC, torch.nn.Module):
    def __init__(self, config, model, device):
        super(AbstractHyperNetwork, self).__init__()
        self.config = config
        self.model = model
        self.device = device

    def forward(self, x):
        return x


class CNNHyperNetwork(AbstractHyperNetwork):
    """Original HyperNetwork implementation uses an MLP to generate CNN kernels/filters not model weights"""
    def __init__(self, config, model, device):
        """
        Refactor of HyperNetwork implementation from this repo: https://github.com/g1910/HyperNetworks

        Args:
            config: A dictionary of configurations
            model: The SubNetwork (that will receive its filters from the HyperNetwork)
        """
        super(CNNHyperNetwork, self).__init__(config, model, device)
        self.config = config
        self.model = model
        self.device = device
        self.hypernet = self.build_hypernetwork()
        self.embedding = self.build_embedding()

    def build_embedding(self):
        layer_embeddings
        return torch.nn.Embedding()

    def build_hypernetwork(self):
        """
        Build and return an MLP as the HyperNetwork

        Returns: Sequential object with MLP model

        """
        return torch.nn.Sequential(MLPModel(self.config, self.device).to(self.device))

    def forward(self, x):
        """Manually built in original implementation, attempting automatic build"""
        # h_in = torch.matmul(x, self.w2) + self.b2
        # h_in = h_in.view(self.in_size, self.z_dim)
        #
        # h_final = torch.matmul(h_in, self.w1) + self.b1
        # kernel = h_final.view(self.out_size, self.in_size, self.f_size, self.f_size)
        kernel = self.hypernet(self.embedding)
        return kernel


class MLPHyperNetwork(AbstractHyperNetwork):
    """Modified HyperNetwork that will aim to modify weights of another MLP"""
    def __init__(self, config, model, device):
        super(MLPHyperNetwork, self).__init__(config, model, device)
        self.config = config
        self.model = model
        self.device = device

    def forward(self, x):
        return x



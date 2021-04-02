"""Code named file for our augmented self-referential models"""
import torch

from typing import *
from logzero import logger
from abc import ABC, abstractmethod

from models.augmented.hypernetwork import HyperNetwork


class Ouroboros(ABC):
    """Abstract class for any novel ideas we implement"""
    @abstractmethod
    def __init__(self):
        super(Ouroboros, self).__init__()
        pass

    def get_param(self, idx):
        pass


class Jung(Ouroboros, torch.nn.Module):
    """Code-named class for MetaNetworks; Adversarial HyperNetworks"""
    def __init__(self, config, model):
        super(Jung, self).__init__()
        self.config = config
        self.model = model
        self.aux_model = self.get_aux()

    def get_aux(self):

        return model


class Kekule(Ouroboros):
    def __init__(self, config, model):
        super(Kekule, self).__init__()
        self.config = config
        self.model = model


class Godel:
    """ Code name for Dual HyperNetworks """
    def __init__(self, z_dim=64, device="cpu"):
        super(Godel, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(16)

        self.z_dim = z_dim
        self.hope = HyperNetwork(z_dim=self.z_dim, device=device)

        self.zs_size = [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1],
                        [2, 1], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2],
                        [4, 2], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4]]

        self.filter_size = [[16,16], [16,16], [16,16], [16,16], [16,16], [16,16], [16,32], [32,32], [32,32], [32,32],
                            [32,32], [32,32], [32,64], [64,64], [64,64], [64,64], [64,64], [64,64]]

        self.res_net = torch.nn.ModuleList()

        for i in range(18):
            down_sample = False
            if i > 5 and i % 6 == 0:
                down_sample = True
            self.res_net.append(ResNetBlock(self.filter_size[i][0], self.filter_size[i][1], downsample=down_sample))

        self.zs = torch.nn.ModuleList()

        for i in range(36):
            self.zs.append(Embedding(self.zs_size[i], self.z_dim, device))

        self.global_avg = torch.nn.AvgPool2d(8)
        self.final = torch.nn.Linear(64,10)

    def forward(self, x):

        x = F.relu(self.bn1(self.conv1(x)))

        for i in range(18):
            # if i != 15 and i != 17:
            w1 = self.zs[2*i](self.hope)
            w2 = self.zs[2*i+1](self.hope)
            x = self.res_net[i](x, w1, w2)

        x = self.global_avg(x)
        x = self.final(x.view(-1,64))

        return x


class Escher:
    """ Code name for Nested HyperNetworks """
    def __init__(self, z_dim=64, device="cpu"):
        super(Escher, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(16)

        self.z_dim = z_dim
        self.hope1 = HyperNetwork(z_dim=self.z_dim, device=device)
        self.hope2 = HyperNetwork(z_dim=self.z_dim, device=device)

        self.zs_size = [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1],
                        [2, 1], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2],
                        [4, 2], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4]]

        self.filter_size = [[16,16], [16,16], [16,16], [16,16], [16,16], [16,16], [16,32], [32,32], [32,32], [32,32],
                            [32,32], [32,32], [32,64], [64,64], [64,64], [64,64], [64,64], [64,64]]

        self.res_net = torch.nn.ModuleList()

        for i in range(18):
            down_sample = False
            if i > 5 and i % 6 == 0:
                down_sample = True
            self.res_net.append(ResNetBlock(self.filter_size[i][0], self.filter_size[i][1], downsample=down_sample))

        self.zs = torch.nn.ModuleList()

        for i in range(36):
            self.zs.append(Embedding(self.zs_size[i], self.z_dim, device))

        self.global_avg = torch.nn.AvgPool2d(8)
        self.final = torch.nn.Linear(64,10)

    def forward(self, x):

        x = F.relu(self.bn1(self.conv1(x)))

        for i in range(18):
            # if i != 15 and i != 17:
            w1 = self.zs[2*i](self.hope1)
            w2 = self.zs[2*i+1](self.hope1)
            x = self.res_net[i](x, w1, w2)

        x = self.global_avg(x)
        x = self.final(x.view(-1,64))

        return x


class Bach:
    def __init__(self, config, model):
        super(Bach, self).__init__()
        self.config = config
        self.model = model

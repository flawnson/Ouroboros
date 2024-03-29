import torch
import numpy as np
import torch.nn.functional as F

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
        return None

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


class HyperNetwork(torch.nn.Module):

    def __init__(self, f_size=3, z_dim=64, out_size=16, in_size=16, device="cpu"):
        super(HyperNetwork, self).__init__()
        self.z_dim = z_dim
        self.f_size = f_size
        self.out_size = out_size
        self.in_size = in_size

        self.w1 = torch.nn.Parameter(torch.fmod(torch.randn((self.z_dim, self.in_size*self.z_dim)).to(device),2))
        self.b1 = torch.nn.Parameter(torch.fmod(torch.randn((self.in_size*self.z_dim)).to(device),2))

        self.w2 = torch.nn.Parameter(torch.fmod(torch.randn((self.z_dim, self.out_size*self.f_size*self.f_size)).to(device),2))
        self.b2 = torch.nn.Parameter(torch.fmod(torch.randn((self.out_size*self.f_size*self.f_size)).to(device),2))

    def forward(self, z):

        h_in = torch.matmul(z, self.w1) + self.b1
        h_in = h_in.view(self.in_size, self.z_dim)

        h_final = torch.matmul(h_in, self.w2) + self.b2
        kernel = h_final.view(self.out_size, self.in_size, self.f_size, self.f_size)

        return kernel


class IdentityLayer(torch.nn.Module):
    def forward(self, x):
        return x


class ResNetBlock(torch.nn.Module):

    def __init__(self, in_size=16, out_size=16, downsample=False):
        super(ResNetBlock,self).__init__()
        self.out_size = out_size
        self.in_size = in_size
        if downsample:
            self.stride1 = 2
            self.reslayer = torch.nn.Conv2d(in_channels=self.in_size, out_channels=self.out_size, stride=2, kernel_size=1)
        else:
            self.stride1 = 1
            self.reslayer = IdentityLayer()

        self.bn1 = torch.nn.BatchNorm2d(out_size)
        self.bn2 = torch.nn.BatchNorm2d(out_size)

    def forward(self, x, conv1_w, conv2_w):

        residual = self.reslayer(x)

        out = F.relu(self.bn1(F.conv2d(x, conv1_w, stride=self.stride1, padding=1)), inplace=True)
        out = self.bn2(F.conv2d(out, conv2_w, padding=1))

        out += residual

        out = F.relu(out)

        return out


class Embedding(torch.nn.Module):

    def __init__(self, z_num, z_dim, device):
        super(Embedding, self).__init__()

        self.z_list = torch.nn.ParameterList()
        self.z_num = z_num
        self.z_dim = z_dim

        h, k = self.z_num

        # Generating unique (random) layer embeddings to be passed into HyperNetwork
        for i in range(h*k):
            self.z_list.append(torch.nn.Parameter(torch.fmod(torch.randn(self.z_dim).to(device), 2)))

    def forward(self, hyper_net):
        ww = []
        k, h = self.z_num
        for i in range(h):
            w = []
            for j in range(k):
                w.append(hyper_net(self.z_list[i*k + j]))  # Wierd indexing to ensure the same layer embeddings not used more than once
            ww.append(torch.cat(w, dim=1))  # "Horizontal" concat of kernels generated by Hypernet
        return torch.cat(ww, dim=0)  # "Vertical" concat of kernels generated by Hypernet

    def _forward(self, hypernet):
        # MESSING WITH DIFFERENT ORDER OF CONCAT
        ww = []
        k, h = self.z_num
        for j in range(k):
            w = []
            for i in range(h):
                w.append(hypernet(self.z_list[i*k + j]))  # Wierd indexing to ensure the same layer embeddings not used more than once
            ww.append(torch.cat(w, dim=0))  # "Vertical" concat of kernels generated by Hypernet
        return torch.cat(ww, dim=1)  # "Horizontal" concat of kernels generated by Hypernet


class PrimaryNetwork(torch.nn.Module):

    def __init__(self, z_dim=64, device="cpu"):
        super(PrimaryNetwork, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(16)

        self.hope = HyperNetwork(z_dim=z_dim, device=device)

        # Each size value in each pair is actually 16x that value to correspond with the filter_size
        # 36 is 2x the number of filters which is 18, because there are 2 layers in each resnet
        self.zs_size = [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1],
                        [1, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2],
                        [2, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4]]

        self.zs_size = [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4]

        self.filter_size = [16, 16, 16, 16, 16, 16, 16, 32, 32, 32, 32, 32, 32, 64, 64, 64, 64, 64, 64]

        # ResNet Init
        self.res_net = torch.nn.ModuleList()
        for i, (in_size, out_size) in enumerate(zip(self.filter_size, self.filter_size[1:])):
            down_sample = False
            if i > 5 and i % 6 == 0:  # Downsampling to different output size as per the ResNet specification
                down_sample = True
            self.res_net.append(ResNetBlock(in_size, out_size, downsample=down_sample))

        # # Embedding Init
        # self.zs = torch.nn.ModuleList()
        # for i in range(len(self.zs_size)):
        #     self.zs.append(Embedding(self.zs_size[i], z_dim, device))

        # Embedding Init
        self.zs = [torch.nn.ModuleList(), torch.nn.ModuleList()]
        for embed_layer in self.zs:
            for zs1, zs2 in zip(self.zs_size, self.zs_size[1:]):
                embed_layer.append(Embedding([zs1, zs2], z_dim, device))

        self.global_avg = torch.nn.AvgPool2d(8)
        self.final = torch.nn.Linear(64,10)

    def forward(self, x):

        x = F.relu(self.bn1(self.conv1(x)))

        for i in range(len(self.res_net)):
            for idx, embed_layer in enumerate(self.zs):
                # The HyperNetwork output (kernel) gets passed into the Embedding layer
                # The Embedding layer aggregates the kernels and outputs the weights for the resnet layer
                w1 = self.zs[idx][i](self.hope)  # All even numbered Embedding layers are for weight 1
                w2 = self.zs[idx][i+1](self.hope)  # All odd numbered Embedding layers are for weight 2
                x = self.res_net[i](x, w1, w2)

        x = self.global_avg(x)
        x = self.final(x.view(-1,64))

        return x




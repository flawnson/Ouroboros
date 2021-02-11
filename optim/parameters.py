import torch
import random

from typing import *
from logzero import logger


class ModelParameters(object):
    def __init__(self, config, model, device):
        self.config = config
        self.device = device
        self.model = model
        #self._params = model.parameters()  # Default to PyTorch's parameters
        print("ModelParameters")
        print(self.model)
        self.num_params = self.model.num_params

        self.params = torch.eye(self.model.num_params, device=self.device) #takes a lot of memory I feel like

    def get_param(self, idx):
        return self.model.get_param(idx)
        
    def __len__(self):
        return self.num_params



    # @property
    # def params(self):
    #     return self._params
    #
    # @params.setter
    # def params(self, value):
    #     # params_data = torch.eye(self.model.num_params, device=self.device)
    #     index_list = list(range(self.model.num_params))
    #     # random.shuffle(params_data)
    #     # divide into training/val
    #     # split = int(len(params_data) * self.data_config["train_size"])
    #     # train_params = params_data[:split]
    #     # train_idx = index_list[:split]
    #     # test_params = params_data[split:]
    #     # test_idx = index_list[split:]
    #
    #     self._params = index_list

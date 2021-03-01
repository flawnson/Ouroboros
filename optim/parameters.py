import torch
import random
import numpy as np

from typing import *
from logzero import logger


class ModelParameters(object):
    """
    ModelParameters class is a wrapper for functionality concerning model parameters.
    """
    def __init__(self, config, model, device):
        """
        Initializes a ModelParameter class.

        Args:
            config: Configuration dictionary of the run.
            model: The Pytorch model to wrap.

        Attributes:
            config: Configuration dictionary of the run.
            model: The Pytorch model to wrap.
            num_params: Number of parameters in the model.
            params: A list of model param indices from 0 to num_params - 1.
        """
        self.config = config
        self.model = model
        #self._params = model.parameters()  # Default to PyTorch's parameters
        self.num_params = self.model.num_params

        #Not sure if self.params list should be a torch tensor
        self.params = torch.tensor(list(range(self.num_params))) #indices of all params: [1, 2, ......, num_params - 1]
        # logger.info("Model Structure: ")
        # logger.info(self.model)

    def get_param(self, idx):
        """
        Gets the parameter at the specified index in the model

        Args:
            idx: An integer value specifying the index value (between 0 and num_params - 1)

        Returns:
            A parameter at index idx as a torch tensor.
        """
        #idx is an int
        #idx -> one hot vector
        #convert to one hot on the fly
        values = [idx]
        n_values = np.max(values) + 1
        onehot = np.eye(n_values)[values][0] #not sure of dimensions (may want to print)
        print("Onehot: ", onehot)

        return self.model.get_param(onehot)

    def __len__(self):
        """
        Gets the number of parameters in the model.

        Returns:
            The number of parameters in the model (num_params).
        """
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

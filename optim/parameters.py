import torch

from typing import *
from logzero import logger


class ModelParameters(object):
    """
    ModelParameters class is a wrapper for functionality concerning model parameters.
    """
    def __init__(self, config: Dict, model: torch.nn.Module, device: torch.device):
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
        self.params = torch.tensor(list(range(self.num_params)), device=device) #indices of all params: [1, 2, ......, num_params - 1]
        # logger.info("Model Structure: ")
        # logger.info(self.model)
        self.device = device

    def to_onehot(self, idxs: torch.tensor):
        onehot = torch.zeros(self.num_params, device=self.device)
        onehot[idxs.item()] = 1
        # onehots = [torch.zeros(self.num_params, device=self.device)[idx.item()] for idx in idxs]  # Was testing different batch sizes
        return onehot

    def get_param(self, idx: int) -> float:
        """
        Gets the parameter at the specified index in the model

        Args:
            idx: An integer value specifying the index value (between 0 and num_params - 1)

        Returns:
            A parameter at index idx as a torch tensor.
        """
        return self.model.get_param(idx)

    def __len__(self):
        """
        Gets the number of parameters in the model.

        Returns:
            The number of parameters in the model (num_params).
        """
        return self.num_params


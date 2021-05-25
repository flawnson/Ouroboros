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
        self.num_params = self.model.num_params
        self.params = torch.tensor(list(range(self.num_params)), device=device) #indices of all params: [1, 2, ......, num_params - 1]
        #if subset is specified, select only a small portion of model params
        subset = config["data_config"]["param_subset"]
        if isinstance(subset, int):
            self.num_params = subset
            self.params = torch.tensor(list(range(subset)), device=device) #indices of all params: [1, 2, ......, subset]
            print("Param size: ", self.params.size())
            print("Num params: ", self.num_params)

        self.device = device

    def get_param(self, idx: int) -> float:
        """
        Gets the parameter at the specified index in the model

        Args:
            idx: An integer value specifying the index value (between 0 and num_params - 1)

        Returns:
            A parameter at index idx as a torch tensor.
        """
        return self.model.get_param(idx)

    def __getitem__(self, idx):
        """
        For enumeration purposes
        """
        return self.params[idx]

    def __len__(self):
        """
        Gets the number of parameters in the model.

        Returns:
            The number of parameters in the model (num_params).
        """
        return self.num_params

import torch

from typing import *
from logzero import logger

from models.augmented.quine import Quine
from utils.utilities import get_example_size
from utils.reduction import Reduction


class Classical(Quine, torch.nn.Module):
    def __init__(self, config: Dict, model: torch.nn.Module, dataset, device: torch.device):
        super(Classical, self).__init__(config, model, device)
        super(torch.nn.Module)
        self.model_aug_config = config["model_aug_config"]
        self.model = model
        self.dataset = dataset
        self.device = device

    def classical_input(self) -> torch.nn.Sequential:
        """
        Used to random project MNIST

        Returns:
            torch.nn.Sequential of projection layer

        """
        rand_proj_layer = torch.nn.Linear(get_example_size(self.dataset),
                                          self.model_aug_config["n_hidden"] // self.model_aug_config["n_inputs"],
                                          bias=False)  # Modify so there are half as many hidden units
        rand_proj_layer.weight.data = torch.tensor(self.reduction(get_example_size(self.dataset)), dtype=torch.float32)
        for p in rand_proj_layer.parameters():
            p.requires_grad_(False)
        return torch.nn.Sequential(rand_proj_layer)

    def classical_output(self) -> torch.nn.Sequential:
        # TODO: Make cleaner
        digit_predictor_layers = []
        current_layer = torch.nn.Linear(self.model_aug_config["n_hidden"], 10, bias=True)
        logsoftmax = torch.nn.LogSoftmax(dim=0) #should have no learnable weights
        digit_predictor_layers.append(current_layer)
        digit_predictor_layers.append(logsoftmax)
        self.param_list.append(current_layer.weight)
        self.param_names.append("dp_layer{}_weight".format(0))
        self.param_list.append(current_layer.bias)
        self.param_names.append("dp_layer{}_bias".format(0))
        return torch.nn.Sequential(*digit_predictor_layers)

    def regenerate(self):
        # Exists to satisfy Quine abstractmethod
        pass

    def forward(self, x: torch.tensor, y=None) -> torch.tensor:
        x = x.reshape(-1)  # Flatten MNIST input in place
        x = self.classical_input()(x)
        x = self.model(x)
        x = self.classical_output()(x)

        return x



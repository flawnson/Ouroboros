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

    def forward(self, x: torch.tensor, y=None) -> torch.tensor:
        x = x.reshape(-1)  # Flatten MNIST input in place
        x = self.classical_input()(x)
        x = self.model(x)

        return x



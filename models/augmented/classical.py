import torch

from typing import *
from logzero import logger


class Classical(torch.nn.Module):
    def __init__(self, config: Dict, model: torch.nn.Module, device: torch.device):
        super(Classical, self).__init__()
        self.config = config
        self.model = model
        self.device = device

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.model(x)

        return x



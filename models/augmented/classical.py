import torch

from typing import *
from logzero import logger


class Classical(torch.nn.Module):
    def __init__(self, config: Dict, model: torch.nn.Module, device: torch.device):
        super(Classical, self).__init__()
        super(torch.nn.Module)
        self.config = config
        self.model = model
        self.device = device

    def classical_output(self) -> torch.nn.Sequential:
        # # Log softmax in case of loss not being cross entropy
        # log_softmax = torch.nn.LogSoftmax(dim=0) # should have no learnable weights
        pass

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = x.reshape(x.shape[0], -1)  # Flatten MNIST input in place
        x = self.model(x)

        return x



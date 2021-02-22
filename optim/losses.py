"File for all implemented loss functions"

import torch
import torch.functional as F

from models.augmented.quine import Vanilla, Auxiliary
from models.augmented.ouroboros import Ouroboros

from typing import *
from logzero import logger


class Loss:

    def __init__(self, config: Dict, model: torch.nn.Module, predictions: torch.tensor, targets: torch.tensor):
        self.config = config
        self.model = model
        self.predictions = predictions
        self.targets = targets

    def sr_loss(self) -> float:
        loss_sr = (torch.linalg.norm(self.predictions["param"] - self.targets["param"], ord=2)) ** 2

        return loss_sr

    def task_loss(self) -> float:
        loss_task = F.nll_loss(self.predictions.unsqueeze(dim=0), self.targets)

        return loss_task

    def combined_loss(self) -> float:
        loss_combined = self.sr_loss() + self.config["lambda"] * self.task_loss()

        return loss_combined

    def new_loss(self) -> float:
        pass

    def get_loss(self) -> float:
        if isinstance(self.model, Vanilla):
            return self.sr_loss()
        elif isinstance(self.model, Auxiliary):
            return self.combined_loss()
        elif isinstance(self.model, Ouroboros):
            return self.new_loss()
        else:
            raise NotImplementedError("The specified loss is not implemented for this class")



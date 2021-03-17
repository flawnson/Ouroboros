"File for all implemented loss functions"

import torch
import torch.nn.functional as F

from models.augmented.quine import Quine, Vanilla, Auxiliary
from models.augmented.ouroboros import Ouroboros

from typing import *


class ClassicalLoss:
    def __init__(self, config: Dict, predictions: torch.tensor, targets: torch.tensor):
        self.config = config
        self.predictions = predictions
        self.targets = targets


class QuineLoss:

    def __init__(self, config: Dict, predictions: torch.tensor, targets: torch.tensor):
        self.config = config
        self.predictions = predictions
        self.targets = targets

    @staticmethod
    def calculate_relative_difference(x, y):
        return abs(x-y)/max(abs(x), abs(y))

    def relative_difference(self) -> float:
        rel_diff = relative_difference(self.predictions["param"].item(), self.targets["param"].item())
        return rel_diff

    def sr_loss(self) -> float:

        loss_sr = (torch.linalg.norm(self.predictions["param"] - self.targets["param"], ord=2)) ** 2

        return loss_sr

    def task_loss(self) -> float:
        loss_task = F.nll_loss(self.predictions["aux"].unsqueeze(dim=0), self.targets["aux"]) #create dictionary indices

        return loss_task

    def combined_loss(self) -> float:
        loss_combined = self.sr_loss() + self.config["run_config"]["lambda"] * self.task_loss()

        return loss_combined


def loss(config, model, predictions, targets) -> Union[Dict, float]:
    # if isinstance(model, torch.nn.Module):
    #     classical_loss = ClassicalLoss(config, predictions, targets)
    #     return classical_loss
    if isinstance(model, Quine):
        quine_loss = QuineLoss(config, predictions, targets)
        if type(model) is Vanilla:
            return quine_loss.sr_loss()
        elif type(model) is Auxiliary:
            return {"sr_loss": quine_loss.sr_loss(),
                    "task_loss": quine_loss.task_loss(),
                    "combined_loss": quine_loss.combined_loss()}
    elif isinstance(model, Ouroboros):
        pass
    else:
        raise NotImplementedError("The specified loss is not implemented for this class")

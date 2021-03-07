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
        loss_combined = self.sr_loss() + self.config["lambda"] * self.task_loss()

        return loss_combined

    def new_loss(self) -> float:
        pass

    def get_loss(self) -> float:
        if type(self.model) is Vanilla:
            print("vanilla model loss")
            return self.sr_loss()
        elif type(self.model) is Auxiliary:
            print("aux model loss")
            return {"sr_loss": self.sr_loss(),
                    "task_loss": self.task_loss(),
                    "combined_loss": self.combined_loss()}
        elif type(self.model) is Ouroboros:
            print("ouroboros model loss")
            return self.new_loss()
        else:
            raise NotImplementedError("The specified loss is not implemented for this class")

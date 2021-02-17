"File for all implemented loss functions"

import torch
import torch.functional as F

from models.augmented.quine import Vanilla, Auxiliary

from typing import *
from logzero import logger


class QuineLoss:

    def __init__(self, config: Dict, model: torch.nn.Module, predictions: torch.tensor, targets: torch.tensor):
        self.config = config
        self.model = model

    def sr_loss(self, predictions, targets):
        loss_sr = (torch.linalg.norm(predictions["param"] - targets["param"], ord=2)) ** 2

        return loss_sr

    def task_loss(self, predictions, targets):
        loss_task = F.nll_loss(predictions.unsqueeze(dim=0), targets)

        return loss_task

    def combined_loss(self):
        loss_combined = self.sr_loss() + self.config["lambda_val"] * self.task_loss()

        return loss_combined

    def get_loss(self):
        if isinstance(self.model, Vanilla):
            return self.sr_loss()
        elif isinstance(self.model, Auxiliary):
            return self.combined_loss()
        else:
            raise NotImplementedError("The specified loss is not implemented for this class")





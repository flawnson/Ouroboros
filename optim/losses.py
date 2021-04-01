"File for all implemented loss functions"

import torch
import torch.nn.functional as F

from typing import *
from logzero import logger
from torch.nn.functional import nll_loss, l1_loss, mse_loss, cross_entropy, binary_cross_entropy, kl_div

from models.augmented.quine import Quine, Vanilla, Auxiliary
from models.augmented.classical import Classical
from models.augmented.ouroboros import Ouroboros
from models.augmented.hypernetwork import PrimaryNetwork


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


def loss(config: Dict, model: torch.nn.Module, logits, targets) -> Union[Dict, float]:
    optim_config = config["optim_config"]
    if type(model) == Classical:
        return {"loss": eval(optim_config["loss_func"])(logits["aux"].unsqueeze(dim=0),
                                                        targets,
                                                        **optim_config["loss_kwargs"])}
    if isinstance(model, Quine):
        quine_loss = QuineLoss(config, logits, targets)
        if type(model) is Vanilla:
            return {"sr_loss": quine_loss.sr_loss()}
        elif type(model) is Auxiliary:
            return {"sr_loss": quine_loss.sr_loss(),
                    "task_loss": quine_loss.task_loss(),
                    "combined_loss": quine_loss.combined_loss()}
    elif isinstance(model, Ouroboros):
        pass
    elif type(model) == PrimaryNetwork:
        return {"loss": eval(optim_config["loss_func"])(logits, targets, **optim_config["loss_kwargs"])}
    else:
        raise NotImplementedError("The specified loss is not implemented for this class")

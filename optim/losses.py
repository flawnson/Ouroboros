"File for all implemented loss functions"

import torch
import numpy as np
import torch.nn.functional as F

from typing import *
from logzero import logger
from torch.nn.functional import nll_loss, l1_loss, mse_loss, cross_entropy, binary_cross_entropy, kl_div

from models.standard.linear_model import LinearModel
from models.augmented.quine import Quine, Vanilla, Auxiliary, SequentialAuxiliary
from models.augmented.classical import Classical
from models.augmented.ouroboros import Ouroboros
from models.augmented.hypernetwork import ResNetPrimaryNetwork


class QuineLoss:
    """
    QuineLoss class is a wrapper for functionality concerning model parameters.
    """
    def __init__(self, config: Dict, predictions: torch.tensor, targets: torch.tensor):
        """
        Initializes a QuineLoss class.

        Args:
            config: Configuration dictionary of the run.
            predictions: The model's predictions.
            targets: The ground truth target.

        Attributes:
            config: Configuration dictionary of the run.
            predictions: The model's predictions.
            targets: The ground truth target.
        """
        self.config = config
        self.predictions = predictions
        self.targets = targets

    def sr_loss(self) -> float:
        """
        Calculates the self replicating loss

        Returns:
            The loss value as a float.
        """
        loss_sr = (torch.linalg.norm(self.predictions["param"] - self.targets["param"], ord=2)) ** 2

        return loss_sr

    def sequential_sr_loss(self) -> float:
        if self.config["train_mode"] == "param":
            sequential_sr_loss = (torch.linalg.norm(self.predictions["param"] - self.targets["param"], ord=2)) ** 2
            return sequential_sr_loss
        else:
            return np.NAN

    def task_loss(self) -> float:
        """
        Calculates the auxiliary task loss.

        Returns:
            The loss value as a float.
        """
        loss_task = F.nll_loss(self.predictions["aux"], self.targets["aux"]) #create dictionary indices

        return loss_task

    def sequential_task_loss(self) -> float:
        #  Following example from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        if self.config["train_mode"] == "aux":
            sequential_task_loss = torch.nn.CrossEntropyLoss()(self.predictions["aux"], self.targets["aux"])
            return sequential_task_loss
        else:
            return np.NAN

    def combined_loss(self) -> float:
        """
        Calculates the combined loss (self replicating + task loss)

        Returns:
            The loss value as a float.
        """
        loss_combined = self.sr_loss() + self.config["run_config"]["lambda"] * self.task_loss()

        return loss_combined

    def sequential_combined_loss(self):
        sequential_loss_combined = self.sequential_sr_loss() + self.config["run_config"]["lambda"] + self.sequential_task_loss()

        return sequential_loss_combined


def loss(config: Dict, model: torch.nn.Module, logits, targets) -> Union[Dict, float]:
    """
    Creates and returns a dictionary with the loss values depending on the model type.

    Args:
        config: Configuration dictionary of the run.
        model: The Pytorch model of this run.
        logits: The predictions of the model.
        targets: The ground truth target.

    Returns:
        A dictionary with the calculated loss value(s) depending on the type of the model.
    """
    optim_config = config["optim_config"]
    if type(model) == Classical:
        return {"loss": eval(optim_config["loss_func"])(logits,
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
        elif type(model) is SequentialAuxiliary:
            return {"sr_loss": quine_loss.sequential_sr_loss(),
                    "task_loss": quine_loss.sequential_task_loss(),
                    "combined_loss": quine_loss.sequential_combined_loss()}
    elif isinstance(model, Ouroboros):
        pass
    elif type(model) == ResNetPrimaryNetwork:
        return {"loss": eval(optim_config["loss_func"])(logits, targets, **optim_config["loss_kwargs"])}
    else:
        try:
            return {"loss": eval(optim_config["loss_func"])(logits.unsqueeze(dim=0),
                                                            targets,
                                                            **optim_config["loss_kwargs"])}
        except Exception:
            raise NotImplementedError("The specified loss is not implemented for this class")

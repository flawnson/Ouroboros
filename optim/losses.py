"File for all implemented loss functions"

import torch
import torch.nn.functional as F

from typing import *
from logzero import logger
from torch.nn.functional import nll_loss, l1_loss, mse_loss, cross_entropy, binary_cross_entropy, kl_div

from models.standard.mlp_model import MLPModel
from models.augmented.quine import Quine, Vanilla, Auxiliary
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

    def task_loss(self) -> float:
        """
        Calculates the auxiliary task loss.

        Returns:
            The loss value as a float.
        """
        loss_task = F.nll_loss(self.predictions["aux"].unsqueeze(dim=0), self.targets["aux"]) #create dictionary indices

        return loss_task

    def combined_loss(self) -> float:
        """
        Calculates the combined loss (self replicating + task loss)

        Returns:
            The loss value as a float.
        """
        loss_combined = self.sr_loss() + self.config["run_config"]["lambda"] * self.task_loss()

        return loss_combined


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
        return {"loss": eval(optim_config["loss_func"])(logits.unsqueeze(dim=0),
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
    elif type(model) == ResNetPrimaryNetwork:
        return {"loss": eval(optim_config["loss_func"])(logits, targets, **optim_config["loss_kwargs"])}
    else:
        try:
            return {"loss": eval(optim_config["loss_func"])(logits.unsqueeze(dim=0),
                                                            targets,
                                                            **optim_config["loss_kwargs"])}
        except Exception:
            raise NotImplementedError("The specified loss is not implemented for this class")

import torch

from typing import *
from logzero import logger
from abc import ABC, abstractmethod

from torch.utils.data import DataLoader

from optim.algos import OptimizerObj, LRScheduler
from optim.losses import loss
from utils.scores import scores
from utils.logging import PTTBLogger
from utils.utilities import timed


class AbstractTrainer(ABC):

    def __init__(self, config: Dict, model: torch.nn.Module, dataset: Dict[str, DataLoader], device: torch.device):
        """
        Initializes a ClassicalTrainer class.

        Args:
            config: Configuration dictionary of the run.
            model: The Pytorch model.
            dataset: The data that will be used in the run.
            device: Device that training will be run on.

        Attributes:
            config: Configuration dictionary of the run.
            run_config: Run configurations of the run.
            model: The Pytorch model.
            optimizer: Optimizer used for the run.
            scheduler: Learning rate scheduler used for the run.
            tb_logger: Tensorboard logger.
            dataset: The data that will be used in the run.
            device: Device that training will be run on.
        """
        self.config = config
        self.run_config = config["run_config"]
        self.model = model
        self.optimizer = OptimizerObj(config, model).optim_obj
        self.scheduler = LRScheduler(config, self.optimizer).schedule_obj
        self.tb_logger = PTTBLogger(config)
        self.dataset = dataset
        self.device = device

    @abstractmethod
    def train(self):
        """
        Run data into model, collect output and target label for loss calculations.
        """
        pass

    @abstractmethod
    @torch.no_grad()
    def test(self):
        """
        Run data into model, collect output and target label for loss calculations.
        """
        pass

    def loss(self, predictions, targets):
        """
        Calculates loss based on predictions and targets.
        Args:
            predictions: Model output predictions.
            targets: Target labels corresponding to the model predictions.

        Returns:
            A dictionary of loss values depending on the model type.
        """
        return loss(self.config, self.model, predictions, targets)

    def score(self, predictions, targets):
        """
        Calculates score.
        Returns:
            A score dictionary.
        """
        return scores(self.config, predictions, targets, self.device)

    @abstractmethod
    def write(self, epoch: int):
        """
        Logs the loss and scores to Tensorboard.
        """
        logger.info(f"Running epoch: #{epoch}")

    @abstractmethod
    def reset(self):
        """
        Reset the temporary state values for epoch_data.
        """
        logger.info("States successfully reset for new epoch")

    @timed
    def run_train(self):
        """
        Main training loop.
        """
        pass

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
        pass

    @abstractmethod
    @torch.no_grad()
    def test(self):
        pass

    def loss(self, predictions, targets):
        return loss(self.config, self.model, predictions, targets)

    def score(self, predictions, targets):
        return scores(self.config, predictions, targets, self.device)

    @abstractmethod
    def write(self, epoch: int):
        logger.info(f"Running epoch: #{epoch}")

    @abstractmethod
    def reset(self):
        logger.info("States successfully reset for new epoch")

    @timed
    def run_train(self):
        pass

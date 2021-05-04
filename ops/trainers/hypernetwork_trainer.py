import torch
import numpy as np

from typing import *
from tqdm import trange
from logzero import logger

from torch.utils.data import DataLoader

from .abstract_trainer import AbstractTrainer
from optim.algos import OptimizerObj, LRScheduler
from optim.losses import loss
from utils.scores import scores
from utils.checkpoint import PTCheckpoint
from utils.logging import PTTBLogger
from utils.utilities import timed


class HyperNetworkTrainer(AbstractTrainer):
    def __init__(self, config: Dict, model: torch.nn.Module, dataset: torch.utils.data, device: torch.device):
        super(HyperNetworkTrainer, self).__init__(config, model, dataset, device)
        self.config = config
        self.data_config = config["data_config"]
        self.model = model
        self.dataset = dataset
        self.optimizer = OptimizerObj(config, self.model).optim_obj
        self.scheduler = LRScheduler(config, self.optimizer).schedule_obj
        self.tb_logger = PTTBLogger(config)
        self.checkpoint = PTCheckpoint(config)
        self.batch_data = {"running_loss": [0] * len(dataset),
                           "correct": [0] * len(dataset)}
        self.epoch_data = {"running_loss": [0] * len(dataset),
                           "correct": [0] * len(dataset)}
        self.device = device

    def train(self, data, batch_idx):
        inputs, labels = data
        inputs, labels = torch.autograd.Variable(inputs.to(self.device)), torch.autograd.Variable(labels.to(self.device))

        outputs = self.model(inputs)
        loss = self.loss(outputs, labels)

        if ((batch_idx + 1) % self.data_config["batch_size"]) == 0:
            loss["loss"].backward()
            self.optimizer.step()
            self.scheduler.step()
            self.epoch_data["running_loss"][0] += loss["loss"]

            self.batch_data["running_loss"][0] = 0.0
            self.optimizer.zero_grad()

            self.epoch_data["running_loss"][1] += self.batch_data["running_loss"][1]  # accumulate for epoch

        self.batch_data["running_loss"][0] += loss["loss"]

    def test(self, data, batch_idx):
        total = 0
        images, labels = data
        outputs = self.model(torch.autograd.Variable(images.to(self.device)))
        _, predicted = torch.max(outputs.to(self.device).data, 1)
        total += labels.size(0)
        self.batch_data["correct"][1] += (predicted == labels).sum()
        # self.epoch_data["accuracy"][1] = (100. * self.batch_data["correct"][1]) / total

    def score(self):
        return scores(self.config, self.dataset, self.epoch_data["correct"], self.device)

    def write(self, epoch, epoch_scores):
        logger.info(f"Total Loss value: {epoch_scores['acc'][0]}")
        logger.info(f"Total Loss value: {self.epoch_data['running_loss'][0]}")

        self.tb_logger.scalar_summary('loss (train)', self.epoch_data["running_loss"][0], epoch)
        self.tb_logger.scalar_summary('accuracy (train)', epoch_scores["acc"][0], epoch)

        self.tb_logger.scalar_summary('loss (test)', self.epoch_data["running_loss"][1], epoch)
        self.tb_logger.scalar_summary('accuracy (test)', epoch_scores["acc"][1], epoch)

        logger.info("Successfully wrote logs to tensorboard")

    def reset(self):
        for i in range(len(self.dataset)):
            self.epoch_data["running_loss"][i] = 0
            self.epoch_data["correct"][i] = 0

        logger.info("States successfully reset for new epoch")

    def run_train(self):
        if all(isinstance(dataloader, DataLoader) for dataloader in self.dataset.values()):
            for epoch in trange(0, self.run_config["num_epochs"], desc="Epochs"):
                logger.info(f"Epoch: {epoch}")

                for batch_idx, data in enumerate(self.dataset[list(self.dataset)[0]]):
                    logger.info(f"Running train batch: #{batch_idx}")
                    self.train(data, batch_idx)

                for batch_idx, data in enumerate(self.dataset[list(self.dataset)[1]]):
                    logger.info(f"Running test batch: #{batch_idx}")
                    self.test(data, batch_idx)

                # Scores cumulated and calculated per epoch, as done in Quine
                epoch_scores = self.score()

                self.checkpoint.checkpoint(self.config,
                                           epoch,
                                           self.model,
                                           self.epoch_data["loss"][0],
                                           self.optimizer)

                self.write(epoch, epoch_scores)
                self.reset()


class DualHyperNetworkTrainer(AbstractTrainer):
    def __init__(self, config, model, dataset, device):
        super(DualHyperNetworkTrainer, self).__init__(config, model, dataset, device)
        self.config = config
        self.data_config = config["data_config"]
        self.model = model
        self.dataset = dataset
        self.optimizer = OptimizerObj(config, self.model).optim_obj
        self.scheduler = LRScheduler(config, self.optimizer).schedule_obj
        self.tb_logger = PTTBLogger(config)
        self.checkpoint = PTCheckpoint(config)
        self.batch_data = {"running_loss": [0, 0],
                           "correct": [0, 0]}
        self.epoch_data = {"running_loss": [0, 0],
                           "correct": [0, 0]}
        self.device = device

    def train(self, data, batch_idx):
        inputs, labels = data
        inputs, labels = torch.autograd.Variable(inputs.to(self.device)), torch.autograd.Variable(labels.to(self.device))

        outputs = self.model(inputs)
        loss = self.loss(outputs, labels)

        if ((batch_idx + 1) % self.data_config["batch_size"]) == 0:
            loss["loss"].backward()
            self.optimizer.step()
            self.scheduler.step()
            self.epoch_data["running_loss"][0] += loss["loss"]

            self.batch_data["running_loss"][0] = 0.0
            self.optimizer.zero_grad()

            self.epoch_data["running_loss"][1] += self.batch_data["running_loss"][1]  # accumulate for epoch

        self.batch_data["running_loss"][0] += loss["loss"]

    def test(self, data, batch_idx):
        total = 0
        images, labels = data
        outputs = self.model(torch.autograd.Variable(images.to(self.device)))
        _, predicted = torch.max(outputs.to(self.device).data, 1)
        total += labels.size(0)
        self.batch_data["correct"][1] += (predicted == labels).sum()
        # self.epoch_data["accuracy"][1] = (100. * self.batch_data["correct"][1]) / total

    def score(self):
        return scores(self.config, self.dataset, self.epoch_data["correct"], self.device)

    def write(self, epoch, epoch_scores):
        logger.info(f"Total Loss value: {epoch_scores['acc'][0]}")
        logger.info(f"Total Loss value: {self.epoch_data['running_loss'][0]}")

        self.tb_logger.scalar_summary('accuracy (train)', epoch_scores["acc"][0], epoch)
        self.tb_logger.scalar_summary('loss (train)', self.epoch_data["running_loss"][0], epoch)

        self.tb_logger.scalar_summary('accuracy (train)', epoch_scores["acc"][1], epoch)
        self.tb_logger.scalar_summary('loss (train)', self.epoch_data["running_loss"][1], epoch)

    def reset(self):
        for i in range(len(self.dataset)):
            self.epoch_data["running_loss"][i] = 0
            self.epoch_data["correct"][i] = 0

    def run_train(self):
        if all(isinstance(dataloader, DataLoader) for dataloader in self.dataset.values()):
            for epoch in trange(0, self.run_config["num_epochs"], desc="Epochs"):
                logger.info(f"Epoch: {epoch}")

                for batch_idx, data in enumerate(self.dataset[list(self.dataset)[0]]):
                    logger.info(f"Running train batch: #{batch_idx}")
                    self.train(data, batch_idx)

                for batch_idx, data in enumerate(self.dataset[list(self.dataset)[1]]):
                    logger.info(f"Running test batch: #{batch_idx}")
                    self.test(data, batch_idx)

                # Scores cumulated and calculated per epoch, as done in Quine
                epoch_scores = self.score()

                self.checkpoint.checkpoint(self.config,
                                           epoch,
                                           self.model,
                                           self.epoch_data["loss"][0],
                                           self.optimizer)

                self.write(epoch, epoch_scores)
                self.reset()

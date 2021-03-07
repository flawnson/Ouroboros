import torch
import numpy as np
import pathlib

import torch.functional as F

from typing import *
from tqdm import trange
from logzero import logger
from abc import ABC, abstractmethod

from torch.utils.data import DataLoader
from torch.nn import Module

from optim.algos import OptimizerObj, LRScheduler
from optim.losses import Loss
from optim.parameters import ModelParameters
from utils.scores import scores
from utils.checkpoint import checkpoint
from utils.logging import Logger
from utils.utilities import timed, relative_difference


class AbstractTrainer(ABC):

    def __init__(self, config: Dict, model_wrapper: ModelParameters, dataset: Dict, device: torch.device):
        self.config = config
        self.run_config = config["run_config"]
        self.wrapper = model_wrapper
        self.params = torch.nn.ParameterList(self.wrapper.model.parameters())

        #Will self.params in OptimizerObj update the parameters in wrapper? If so, will it be by reference?
        #We want the parameters inside the wrapper to change too during training
        self.optimizer = OptimizerObj(config, self.params).optim_obj
        self.scheduler = LRScheduler(config, self.optimizer).schedule_obj

        log_path = self.run_config["log_dir"] + "/" + self.config["run_name"]
        pathlib.Path(log_path).mkdir(parents=True, exist_ok=True)
        self.logger = Logger(log_path)
        self.dataset = dataset
        self.device = device

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def test(self):
        pass

    @abstractmethod
    def loss(self):
        pass

    @abstractmethod
    def score(self):
        pass

    @abstractmethod
    def write(self):
        pass

    @abstractmethod
    def run_train(self):
        pass


class AuxTrainer(AbstractTrainer):
    # TODO: Consider designing Tuning and Benchmarking as subclasses of Trainer
    def __init__(self, config: Dict, model_wrapper: ModelParameters, dataset: Dict, device: torch.device):
        super(AuxTrainer, self).__init__(config, model_wrapper, dataset, device)
        self.config = config
        self.run_config = config["run_config"]
        self.wrapper = model_wrapper
        self.params = torch.nn.ParameterList(self.wrapper.model.parameters())
        self.optimizer = OptimizerObj(config, self.params).optim_obj
        self.scheduler = LRScheduler(config, self.optimizer).schedule_obj

        log_path = self.run_config["log_dir"] + "/" + self.config["run_name"]
        pathlib.Path(log_path).mkdir(parents=True, exist_ok=True)
        self.logger = Logger(log_path)
        self.dataset = dataset
        self.device = device
        self.epoch_loss = {"sr_loss": [0, 0], "task_loss": [0, 0], "combined_loss": [0, 0]}

    def train(self, data, param_idx, batch_idx):
        self.wrapper.model.train()
        self.optimizer.zero_grad()

        idx_vector = torch.squeeze(self.wrapper.to_onehot(param_idx)) #coordinate of the param in one hot vector form
        param = self.wrapper.model.get_param(param_idx)

        #Both predictions and targets will be dictionaries that hold two elements
        predictions = self.wrapper.model(idx_vector, data[0])
        targets = {"aux": data[-1], "param": param}

        loss = self.loss(predictions, targets)

        if ((batch_idx + 1) % self.config["data_config"]["batch_size"]) == 0:
            loss.backward()  # The combined loss is backpropagated right?
            self.optimizer.step()
            self.optimizer.zero_grad()

        self.epoch_loss["sr_loss"][0] += loss["sr_loss"]
        self.epoch_loss["task_loss"][0] += loss["task_loss"]
        self.epoch_loss["combined_loss"][0] += loss["combined_loss"]

        return predictions, targets

    @torch.no_grad()
    def test(self, data, param_idx):
        self.wrapper.model.eval()
        idx_vector = torch.squeeze(self.wrapper.to_onehot(param_idx)) #coordinate of the param in one hot vector form
        param = self.wrapper.model.get_param(param_idx)
        predictions = self.wrapper.model(idx_vector, data)
        targets = {"aux": data[-1], "param": param}

        loss = self.loss(predictions, targets)

        self.epoch_loss["sr_loss"][1] += loss["sr_loss"]
        self.epoch_loss["task_loss"][1] += loss["task_loss"]
        self.epoch_loss["combined_loss"][1] += loss["combined_loss"]

        return predictions, targets

    def loss(self, predictions, targets):

        ##NOTE: in nn-quine we had this:
        #in these fields, index 0 is training value, index 1 is validation value
        #loss_combined, avg_relative_error, loss_sr, loss_task, total_loss_sr, total_loss_task, total_loss_combined = [[0.0, 0.0] for i in range(7)]

        #Everything get's reset for the next epoch
        #loss values are batch loss, total_loss are epoch loss
        #Only total_loss values are logged to tensorboard
        ####
        return Loss(self.config, self.wrapper.model, predictions, targets).get_loss()

    def score(self, predictions, targets):
        return scores(self.config, predictions, targets, self.device)

    def write(self, epoch: int, scores: Dict):
        logger.info(f"Running epoch: #{epoch}")
        logger.info(f"Scores: {scores}")

        # Log values for training
        logger.scalar_summary('sr_loss (train)', self.epoch_loss["sr_loss"][0], epoch)
        logger.scalar_summary('task_loss (train)', self.epoch_loss["task_loss"][0], epoch)
        logger.scalar_summary('combined_loss (train)', self.epoch_loss["combined_loss"][0], epoch)

        # Log values for testing
        logger.scalar_summary('sr_loss (test)', self.epoch_loss["sr_loss"][1], epoch)
        logger.scalar_summary('task_loss (test)', self.epoch_loss["task_loss"][1], epoch)
        logger.scalar_summary('combined_loss (test)', self.epoch_loss["combined_loss"][1], epoch)

    @timed
    def run_train(self):
        if all(isinstance(dataloader, DataLoader) for dataloader in self.dataset.values()):
            for epoch in trange(0, self.run_config["num_epochs"], desc="Epochs"):
                logger.info(f"Epoch: {epoch}")
                for batch_idx, (data, param_idx) in enumerate(self.dataset[list(self.dataset)[0]]):
                    logger.info(f"Running train batch: #{batch_idx}")
                    predictions, targets = self.train(data, param_idx, batch_idx)

                train_scores = self.score(predictions, targets)
                logger.info(train_scores)

                for batch_idx, (data, param_idx) in enumerate(self.dataset[list(self.dataset)[1]]):
                    logger.info(f"Running test batch: #{batch_idx}")
                    predictions, targets = self.test(data, param_idx)

                # Scores cumulated and calculated per epoch, as done in Quine
                test_scores = self.score(predictions, targets)
                logger.info(test_scores)

            checkpoint(self.config, epoch, self.wrapper.model, 0.0, self.optimizer)
            self.write(epoch)

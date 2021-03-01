import torch
import numpy as np

import torch.functional as F

from typing import *
from tqdm import trange
from logzero import logger
from abc import ABC, abstractmethod

from torch.utils.data import DataLoader
from torch.nn import Module

from optim.algos import OptimizerObj, LRScheduler
from optim.losses import Loss
from utils.scores import Scores
from utils.checkpoint import checkpoint


class AbstractTrainer(ABC):

    def __init__(self, config: Dict, model: Module, dataset: Dict, device: torch.device):
        self.config = config
        self.run_config = config["run_config"]
        self.model = model
        self.params = torch.nn.ParameterList(self.model.parameters())
        self.params_data = torch.eye(self.model.num_params, device=device)
        self.optimizer = OptimizerObj(config, self.params).optim_obj
        self.scheduler = LRScheduler(config, self.optimizer).schedule_obj
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
    def __init__(self, config: Dict, model: Module, dataset: Dict, device: torch.device):
        super(AuxTrainer, self).__init__(config, model, dataset, device)
        self.config = config
        self.run_config = config["run_config"]
        self.model = model
        self.params = torch.nn.ParameterList(self.model.parameters())
        self.params_data = torch.eye(self.model.num_params, device=device)
        self.optimizer = OptimizerObj(config, self.params).optim_obj
        self.scheduler = LRScheduler(config, self.optimizer).schedule_obj
        self.dataset = dataset
        self.device = device

    def train(self, data, param_idx, batch_idx):
        self.model.train()
        self.optimizer.zero_grad()
        idx_vector = torch.squeeze(self.params_data[param_idx])  # Pulling out the nested tensor
        # param_idx should already be a tensor on the device when we initialized it using torch.eye
        param = self.model.get_param(param_idx)
        predictions = self.model(idx_vector, data)

        loss = self.loss(self.config, self.model, predictions, targets) #Incomplete? Parameters not passed in

        if ((batch_idx + 1) % self.config["data_config"]["batch_size"]) == 0:
            loss.backward()  # The combined loss is backpropagated right?
            self.optimizer.step()
            self.optimizer.zero_grad()

    @torch.no_grad()
    def test(self, data, param_idx):
        self.model.eval()
        idx_vector = torch.squeeze(self.params_data[param_idx])  # Pulling out the nested tensor
        param = self.model.get_param(param_idx)
        predictions = self.model(idx_vector, data)

        loss = self.loss(self.config, self.model, predictions, targets)
        return loss

    def loss(self, predictions, targets):

        ##NOTE: in nn-quine we had this:
        #in these fields, index 0 is training value, index 1 is validation value
        #loss_combined, avg_relative_error, loss_sr, loss_task, total_loss_sr, total_loss_task, total_loss_combined = [[0.0, 0.0] for i in range(7)]

        #Everything get's reset for the next epoch
        #loss values are batch loss, total_loss are epoch loss
        #Only total_loss values are logged to tensorboard
        ####
        return Loss(self.config, self.model, predictions, targets)

    def score(self):
        return Scores(self.config, self.device).get_scores()

    def write(self, epoch: int):
        logger.info(f"Running epoch: #{epoch}")

    def run_train(self):
        if all(isinstance(dataloader, DataLoader) for dataloader in self.dataset.values()):
            for epoch in trange(0, self.run_config["num_epochs"], desc="Epochs"):
                logger.info(f"Epoch: {epoch}")
                for batch_idx, (data, param_idx) in enumerate(self.dataset[list(self.dataset)[0]]):
                    self.train(data.to(self.device), param_idx, batch_idx)
                    scores = self.score()

                for batch_idx, (data, param_idx) in enumerate(self.dataset[list(self.dataset)[1]]):
                    self.test(data.to(self.device), param_idx)
                    scores = self.score()

            checkpoint(self.config, epoch, self.model, 0.0, self.optimizer)
            self.write(epoch)

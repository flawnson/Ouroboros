import torch
import numpy as np


from typing import *
from tqdm import trange
from logzero import logger
import torch.functional as F
from abc import ABC, abstractmethod

from torch.utils.data import DataLoader
from torch.nn import Module

from models.augmented.quine import Quine, Auxiliary, Vanilla
from models.augmented.classical import Classical
from optim.algos import OptimizerObj, LRScheduler
from optim.losses import Loss
from optim.parameters import ModelParameters
from utils.scores import scores
from utils.checkpoint import checkpoint
from utils.logging import TBLogger
from utils.utilities import timed


class AbstractTrainer(ABC):

    def __init__(self, config: Dict, model: torch.nn.Module, dataset: Dict, device: torch.device):
        self.config = config
        self.run_config = config["run_config"]
        self.model = model
        self.optimizer = OptimizerObj(config, model).optim_obj
        self.scheduler = LRScheduler(config, self.optimizer).schedule_obj
        self.tb_logger = TBLogger(self.config["log_dir"] + "/" + self.config["run_name"])
        self.dataset = dataset
        self.device = device

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    @torch.no_grad()
    def test(self):
        pass

    @abstractmethod
    def loss(self, predictions, targets):
        return Loss(self.config, self.model, predictions, targets).get_loss()

    @abstractmethod
    def score(self, predictions, targets):
        return scores(self.config, predictions, targets, self.device)

    @abstractmethod
    def write(self, epoch: int, **kwargs):
        logger.info(f"Running epoch: #{epoch}")

    @abstractmethod
    def run_train(self):
        pass


class ClassicalTrainer(AbstractTrainer):

    def __init__(self, config: Dict, model: torch.nn.Module, dataset: Dict, device: torch.device):
        super(ClassicalTrainer, self).__init__(config, model, dataset, device)
        self.config = config
        self.run_config = config["run_config"]
        self.model = model
        self.optimizer = OptimizerObj(config, self.model).optim_obj
        self.scheduler = LRScheduler(config, self.optimizer).schedule_obj
        self.tb_logger = TBLogger(self.config["log_dir"] + "/" + self.config["run_name"])
        self.dataset = dataset
        self.device = device

    def train(self, data, targets, batch_idx):
        self.model.train()
        self.optimizer.zero_grad()

        predictions = self.model(data)
        loss = self.loss(predictions, targets)

        if ((batch_idx + 1) % self.config["data_config"]["batch_size"]) == 0:
            loss["combined_loss"].backward()  # The combined loss is backpropagated right?
            self.optimizer.step()
            self.optimizer.zero_grad()

    @torch.no_grad()
    def test(self, data, targets):
        self.model.eval()

        predictions = self.model(data)
        loss = self.loss(predictions, targets)

    def loss(self, predictions, targets):
        return Loss(self.config, self.model, predictions, targets).get_loss()

    def score(self, predictions, targets):
        return scores(self.config, predictions, targets, self.device)

    def write(self, epoch: int):
        logger.info(f"Running epoch: #{epoch}")

    def run_train(self):
        if all(isinstance(dataloader, DataLoader) for dataloader in self.dataset.values()):
            for epoch in trange(0, self.run_config["num_epochs"], desc="Epochs"):
                logger.info(f"Epoch: {epoch}")
                self.train()

            checkpoint(self.config, epoch, self.model, 0.0, self.optimizer)


class VanillaTrainer(AbstractTrainer):
    def __init__(self, config: Dict, model_wrapper: ModelParameters, dataset: Dict, device: torch.device):
        super(VanillaTrainer, self).__init__(config, model_wrapper, dataset, device)
        self.config = config
        self.run_config = config["run_config"]
        self.wrapper = model_wrapper
        self.optimizer = OptimizerObj(config, self.wrapper.model).optim_obj
        self.scheduler = LRScheduler(config, self.optimizer).schedule_obj
        self.tb_logger = TBLogger(self.config["log_dir"] + "/" + self.config["run_name"])
        self.dataset = dataset
        self.device = device
        self.epoch_data = {"sr_loss": [0, 0],
                           "task_loss": [0, 0],
                           "combined_loss": [0, 0],
                           "correct": [0, 0]}  # First position for training scores, second position for test scores


class AuxiliaryTrainer(AbstractTrainer):
    # TODO: Consider designing Tuning and Benchmarking as subclasses of Trainer
    def __init__(self, config: Dict, model_wrapper: ModelParameters, dataset: Dict, device: torch.device):
        super(AuxiliaryTrainer, self).__init__(config, model_wrapper.model, dataset, device)
        self.config = config
        self.run_config = config["run_config"]
        self.wrapper = model_wrapper
        self.optimizer = OptimizerObj(config, self.wrapper.model).optim_obj
        self.scheduler = LRScheduler(config, self.optimizer).schedule_obj
        self.tb_logger = TBLogger(self.config["log_dir"] + "/" + self.config["run_name"])
        self.dataset = dataset
        self.device = device
        self.epoch_data = {"sr_loss": [0, 0],
                           "task_loss": [0, 0],
                           "combined_loss": [0, 0],
                           "correct": [0, 0]}  # First position for training scores, second position for test scores

    def train(self, data, param_idx, batch_idx):
        self.wrapper.model.train()
        self.optimizer.zero_grad()

        idx_vector = torch.squeeze(self.wrapper.to_onehot(param_idx)) #coordinate of the param in one hot vector form
        param = self.wrapper.model.get_param(param_idx)

        #Both predictions and targets will be dictionaries that hold two elements
        predictions: Dict = self.wrapper.model(idx_vector, data[0].to(self.device))
        aux_pred = predictions["aux"].argmax(keepdim=True)  # get the index of the max log-probability
        targets = {"aux": data[-1].to(self.device), "param": param}

        loss = self.loss(predictions, targets)

        if ((batch_idx + 1) % 16) == 0:
            loss["combined_loss"].backward()  # The combined loss is backpropagated right?
            self.optimizer.step()
            self.epoch_data["sr_loss"][0] = 0.0
            self.epoch_data["task_loss"][0] = 0.0
            self.epoch_data["combined_loss"][0] = 0.0
            self.optimizer.zero_grad()

        self.epoch_data["sr_loss"][0] += loss["sr_loss"]
        self.epoch_data["task_loss"][0] += loss["task_loss"]
        self.epoch_data["combined_loss"][0] += loss["combined_loss"]
        self.epoch_data["correct"][0] += aux_pred.eq(data[1].view_as(aux_pred)).sum().item()

        return predictions, targets

    @torch.no_grad()
    def test(self, data, param_idx):
        self.wrapper.model.eval()
        idx_vector = torch.squeeze(self.wrapper.to_onehot(param_idx)) #coordinate of the param in one hot vector form
        param = self.wrapper.model.get_param(param_idx)
        predictions = self.wrapper.model(idx_vector, data)
        aux_pred = predictions["aux"].argmax(keepdim=True)  # get the index of the max log-probability
        targets = {"aux": data[-1], "param": param}

        loss = self.loss(predictions, targets)

        self.epoch_data["sr_loss"][1] += loss["sr_loss"]
        self.epoch_data["task_loss"][1] += loss["task_loss"]
        self.epoch_data["combined_loss"][1] += loss["combined_loss"]
        self.epoch_data["correct"][1] += aux_pred.eq(data[1].view_as(aux_pred)).sum().item()

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
        return scores(self.config, self.dataset, self.epoch_data, self.device)

    def write(self, epoch: int, scores: Dict):
        logger.info(f"Running epoch: #{epoch}")
        logger.info(f"Scores: {scores}")

        # Log values for training
        self.tb_logger.scalar_summary('sr_loss (train)', self.epoch_data["sr_loss"][0], epoch)
        self.tb_logger.scalar_summary('task_loss (train)', self.epoch_data["task_loss"][0], epoch)
        self.tb_logger.scalar_summary('combined_loss (train)', self.epoch_data["combined_loss"][0], epoch)

        # Log values for testing
        self.tb_logger.scalar_summary('sr_loss (test)', self.epoch_data["sr_loss"][1], epoch)
        self.tb_logger.scalar_summary('task_loss (test)', self.epoch_data["task_loss"][1], epoch)
        self.tb_logger.scalar_summary('combined_loss (test)', self.epoch_data["combined_loss"][1], epoch)

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

                # Regeneration (per epoch) step if specified in config
                if self.run_config["regenerate"]: self.wrapper.model.regenerate()

            checkpoint(self.config, epoch, self.wrapper.model, 0.0, self.optimizer)
            self.write(epoch, train_scores)


def trainer(config, model, param_data, dataloaders, device):
    if isinstance(model, Auxiliary):
        return AuxiliaryTrainer(config, param_data, dataloaders, device).run_train()
    if isinstance(model, Vanilla):
        return VanillaTrainer(config, param_data, dataloaders, device).run_train()
    elif isinstance(model, Classical):
        return ClassicalTrainer(config, model, dataloaders, device).run_train()
    else:
        try:
            return AbstractTrainer(config, model, dataloaders, device).run_train()
        except Exception as e:
            logger.info(e)
            raise NotImplementedError(f"Model {model} has no trainer pipeline")










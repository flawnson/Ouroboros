import torch
import numpy as np


from typing import *
from tqdm import trange
from logzero import logger
import torch.functional as F
from abc import ABC, abstractmethod

from torch.utils.data import DataLoader
from torch.nn import Module

from models.standard.mlp_model import MLPModel
from models.augmented.quine import Quine, Auxiliary, Vanilla
from models.augmented.ouroboros import Godel
from models.augmented.classical import Classical
from models.augmented.hypernetwork import PrimaryNetwork
from optim.algos import OptimizerObj, LRScheduler
from optim.losses import loss
from optim.parameters import ModelParameters
from utils.scores import scores
from utils.checkpoint import checkpoint
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
        if all(isinstance(dataloader, DataLoader) for dataloader in self.dataset.values()):
            for epoch in trange(0, self.run_config["num_epochs"], desc="Epochs"):
                logger.info(f"Epoch: {epoch}")
                for batch_idx, (principal_data, complimentary_data) in enumerate(self.dataset[list(self.dataset)[0]]):
                    logger.info(f"Running train batch: #{batch_idx}")
                    logits, targets = self.train(principal_data, complimentary_data, batch_idx)

                for batch_idx, (principal_data, complimentary_data) in enumerate(self.dataset[list(self.dataset)[1]]):
                    logger.info(f"Running test batch: #{batch_idx}")
                    logits, targets = self.test(principal_data, complimentary_data, batch_idx)

                # Scores cumulated and calculated once per epoch, as done in Quine
                test_scores = self.score(logits, targets)
                logger.info(test_scores)

                checkpoint(self.config, epoch, self.model, 0.0, self.optimizer)
                self.write(epoch)
                self.reset()


class ClassicalTrainer(AbstractTrainer):

    def __init__(self, config: Dict, model: torch.nn.Module, dataset: Dict[str, DataLoader], device: torch.device):
        # TODO: Potentially unecessary usage of dictionary in the return of loss function
        super(ClassicalTrainer, self).__init__(config, model, dataset, device)
        self.config = config
        self.data_config = config["data_config"]
        self.run_config = config["run_config"]
        self.model = model
        self.optimizer = OptimizerObj(config, self.model).optim_obj
        self.scheduler = LRScheduler(config, self.optimizer).schedule_obj
        self.tb_logger = PTTBLogger(config)
        self.dataset = dataset
        self.device = device
        self.batch_data = {"loss": [0, 0],
                           "correct": [0, 0]}  # First position for training scores, second position for test scores
        self.epoch_data = {"loss": [0, 0],
                           "correct": [0, 0]}  # First position for training scores, second position for test scores

    def train(self, data, batch_idx):
        self.model.train()
        self.optimizer.zero_grad()

        logits = self.model(data[0])
        predictions = logits.argmax(keepdim=True)
        loss = self.loss(logits, data[1])

        if ((batch_idx + 1) % self.config["data_config"]["batch_size"]) == 0:
            loss["loss"].backward()
            self.optimizer.step()
            self.epoch_data["loss"][0] += self.batch_data["loss"][0]
            self.batch_data["loss"][0] = 0.0
            self.optimizer.zero_grad()

        self.epoch_data["correct"][0] += predictions.eq(data[1].view_as(predictions)).sum().item()

        self.batch_data["loss"][0] += loss["loss"].item()
        return logits, data[1]

    @torch.no_grad()
    def test(self, data, batch_idx):
        self.model.eval()

        logits = self.model(data[0])
        predictions = logits.argmax(keepdim=True)
        loss = self.loss(logits, data[1])

        if ((batch_idx + 1) % self.data_config["batch_size"]) == 0:
            self.epoch_data["loss"][1] += self.batch_data["loss"][1]
            self.batch_data["loss"][1] = 0.0

        self.epoch_data["correct"][1] += predictions.eq(data[1].view_as(predictions)).sum().item()
        self.batch_data["loss"][1] += loss["loss"].item()
        return logits, data[1]

    def loss(self, predictions, targets) -> Dict:
        return loss(self.config, self.model, predictions, targets)

    def score(self) -> Dict:
        return scores(self.config, self.dataset, self.epoch_data["correct"], self.device)

    def write(self, epoch: int, scores: Dict, train_epoch_length: int, test_epoch_length: int):
        logger.info(f"Train scores, Test scores: {scores}")

        logger.info(f"Total Loss value: {self.epoch_data['loss'][0]}")
        logger.info(f"Train epoch length: {train_epoch_length}")
        logger.info(f"Test epoch length: {test_epoch_length}")
        logger.info(f"Batch size: {self.data_config['batch_size']}")

        # Log values for training
        #PRINT STATEMENTS DON'T WORK???
        print("Loss value: ", self.epoch_data["loss"][0])
        print("Train epoch length: ", train_epoch_length)
        print("Batch size: ", self.data_config["batch_size"])
        train_loss = self.epoch_data["loss"][0] / (train_epoch_length // self.data_config["batch_size"])
        self.tb_logger.scalar_summary('loss (train)', train_loss, epoch)
        self.tb_logger.scalar_summary('scores (train)', scores["acc"][0], epoch)

        # Log values for testing
        test_loss = self.epoch_data["loss"][1] / (test_epoch_length // self.data_config["batch_size"])
        self.tb_logger.scalar_summary('loss (test)', test_loss, epoch)
        self.tb_logger.scalar_summary('scores (test)', scores["acc"][1], epoch)

        logger.info("Successfully wrote logs to tensorboard")

    def reset(self):
        self.epoch_data["loss"][0] = 0
        self.epoch_data["correct"][0] = 0

        self.epoch_data["loss"][1] = 0
        self.epoch_data["correct"][1] = 0

        logger.info("States successfully reset for new epoch")

    @timed
    def run_train(self):
        if all(isinstance(dataloader, DataLoader) for dataloader in self.dataset.values()):
            for epoch in trange(0, self.run_config["num_epochs"], desc="Epochs"):
                logger.info(f"Epoch: {epoch}")

                train_epoch_length = len(self.dataset[list(self.dataset)[0]])
                for batch_idx, data in enumerate(self.dataset[list(self.dataset)[0]]):
                    logger.info(f"Running train batch: #{batch_idx}")
                    logits, targets = self.train(data, batch_idx)

                test_epoch_length = len(self.dataset[list(self.dataset)[1]])
                for batch_idx, data in enumerate(self.dataset[list(self.dataset)[1]]):
                    logger.info(f"Running test batch: #{batch_idx}")
                    logits, targets = self.test(data, batch_idx)

                checkpoint(self.config, epoch, self.model, 0.0, self.optimizer)

                epoch_scores = self.score()
                self.write(epoch, epoch_scores, train_epoch_length, test_epoch_length)
                self.reset()


class VanillaTrainer(AbstractTrainer):
    def __init__(self, config: Dict, model_wrapper: ModelParameters, dataset: Dict[str, DataLoader], device: torch.device):
        super(VanillaTrainer, self).__init__(config, model_wrapper.model, dataset, device)
        self.config = config
        self.run_config = config["run_config"]
        self.data_config = config["data_config"]
        self.wrapper = model_wrapper
        self.optimizer = OptimizerObj(config, self.wrapper.model).optim_obj
        self.scheduler = LRScheduler(config, self.optimizer).schedule_obj
        self.tb_logger = PTTBLogger(config)
        self.dataset = dataset
        self.device = device
        self.batch_data = {"sr_loss": [0, 0]}  # First position for training scores, second position for test scores
        self.epoch_data = {"sr_loss": [0, 0]}

    def train(self, param_idx, batch_idx):
        self.wrapper.model.train()
        self.optimizer.zero_grad()

        idx_vector = torch.squeeze(self.wrapper.to_onehot(param_idx)) #coordinate of the param in one hot vector form
        param = self.wrapper.model.get_param(param_idx)

        predictions = {"param": self.wrapper.model(idx_vector)}
        targets = {"param": param}

        loss = self.loss(predictions, targets)

        if ((batch_idx + 1) % self.data_config["batch_size"]) == 0:
            loss["sr_loss"].backward()  # The combined loss is backpropagated right?
            self.optimizer.step()
            self.epoch_data["sr_loss"][0] += self.batch_data["sr_loss"][0] #accumulate for epoch
            self.batch_data["sr_loss"][0] = 0.0
            self.optimizer.zero_grad()

        self.batch_data["sr_loss"][0] += loss["sr_loss"].item()
        return predictions, targets

    @torch.no_grad()
    def test(self, param_idx, batch_idx):
        self.wrapper.model.eval()
        idx_vector = torch.squeeze(self.wrapper.to_onehot(param_idx)) #coordinate of the param in one hot vector form
        param = self.wrapper.model.get_param(param_idx)
        predictions = {"param": self.wrapper.model(idx_vector)}
        targets = {"param": param}

        loss = self.loss(predictions, targets)

        if ((batch_idx + 1) % self.data_config["batch_size"]) == 0:
            self.epoch_data["sr_loss"][1] += self.batch_data["sr_loss"][1] #accumulate for epoch
            self.batch_data["sr_loss"][1] = 0.0

        self.batch_data["sr_loss"][1] += loss["sr_loss"].item()
        return predictions, targets

    def loss(self, predictions, targets):
        return loss(self.config, self.wrapper.model, predictions, targets)

    def score(self):
        # Vanilla doesn't have any scores yet
        # return scores(self.config, self.dataset, self.epoch_data, self.device)
        pass

    def write(self, epoch: int, train_epoch_length: int, test_epoch_length: int):

        # Log values for training
        actual_train_loss = self.epoch_data["sr_loss"][0] / (train_epoch_length // self.data_config["batch_size"])
        self.tb_logger.scalar_summary('sr_loss (train)', actual_train_loss, epoch)

        # Log values for testing
        actual_test_loss = self.epoch_data["sr_loss"][1] / (test_epoch_length // self.data_config["batch_size"])
        self.tb_logger.scalar_summary('sr_loss (test)', actual_test_loss, epoch)

        logger.info("Successfully wrote logs to tensorboard")

    def reset(self):
        self.epoch_data["sr_loss"][0] = 0

        self.epoch_data["sr_loss"][1] = 0

        logger.info("States successfully reset for new epoch")

    @timed
    def run_train(self):
        if all(isinstance(dataloader, DataLoader) for dataloader in self.dataset.values()):
            for epoch in trange(0, self.run_config["num_epochs"], desc="Epochs"):
                logger.info(f"Epoch: {epoch}")

                train_epoch_length = len(self.dataset[list(self.dataset)[0]])
                for batch_idx, param_idx in enumerate(self.dataset[list(self.dataset)[0]]):
                    logger.info(f"Running train batch: #{batch_idx}")
                    predictions, targets = self.train(param_idx, batch_idx)

                test_epoch_length = len(self.dataset[list(self.dataset)[1]])
                for batch_idx, param_idx in enumerate(self.dataset[list(self.dataset)[1]]):
                    logger.info(f"Running test batch: #{batch_idx}")
                    predictions, targets = self.test(param_idx, batch_idx)

                #SCORING IS NOT DONE FOR VANILLA
                # Scores cumulated and calculated per epoch, as done in Quine
                #test_scores = self.score(predictions, targets)
                #logger.info(test_scores)

                # Regeneration (per epoch) step if specified in config
                # if self.run_config["regenerate"]: self.wrapper.model.regenerate()

                checkpoint(self.config, epoch, self.wrapper.model, 0.0, self.optimizer)
                self.write(epoch, train_epoch_length, test_epoch_length)
                self.reset()


class AuxiliaryTrainer(AbstractTrainer):
    # TODO: Consider designing Tuning and Benchmarking as subclasses of Trainer
    def __init__(self, config: Dict, model_wrapper: ModelParameters, dataset: Dict[str, DataLoader], device: torch.device):
        super(AuxiliaryTrainer, self).__init__(config, model_wrapper.model, dataset, device)
        self.config = config
        self.run_config = config["run_config"]
        self.data_config = config["data_config"]
        self.wrapper = model_wrapper
        self.optimizer = OptimizerObj(config, self.wrapper.model).optim_obj
        self.scheduler = LRScheduler(config, self.optimizer).schedule_obj
        self.tb_logger = PTTBLogger(config)
        self.dataset = dataset
        self.device = device
        self.batch_data = {"sr_loss": [0, 0],
                           "task_loss": [0, 0],
                           "combined_loss": [0, 0],
                           "correct": [0, 0]}  # First position for training scores, second position for test scores
        self.epoch_data = {"sr_loss": [0, 0],
                           "task_loss": [0, 0],  # The aux loss, original implementation is nll_loss
                           "combined_loss": [0, 0],
                           "correct": [0, 0]}  # First position for training scores, second position for test scores

    def train(self, data, param_idx, batch_idx):
        self.wrapper.model.train()
        self.optimizer.zero_grad()

        idx_vector = torch.squeeze(self.wrapper.to_onehot(param_idx)) #coordinate of the param in one hot vector form
        param = self.wrapper.model.get_param(param_idx)

        # Both predictions and targets will be dictionaries that hold two elements
        predictions = {"param": self.wrapper.model(idx_vector, data[0].to(self.device))[0],
                       "aux": self.wrapper.model(idx_vector, data[0].to(self.device))[1]}
        aux_pred = predictions["aux"].argmax(keepdim=True)  # get the index of the max log-probability
        targets = {"param": param, "aux": data[-1].to(self.device)}

        loss = self.loss(predictions, targets)

        if ((batch_idx + 1) % self.data_config["batch_size"]) == 0:
            loss["combined_loss"].backward()  # The combined loss is backpropagated right?
            self.optimizer.step()
            self.epoch_data["sr_loss"][0] += self.batch_data["sr_loss"][0] #accumulate for epoch
            self.epoch_data["task_loss"][0] += self.batch_data["task_loss"][0] #accumulate for epoch
            self.epoch_data["combined_loss"][0] += self.batch_data["combined_loss"][0] #accumulate for epoch

            self.batch_data["sr_loss"][0] = 0.0
            self.batch_data["task_loss"][0] = 0.0
            self.batch_data["combined_loss"][0] = 0.0
            self.optimizer.zero_grad()

        self.batch_data["sr_loss"][0] += loss["sr_loss"].item()
        self.batch_data["task_loss"][0] += loss["task_loss"].item()
        self.batch_data["combined_loss"][0] += loss["combined_loss"].item()
        self.epoch_data["correct"][0] += aux_pred.eq(data[1].view_as(aux_pred)).sum().item()

        return predictions, targets

    @torch.no_grad()
    def test(self, data, param_idx, batch_idx):
        self.wrapper.model.eval()
        idx_vector = torch.squeeze(self.wrapper.to_onehot(param_idx)) #coordinate of the param in one hot vector form
        param = self.wrapper.model.get_param(param_idx)
        outputs = {"param": self.wrapper.model(idx_vector, data[0].to(self.device))[0],
                   "aux": self.wrapper.model(idx_vector, data[0].to(self.device))[1]}
        aux_pred = outputs["aux"].argmax(keepdim=True)  # get the index of the max log-probability
        targets = {"aux": data[-1], "param": param}

        loss = self.loss(outputs, targets)

        if ((batch_idx + 1) % self.data_config["batch_size"]) == 0:
            self.epoch_data["sr_loss"][1] += self.batch_data["sr_loss"][1] #accumulate for epoch
            self.epoch_data["task_loss"][1] += self.batch_data["task_loss"][1] #accumulate for epoch
            self.epoch_data["combined_loss"][1] += self.batch_data["combined_loss"][1] #accumulate for epoch

            self.batch_data["sr_loss"][1] = 0.0
            self.batch_data["task_loss"][1] = 0.0
            self.batch_data["combined_loss"][1] = 0.0

        self.batch_data["sr_loss"][1] += loss["sr_loss"].item()
        self.batch_data["task_loss"][1] += loss["task_loss"].item()
        self.batch_data["combined_loss"][1] += loss["combined_loss"].item()
        self.epoch_data["correct"][1] += aux_pred.eq(data[1].view_as(aux_pred)).sum().item()

        return outputs, targets

    def loss(self, predictions, targets):
        return loss(self.config, self.wrapper.model, predictions, targets)

    def score(self):
        return scores(self.config, self.dataset, self.epoch_data["correct"], self.device)

    def write(self, epoch: int, scores: Dict, train_epoch_length: int, test_epoch_length: int):
        logger.info(f"Train scores, Test scores: {scores}")

        # Log values for training
        actual_sr_train_loss = self.epoch_data["sr_loss"][0] / (train_epoch_length // self.data_config["batch_size"])
        actual_task_train_loss = self.epoch_data["task_loss"][0] / (train_epoch_length // self.data_config["batch_size"])
        actual_combined_train_loss = self.epoch_data["combined_loss"][0] / (train_epoch_length // self.data_config["batch_size"])
        self.tb_logger.scalar_summary('sr_loss (train)', actual_sr_train_loss, epoch)
        self.tb_logger.scalar_summary('task_loss (train)', actual_task_train_loss, epoch)
        self.tb_logger.scalar_summary('combined_loss (train)', actual_combined_train_loss, epoch)
        self.tb_logger.scalar_summary('scores (train)', scores["acc"][0], epoch)

        # Log values for testing
        actual_sr_test_loss = self.epoch_data["sr_loss"][1] / (test_epoch_length // self.data_config["batch_size"])
        actual_task_test_loss = self.epoch_data["task_loss"][1] / (test_epoch_length // self.data_config["batch_size"])
        actual_combined_test_loss = self.epoch_data["combined_loss"][1] / (test_epoch_length // self.data_config["batch_size"])
        self.tb_logger.scalar_summary('sr_loss (test)', actual_sr_test_loss, epoch)
        self.tb_logger.scalar_summary('task_loss (test)', actual_task_test_loss, epoch)
        self.tb_logger.scalar_summary('combined_loss (test)', actual_combined_test_loss, epoch)
        self.tb_logger.scalar_summary('scores (test)', scores["acc"][1], epoch)

        logger.info("Successfully wrote logs to tensorboard")

    def reset(self):
        self.epoch_data["sr_loss"][0] = 0
        self.epoch_data["task_loss"][0] = 0
        self.epoch_data["combined_loss"][0] = 0
        self.epoch_data["correct"][0] = 0

        self.epoch_data["sr_loss"][1] = 0
        self.epoch_data["task_loss"][1] = 0
        self.epoch_data["combined_loss"][1] = 0
        self.epoch_data["correct"][1] = 0

        logger.info("States successfully reset for new epoch")

    @timed
    def run_train(self):
        if all(isinstance(dataloader, DataLoader) for dataloader in self.dataset.values()):
            for epoch in trange(0, self.run_config["num_epochs"], desc="Epochs"):
                logger.info(f"Epoch: {epoch}")

                train_epoch_length = len(self.dataset[list(self.dataset)[0]])
                for batch_idx, (data, param_idx) in enumerate(self.dataset[list(self.dataset)[0]]):
                    logger.info(f"Running train batch: #{batch_idx}")
                    outputs, targets = self.train(data, param_idx, batch_idx)

                test_epoch_length = len(self.dataset[list(self.dataset)[1]])
                for batch_idx, (data, param_idx) in enumerate(self.dataset[list(self.dataset)[1]]):
                    logger.info(f"Running test batch: #{batch_idx}")
                    outputs, targets = self.test(data, param_idx, batch_idx)

                # Scores cumulated and calculated per epoch, as done in Quine
                epoch_scores = self.score()

                # Regeneration (per epoch) step if specified in config
                if self.run_config["regenerate"]: self.wrapper.model.regenerate()

                checkpoint(self.config, epoch, self.wrapper.model, 0.0, self.optimizer)
                self.write(epoch, epoch_scores, train_epoch_length, test_epoch_length)
                self.reset()


class HyperNetworkTrainer(AbstractTrainer):
    def __init__(self, config, model, dataset, device):
        super(HyperNetworkTrainer, self).__init__(config, model, dataset, device)
        self.config = config
        self.data_config = config["data_config"]
        self.model = model
        self.dataset = dataset
        self.optimizer = OptimizerObj(config, self.model).optim_obj
        self.scheduler = LRScheduler(config, self.optimizer).schedule_obj
        self.tb_logger = PTTBLogger(config)
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

        self.tb_logger.scalar_summary('loss (train)', self.epoch_data["running_loss"][0], epoch)
        self.tb_logger.scalar_summary('accuracy (train)', epoch_scores["acc"][0], epoch)

        self.tb_logger.scalar_summary('loss (test)', self.epoch_data["running_loss"][1], epoch)
        self.tb_logger.scalar_summary('accuracy (test)', epoch_scores["acc"][1], epoch)

        logger.info("Successfully wrote logs to tensorboard")

    def reset(self):
        self.epoch_data["running_loss"][0] = 0
        self.epoch_data["correct"][0] = 0

        self.epoch_data["running_loss"][1] = 0
        self.epoch_data["correct"][1] = 0

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

                checkpoint(self.config, epoch, self.model, 0.0, self.optimizer)
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
        self.epoch_data["running_loss"][0] = 0
        self.epoch_data["correct"][0] = 0

        self.epoch_data["running_loss"][1] = 0
        self.epoch_data["correct"][1] = 0

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

                checkpoint(self.config, epoch, self.model, 0.0, self.optimizer)
                self.write(epoch, epoch_scores)
                self.reset()


def trainer(config: Dict, model: torch.nn.Module, param_data: torch.nn.Module, dataloaders: List[torch.utils.data.DataLoader], device: torch.device):
    if isinstance(model, (Classical, MLPModel)):
        return ClassicalTrainer(config, model, dataloaders, device).run_train()
    elif isinstance(model, Auxiliary):
        return AuxiliaryTrainer(config, param_data, dataloaders, device).run_train()
    elif isinstance(model, Vanilla):
        return VanillaTrainer(config, param_data, dataloaders, device).run_train()
    elif isinstance(model, PrimaryNetwork):
        return HyperNetworkTrainer(config, model, dataloaders, device).run_train()
    elif isinstance(model, Godel):
        return DualHyperNetworkTrainer(config, model, dataloaders, device).run_train()
    else:
        try:
            return AbstractTrainer(config, model, dataloaders, device).run_train()
        except Exception as e:
            logger.info(e)
            raise NotImplementedError(f"Model {model} has no trainer pipeline")

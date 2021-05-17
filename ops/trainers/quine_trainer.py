import torch

from typing import *
from tqdm import trange
from logzero import logger

from .abstract_trainer import AbstractTrainer
from torch.utils.data import DataLoader
from optim.algos import OptimizerObj, LRScheduler
from optim.losses import loss
from optim.parameters import ModelParameters
from utils.scores import scores
from utils.checkpoint import PTCheckpoint
from utils.logging import PTTBLogger
from utils.utilities import timed


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
        self.checkpoint = PTCheckpoint(config)
        self.dataset = dataset
        self.device = device
        self.batch_data = {"sr_loss": [0] * len(dataset)}  # First position for training scores, second position for test scores
        self.epoch_data = {"sr_loss": [0] * len(dataset)}

    def train(self, param_idx, batch_idx):
        self.wrapper.model.train()
        self.optimizer.zero_grad()

        idx_vector = torch.squeeze(self.wrapper.model.to_onehot(param_idx)) #coordinate of the param in one hot vector form
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
        idx_vector = torch.squeeze(self.wrapper.model.to_onehot(param_idx)) #coordinate of the param in one hot vector form
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
        for i in range(len(self.dataset)):
            self.epoch_data["sr_loss"][i] = 0

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

                self.checkpoint.checkpoint(self.config,
                                           epoch,
                                           self.wrapper.model,
                                           self.epoch_data["sr_loss"][0],
                                           self.optimizer)

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
        self.checkpoint = PTCheckpoint(config)
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

        idx_vector = torch.squeeze(self.wrapper.model.to_onehot(param_idx)) #coordinate of the param in one hot vector form
        param = self.wrapper.model.get_param(param_idx)

        # Both predictions and targets will be dictionaries that hold two elements
        output = self.wrapper.model(idx_vector, data[0].to(self.device))
        predictions = {"param": output[0],
                       "aux": output[1]}
        aux_pred = predictions["aux"].argmax(keepdim=True)  # get the index of the max log-probability
        targets = {"param": param.to(self.device), "aux": data[-1].to(self.device)}

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
        idx_vector = torch.squeeze(self.wrapper.model.to_onehot(param_idx)) #coordinate of the param in one hot vector form
        param = self.wrapper.model.get_param(param_idx)
        test_output = self.wrapper.model(idx_vector, data[0].to(self.device))
        outputs = {"param": test_output[0],
                   "aux": test_output[1]}
        aux_pred = outputs["aux"].argmax(keepdim=True)  # get the index of the max log-probability
        targets = {"aux": data[-1].to(self.device), "param": param.to(self.device)}

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
        for i in range(len(self.dataset)):
            self.epoch_data["sr_loss"][i] = 0
            self.epoch_data["task_loss"][i] = 0
            self.epoch_data["combined_loss"][i] = 0
            self.epoch_data["correct"][i] = 0

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

                self.checkpoint.checkpoint(self.config,
                                           epoch,
                                           self.wrapper.model,
                                           self.epoch_data["sr_loss"][0],
                                           self.optimizer)

                self.write(epoch, epoch_scores, train_epoch_length, test_epoch_length)
                self.reset()

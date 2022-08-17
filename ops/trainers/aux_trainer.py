
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
        self.epoch_data = {"sr_loss": [0, 0],
                           "task_loss": [0, 0],  # The aux loss, original implementation is nll_loss
                           "combined_loss": [0, 0],
                           "correct": [0, 0],
                           "total": [0, 0],  # First position for training scores, second position for test scores
                           "predictions": [[], []],
                           "targets": [[], []]}  # Only accumulate outputs and targets for aux data
        self.param_idx_map = dict({})  # Maps param_idx to value, to be used in regeneration

    def train(self, data, param_idxs):
        self.wrapper.model.train()
        self.optimizer.zero_grad()

        data[0] = data[0].to(self.device)
        data[1] = data[1].to(self.device)
        param_idxs = param_idxs.to(self.device)

        # Create onehot vectors and parameter indexes for the entire batch
        idx_vectors = []
        params = []
        for param_idx in param_idxs:
            idx_vector = torch.squeeze(self.wrapper.model.to_onehot(param_idx)) # coordinate of the param in one hot vector form
            param = self.wrapper.model.get_param(param_idx)
            idx_vectors.append(idx_vector)
            params.append(param)
        idx_vectors = torch.stack((idx_vectors)).to(self.device)
        params = torch.tensor(params, device=self.device)

        # Both predictions and targets will be dictionaries that hold two elements
        logits = self.wrapper.model(idx_vectors, data[0])
        predictions = {"param": logits[0],
                       "aux": logits[1]}
        for i, param_idx in enumerate(param_idxs):
            self.param_idx_map[param_idx.item()] = logits[0][i]
        aux_pred = torch.argmax(predictions["aux"], dim=1) # get the index of the max log-probability
        targets = {"param": params, "aux": data[-1]}

        loss = self.loss(predictions, targets)

        self.epoch_data["total"][0] += data[0].shape[0] #accumulate total number of samples in this batch
        self.epoch_data["correct"][0] += aux_pred.eq(data[1].view_as(aux_pred)).sum().item()
        self.epoch_data["sr_loss"][0] += loss["sr_loss"].item()  # accumulate
        self.epoch_data["task_loss"][0] += loss["task_loss"].item()  # accumulate
        self.epoch_data["combined_loss"][0] += loss["combined_loss"].item()  # accumulate
        self.epoch_data["predictions"][0] += logits[-1].cpu().detach().tolist()
        self.epoch_data["targets"][0] += data[-1].cpu().detach().tolist()

        loss["combined_loss"].backward()
        self.optimizer.step()

        return predictions, targets

    @torch.no_grad()
    def test(self, data, param_idxs):
        self.wrapper.model.eval()

        data[0] = data[0].to(self.device)
        data[1] = data[1].to(self.device)
        param_idxs = param_idxs.to(self.device)

        # Create onehot vectors and parameter indexes for the entire batch
        idx_vectors = []
        params = []
        for param_idx in param_idxs:
            idx_vector = torch.squeeze(self.wrapper.model.to_onehot(param_idx)) # coordinate of the param in one hot vector form
            param = self.wrapper.model.get_param(param_idx)
            idx_vectors.append(idx_vector)
            params.append(param)
        idx_vectors = torch.stack((idx_vectors)).to(self.device)
        params = torch.tensor(params, device=self.device)

        logits = self.wrapper.model(idx_vectors, data[0])
        predictions = {"param": logits[0],
                        "aux": logits[1]}
        for i, param_idx in enumerate(param_idxs):
            self.param_idx_map[param_idx.item()] = logits[0][i]
        aux_pred = torch.argmax(predictions["aux"], dim=1) # get the indices of the max log-probability
        targets = {"param": params, "aux": data[-1]}

        loss = self.loss(predictions, targets)

        self.epoch_data["total"][1] += data[0].shape[0] #accumulate total number of samples in this batch
        self.epoch_data["correct"][1] += aux_pred.eq(data[1].view_as(aux_pred)).sum().item()
        self.epoch_data["sr_loss"][1] += loss["sr_loss"].item() #accumulate for epoch
        self.epoch_data["task_loss"][1] += loss["task_loss"].item() #accumulate for epoch
        self.epoch_data["combined_loss"][1] += loss["combined_loss"].item() #accumulate for epoch
        self.epoch_data["predictions"][1] += logits[-1].cpu().detach().tolist()
        self.epoch_data["targets"][1] += data[-1].cpu().detach().tolist()

        return predictions, targets

    def loss(self, predictions, targets):
        return loss(self.config, self.wrapper.model, predictions, targets)

    def score(self):
        return scores(self.config, self.dataset, self.epoch_data, self.device)

    def write(self, epoch: int, scores: Dict):
        logger.info(f"Train scores, Test scores: {scores}")

        # Log average batch values for training
        actual_sr_train_loss = self.epoch_data["sr_loss"][0] / self.train_epoch_length
        actual_task_train_loss = self.epoch_data["task_loss"][0] / self.train_epoch_length
        actual_combined_train_loss = self.epoch_data["combined_loss"][0] / self.train_epoch_length

        self.tb_logger.scalar_summary('sr_loss (train)', actual_sr_train_loss, epoch)
        self.tb_logger.scalar_summary('task_loss (train)', actual_task_train_loss, epoch)
        self.tb_logger.scalar_summary('combined_loss (train)', actual_combined_train_loss, epoch)
        self.tb_logger.scalar_summary('scores (train)', scores["acc"][0], epoch)

        if self.wandb_logger is not None:
            self.wandb_logger.logger.log(data={
                'train/sr_loss': actual_sr_train_loss,
                'train/task_loss': actual_task_train_loss,
                'train/combined_loss': actual_combined_train_loss,
                'train/scores': scores["acc"][0],
            }, step=epoch, commit=False)


        # Log average batch values for testing
        actual_sr_test_loss = self.epoch_data["sr_loss"][1] / self.test_epoch_length
        actual_task_test_loss = self.epoch_data["task_loss"][1] / self.test_epoch_length
        actual_combined_test_loss = self.epoch_data["combined_loss"][1] / self.test_epoch_length

        self.tb_logger.scalar_summary('sr_loss (test)', actual_sr_test_loss, epoch)
        self.tb_logger.scalar_summary('task_loss (test)', actual_task_test_loss, epoch)
        self.tb_logger.scalar_summary('combined_loss (test)', actual_combined_test_loss, epoch)
        self.tb_logger.scalar_summary('scores (test)', scores["acc"][1], epoch)

        if self.wandb_logger is not None:
            self.wandb_logger.logger.log(data={
                'test/sr_loss': actual_sr_test_loss,
                'test/task_loss': actual_task_test_loss,
                'test/combined_loss': actual_combined_test_loss,
                'test/scores': scores["acc"][1],
            }, step=epoch, commit=True)

        logger.info("Successfully wrote logs to tensorboard")

    def reset(self):
        for i in range(len(self.dataset)):
            self.epoch_data["sr_loss"][i] = 0
            self.epoch_data["task_loss"][i] = 0
            self.epoch_data["combined_loss"][i] = 0
            self.epoch_data["correct"][i] = 0
            self.epoch_data["total"][i] = 0

        logger.info("States successfully reset for new epoch")

    @timed
    def run_train(self):
        if all(isinstance(dataloader, DataLoader) for dataloader in self.dataset.values()):
            for epoch in trange(0, self.run_config["num_epochs"], desc="Epochs"):
                logger.info(f"Epoch: {epoch}")
                if self.wandb_logger.logger is not None:
                    self.wandb_logger.logger.log({
                            'epoch': epoch
                        }, commit=False)

                self.train_epoch_length = len(self.dataset[list(self.dataset)[0]]) # number of training batches
                for batch_idx, (data, param_idxs) in enumerate(self.dataset[list(self.dataset)[0]]):
                    logger.info(f"Running train batch: #{batch_idx}")
                    outputs, targets = self.train(data, param_idxs)

                self.test_epoch_length = len(self.dataset[list(self.dataset)[1]]) # number of testing batches
                for batch_idx, (data, param_idxs) in enumerate(self.dataset[list(self.dataset)[1]]):
                    logger.info(f"Running test batch: #{batch_idx}")
                    outputs, targets = self.test(data, param_idxs)

                # Scores cumulated and calculated per epoch, as done in Quine
                epoch_scores = self.score()

                # Regeneration (per epoch) step if specified in config
                if self.run_config["regenerate"]: self.wrapper.model.regenerate(self.param_idx_map)

                self.checkpoint.checkpoint(self.config,
                                           epoch,
                                           self.wrapper.model,
                                           self.epoch_data["sr_loss"][0],
                                           self.optimizer)

                self.write(epoch, epoch_scores)
                self.reset()
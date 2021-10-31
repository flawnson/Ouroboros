
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
        self.epoch_data = {"sr_loss": [0] * len(dataset)}
        self.param_idx_map = dict({}) # Maps param_idx to value, to be used in regeneration

    def train(self, param_idxs):
        self.wrapper.model.train()
        self.optimizer.zero_grad()

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
        output = self.wrapper.model(idx_vectors)
        predictions = {"param": output}
        for i, param_idx in enumerate(param_idxs):
            self.param_idx_map[param_idx.item()] = output[i]
        targets = {"param": params}

        loss = self.loss(predictions, targets)

        self.epoch_data["sr_loss"][0] += loss["sr_loss"].item()  # accumulate

        loss["sr_loss"].backward()
        self.optimizer.step()

        return predictions, targets

    @torch.no_grad()
    def test(self, param_idxs):
        self.wrapper.model.eval()

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

        test_output = self.wrapper.model(idx_vectors)
        predictions = {"param": test_output}
        for i, param_idx in enumerate(param_idxs):
            self.param_idx_map[param_idx.item()] = test_output[i]
        targets = {"param": params}

        loss = self.loss(predictions, targets)
        self.epoch_data["sr_loss"][1] += loss["sr_loss"].item() #accumulate for epoch

        return predictions, targets

    def loss(self, predictions, targets):
        return loss(self.config, self.wrapper.model, predictions, targets)

    def score(self):
        # Vanilla doesn't have any scoring
        pass

    def write(self, epoch: int):
        # Log values for training
        actual_train_loss = self.epoch_data["sr_loss"][0] / self.train_epoch_length
        self.tb_logger.scalar_summary('sr_loss (train)', actual_train_loss, epoch)

        if self.wandb_logger is not None:
            self.wandb_logger.logger.log({
                'train/sr_loss': actual_train_loss,
            }, step=epoch, commit=False)

        # Log values for testing
        actual_test_loss = self.epoch_data["sr_loss"][1] / self.test_epoch_length
        self.tb_logger.scalar_summary('sr_loss (test)', actual_test_loss, epoch)

        if self.wandb_logger is not None:
            self.wandb_logger.logger.log({
                'test/sr_loss': actual_test_loss,
            }, step=epoch, commit=True)

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
                if self.wandb_logger is not None:
                    self.wandb_logger.logger.log({
                        'epoch': epoch
                    }, commit=False)

                self.train_epoch_length = len(self.dataset[list(self.dataset)[0]])
                for batch_idx, param_idxs in enumerate(self.dataset[list(self.dataset)[0]]):
                    logger.info(f"Running train batch: #{batch_idx}")
                    predictions, targets = self.train(param_idxs)

                self.test_epoch_length = len(self.dataset[list(self.dataset)[1]])
                for batch_idx, param_idxs in enumerate(self.dataset[list(self.dataset)[1]]):
                    logger.info(f"Running test batch: #{batch_idx}")
                    predictions, targets = self.test(param_idxs)

                self.checkpoint.checkpoint(self.config,
                                           epoch,
                                           self.wrapper.model,
                                           self.epoch_data["sr_loss"][0],
                                           self.optimizer)

                self.write(epoch)
                self.reset()


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


class SequentialAuxiliaryTrainer(AbstractTrainer):
    def __init__(self, config, model, dataset, device):
        super(SequentialAuxiliaryTrainer, self).__init__(config, model, dataset, device)
        self.config = config
        self.model = model
        self.dataset = dataset
        self.device = device
        self.bptt_counter = 0  # Must start at zero and increment by bptt each epoch
        self.train_mode = None
        self.epoch_data = {"sr_loss": [0, 0],
                           "task_loss": [0, 0],  # The aux loss, original implementation is nll_loss
                           "combined_loss": [0, 0],
                           "correct": [0, 0],
                           "total": [0, 0]}  # First position for training scores, second position for test scores
        # self.datatensor = torch.cat(self.dataset[list(self.dataset)[0]][0].dataset.datasets)

    def loss(self, predictions, targets):
        self.config["train_mode"] = self.train_mode
        return loss(self.config, self.model, predictions, targets)

    def sequence_train(self, batch, src_mask):
        self.model.train()
        self.optimizer.zero_grad()

        i = self.bptt_counter * self.config["data_config"]["bptt"]
        self.bptt_counter += 1

        if batch[0].size(1) != self.config["data_config"]["bptt"]:
            src_mask = self.model.model.generate_square_subsequent_mask(batch.size(1)).to(self.device)

        output = self.model(batch[0], src_mask)
        predictions = {"aux": output.view(-1, len(self.config["vocab"]))}
        targets = {"aux": batch[1].reshape(-1)}

        loss = self.loss(predictions, targets)

        self.epoch_data["task_loss"][0] += loss["task_loss"].item()  # accumulate

        loss["task_loss"].backward()
        torch.nn.utils.clip_grad_norm_(self.model.model.parameters(), 0.5)
        self.optimizer.step()

    def param_train(self, batch):
        self.model.train()
        self.optimizer.zero_grad()

        output = self.model(batch[0])
        predictions = {"aux": output.view(-1, len(self.config["vocab"]))}
        targets = {"aux": batch[1].reshape(-1)}

        self.epoch_data["sr_loss"][0] += loss["sr_loss"].item()  # accumulate

    @torch.no_grad()
    def sequence_test(self, batch, src_mask):
        self.model.eval()

        if batch[1].size(0) != self.config["data_config"]["bptt"]:
            src_mask = self.model.generate_square_subsequent_mask(batch[0].size(0)).to(self.device)

        output = self.model(batch[0], src_mask)
        predictions = {"aux": output.view(-1, len(self.config["vocab"]))}
        targets = {"aux": batch[1].reshape(-1)}

        loss = self.loss(predictions, targets)

        self.epoch_data["task_loss"][0] += loss["task_loss"].item()  # accumulate

    def param_test(self, param_idx):
        self.model.eval()

    def score(self):
        return scores(self.config, self.dataset, self.epoch_data, self.device)

    def reset(self):
        for i in range(len(self.dataset)):
            self.epoch_data["task_loss"][i] = 0

        logger.info("States successfully reset for new epoch")

    def write(self):
        pass

    @timed
    def run_train(self):
        for epoch in trange(0, self.run_config["num_epochs"], desc="Epochs"):
            logger.info(f"Epoch: {epoch}")
            if self.wandb_logger is not None:
                self.wandb_logger.logger.log({
                    'epoch': epoch
                }, commit=False)

            src_mask = self.model.model.generate_square_subsequent_mask(self.config["data_config"]["bptt"]).to(self.device)
            self.train_epoch_length = len(self.dataset[list(self.dataset)[0]])  # number of training batches
            self.train_mode = "aux"
            for batch_idx, data in enumerate(self.dataset[list(self.dataset)[0]][0]):
                logger.info(f"Running sequence train batch: #{batch_idx}")
                self.sequence_train(data, src_mask)
            self.train_mode = "param"
            for batch_idx, param_idx in enumerate(self.dataset[list(self.dataset)[0]][1]):
                logger.info(f"Running param train batch: #{batch_idx}")
                self.param_train(param_idx)

            self.test_epoch_length = len(self.dataset[list(self.dataset)[1]])  # number of testing batches
            self.train_mode = "aux"
            for batch_idx, data in enumerate(self.dataset[list(self.dataset)[1]][0]):
                logger.info(f"Running sequence test batch: #{batch_idx}")
                self.sequence_test(data, src_mask)
            self.train_mode = "param"
            for batch_idx, param_idx in enumerate(self.dataset[list(self.dataset)[1]][1]):
                logger.info(f"Running param test batch: #{batch_idx}")
                self.param_test(param_idx)

            # Scores cumulated and calculated per epoch, as done in Quine
            epoch_scores = self.score()

            # Regeneration (per epoch) step if specified in config
            if self.run_config["regenerate"]: self.model.regenerate(self.param_idx_map)

            self.checkpoint.checkpoint(self.config,
                                       epoch,
                                       self.model,
                                       self.epoch_data["sr_loss"][0],
                                       self.optimizer)

            self.write(epoch, epoch_scores)
            self.reset()

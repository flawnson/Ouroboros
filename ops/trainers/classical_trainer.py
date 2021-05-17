import torch

from typing import *
from tqdm import trange
from logzero import logger

from .abstract_trainer import AbstractTrainer
from torch.utils.data import DataLoader
from optim.algos import OptimizerObj, LRScheduler
from optim.losses import loss
from utils.scores import scores
from utils.checkpoint import PTCheckpoint
from utils.logging import PTTBLogger
from utils.utilities import timed


class ClassicalTrainer(AbstractTrainer):

    def __init__(self, config: Dict, model: torch.nn.Module, dataset: Dict[str, DataLoader], device: torch.device):
        """
        Initializes a ClassicalTrainer class.

        Args:
            config: Configuration dictionary of the run.
            model: The Pytorch model.
            dataset: The data that will be used in the run.
            device: Device that training will be run on.

        Attributes:
            config: Configuration dictionary of the run.
            data_config: Data configurations of the run.
            run_config: Run configurations of the run.
            model: The Pytorch model.
            optimizer: Optimizer used for the run.
            scheduler: Learning rate scheduler used for the run.
            tb_logger: Tensorboard logger.
            dataset: The data that will be used in the run.
            device: Device that training will be run on.
            batch_data: Temporarily store per-batch scores and values for logging.
            epoch_data: Temporarily store per-epoch scores and values for logging.
        """

        # TODO: Potentially unecessary usage of dictionary in the return of loss function
        super(ClassicalTrainer, self).__init__(config, model, dataset, device)
        self.config = config
        self.data_config = config["data_config"]
        self.run_config = config["run_config"]
        self.model = model
        self.optimizer = OptimizerObj(config, self.model).optim_obj
        self.scheduler = LRScheduler(config, self.optimizer).schedule_obj
        self.tb_logger = PTTBLogger(config)
        self.checkpoint = PTCheckpoint(config)
        self.dataset = dataset
        self.device = device
        self.batch_data = {"loss": [0] * len(dataset),
                           "correct": [0] * len(dataset)}  # First position for training scores, second position for test scores
        self.epoch_data = {"loss": [0] * len(dataset),
                           "correct": [0] * len(dataset)}  # First position for training scores, second position for test scores

    def train(self, data, batch_idx):
        """
        Run data into model, collect output and target label for loss calculations.
        Args:
            data: A single batch of data.
            batch_idx: The index of the batch.

        Returns:
            The model output predictions, and the target label.
        """
        self.model.train()
        self.optimizer.zero_grad()

        logits = self.model(data[0].to(self.device))
        predictions = logits.argmax(keepdim=True)
        loss = self.loss(logits, data[1].to(self.device))

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
        """
        Run data into model, collect output and target label for loss calculations.
        Args:
            data: A single batch of data.
            batch_idx: The index of the batch.

        Returns:
            The model output predictions, and the target label.
        """
        self.model.eval()

        logits = self.model(data[0].to(self.device))
        predictions = logits.argmax(keepdim=True)
        loss = self.loss(logits, data[1].to(self.device))

        if ((batch_idx + 1) % self.data_config["batch_size"]) == 0:
            self.epoch_data["loss"][1] += self.batch_data["loss"][1]
            self.batch_data["loss"][1] = 0.0

        self.epoch_data["correct"][1] += predictions.eq(data[1].view_as(predictions)).sum().item()
        self.batch_data["loss"][1] += loss["loss"].item()
        return logits, data[1]

    def loss(self, predictions, targets) -> Dict:
        """
        Calculates loss based on predictions and targets.
        Args:
            predictions: Model output predictions.
            targets: Target labels corresponding to the model predictions.

        Returns:
            A dictionary of loss values depending on the model type.
        """
        return loss(self.config, self.model, predictions, targets)

    def score(self) -> Dict:
        """
        Calculates score.
        Returns:
            A score dictionary.
        """
        return scores(self.config, self.dataset, self.epoch_data["correct"], self.device)

    def write(self, epoch: int, scores: Dict, train_epoch_length: int, test_epoch_length: int):
        """
        Logs the loss and scores to Tensorboard.
        """
        logger.info(f"Train scores, Test scores: {scores}")
        logger.info(f"Total Loss value: {self.epoch_data['loss'][0]}")
        logger.info(f"Train epoch length: {train_epoch_length}")
        logger.info(f"Test epoch length: {test_epoch_length}")
        logger.info(f"Batch size: {self.data_config['batch_size']}")

        train_loss = self.epoch_data["loss"][0] / (train_epoch_length // self.data_config["batch_size"])
        self.tb_logger.scalar_summary('loss (train)', train_loss, epoch)
        self.tb_logger.scalar_summary('scores (train)', scores["acc"][0], epoch)

        # Log values for testing
        test_loss = self.epoch_data["loss"][1] / (test_epoch_length // self.data_config["batch_size"])
        self.tb_logger.scalar_summary('loss (test)', test_loss, epoch)
        self.tb_logger.scalar_summary('scores (test)', scores["acc"][1], epoch)

        logger.info("Successfully wrote logs to tensorboard")

    def reset(self):
        """
        Reset the temporary state values for epoch_data.
        """
        for i in range(len(self.dataset)):
            self.epoch_data["loss"][i] = 0
            self.epoch_data["correct"][i] = 0

        logger.info("States successfully reset for new epoch")

    @timed
    def run_train(self):
        """
        Main training loop.
        """
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

                self.checkpoint.checkpoint(self.config,
                                           epoch,
                                           self.model,
                                           self.epoch_data["loss"][0],
                                           self.optimizer)

                epoch_scores = self.score()
                self.write(epoch, epoch_scores, train_epoch_length, test_epoch_length)
                self.reset()

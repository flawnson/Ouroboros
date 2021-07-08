
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
                for batch_idx, param_idx in enumerate(self.dataset[list(self.dataset)[0]]):
                    logger.info(f"Running train batch: #{batch_idx}")
                    predictions, targets = self.train(param_idx)

                self.test_epoch_length = len(self.dataset[list(self.dataset)[1]])
                for batch_idx, param_idx in enumerate(self.dataset[list(self.dataset)[1]]):
                    logger.info(f"Running test batch: #{batch_idx}")
                    predictions, targets = self.test(param_idx)

                self.checkpoint.checkpoint(self.config,
                                           epoch,
                                           self.wrapper.model,
                                           self.epoch_data["sr_loss"][0],
                                           self.optimizer)

                self.write(epoch)
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
        self.epoch_data = {"sr_loss": [0, 0],
                           "task_loss": [0, 0],  # The aux loss, original implementation is nll_loss
                           "combined_loss": [0, 0],
                           "correct": [0, 0],
                           "total": [0, 0]}  # First position for training scores, second position for test scores
        self.param_idx_map = dict({}) # Maps param_idx to value, to be used in regeneration

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
        output = self.wrapper.model(idx_vectors, data[0])
        predictions = {"param": output[0],
                       "aux": output[1]}
        for i, param_idx in enumerate(param_idxs):
            self.param_idx_map[param_idx.item()] = output[0][i]
        aux_pred = torch.argmax(predictions["aux"], dim=1) # get the index of the max log-probability
        targets = {"param": params, "aux": data[-1]}

        loss = self.loss(predictions, targets)

        self.epoch_data["total"][0] += data[0].shape[0] #accumulate total number of samples in this batch
        self.epoch_data["correct"][0] += aux_pred.eq(data[1].view_as(aux_pred)).sum().item()
        self.epoch_data["sr_loss"][0] += loss["sr_loss"].item()  # accumulate
        self.epoch_data["task_loss"][0] += loss["task_loss"].item()  # accumulate
        self.epoch_data["combined_loss"][0] += loss["combined_loss"].item()  # accumulate

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

        test_output = self.wrapper.model(idx_vectors, data[0])
        predictions = {"param": test_output[0],
                        "aux": test_output[1]}
        for i, param_idx in enumerate(param_idxs):
            self.param_idx_map[param_idx.item()] = test_output[0][i]
        aux_pred = torch.argmax(predictions["aux"], dim=1) # get the indices of the max log-probability
        targets = {"param": params, "aux": data[-1]}

        loss = self.loss(predictions, targets)

        self.epoch_data["total"][1] += data[0].shape[0] #accumulate total number of samples in this batch
        self.epoch_data["correct"][1] += aux_pred.eq(data[1].view_as(aux_pred)).sum().item()
        self.epoch_data["sr_loss"][1] += loss["sr_loss"].item() #accumulate for epoch
        self.epoch_data["task_loss"][1] += loss["task_loss"].item() #accumulate for epoch
        self.epoch_data["combined_loss"][1] += loss["combined_loss"].item() #accumulate for epoch

        return predictions, targets

    def loss(self, predictions, targets):
        return loss(self.config, self.wrapper.model, predictions, targets)

    def score(self):
        return scores(self.config, self.dataset, self.epoch_data["total"], self.epoch_data["correct"], self.device)

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
                if self.wandb_logger is not None:
                    self.wandb_logger.logger.log({
                            'epoch': epoch
                        }, commit=False)

                self.train_epoch_length = len(self.dataset[list(self.dataset)[0]]) # number of training batches
                for batch_idx, (data, param_idx) in enumerate(self.dataset[list(self.dataset)[0]]):
                    logger.info(f"Running train batch: #{batch_idx}")
                    outputs, targets = self.train(data, param_idx)

                self.test_epoch_length = len(self.dataset[list(self.dataset)[1]]) # number of testing batches
                for batch_idx, (data, param_idx) in enumerate(self.dataset[list(self.dataset)[1]]):
                    logger.info(f"Running test batch: #{batch_idx}")
                    outputs, targets = self.test(data, param_idx)

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

    def loss(self, predictions, targets):
        self.config["train_mode"] = self.train_mode
        return loss(self.config, self.model, predictions, targets)

    def sequence_train(self, batch, src_mask):
        self.model.train()
        self.optimizer.zero_grad()

        i = self.bptt_counter * self.config["data_config"]["bptt"]
        self.bptt_counter += 1
        seq_len = min(self.config["data_config"]["bptt"], len(self.dataset[list(self.dataset)[0]][0]) - 1 - i)
        data = self.dataset[list(self.dataset)[0]][0].dataset.subset[i:i + seq_len]
        targets = data[i + 1:i + 1 + seq_len].reshape(-1)

        if data.size(0) != self.config["data_config"]["bptt"]:
            src_mask = self.model.model.generate_square_subsequent_mask(data.size(0)).to(self.device)

        output = self.model(data, src_mask)
        predictions = {"aux": output.view(-1, len(self.dataset[list(self.dataset)[0]][0].dataset.vocab))}
        targets = {"aux": targets}

        loss = self.loss(predictions, targets)

        self.epoch_data["task_loss"][0] += loss["task_loss"].item()  # accumulate

        loss["task_loss"].backward()
        torch.nn.utils.clip_grad_norm_(self.model.model.parameters(), 0.5)
        self.optimizer.step()

    def param_train(self, param_idx):
        self.model.train()
        self.optimizer.zero_grad()

    def sequence_test(self, data):
        self.model.eval()
        print(1 + 3)

    def param_test(self, param_idx):
        self.model.eval()
        print(1 + 4)

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

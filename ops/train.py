import torch
import numpy as np

from typing import *
from tqdm import trange
from logzero import logger

from torch.utils.data import DataLoader

from optim.algos import OptimizerObj, LRScheduler
from utils.scores import Scores


class Trainer(object):
    # TODO: Consider designing Tuning and Benchmarking as subclasses of Trainer
    def __init__(self, config: Dict, model: torch.nn.Module, dataset: Union[DataLoader], split_masks, device):
        self.config = config
        self.run_config = config["run_config"]
        self.model = model
        self.params = torch.nn.ParameterList(self.model.parameters())
        self.params_data = torch.eye(self.model.num_params, device=device)
        self.optimizer = OptimizerObj(config, self.params).optim_obj
        self.scheduler = LRScheduler(config, self.optimizer).schedule_obj
        self.dataset = dataset
        self.split_masks = split_masks
        self.device = device

    def train(self, data, param_idx):
        self.model.train()
        self.optimizer.zero_grad()
        idx_vector = torch.squeeze(self.params_data[param_idx])  # Pulling out the nested tensor
        # param_idx should already be a tensor on the device when we initialized it using torch.eye
        param = self.model.get_param(param_idx)
        predicted_param, predicted_aux = self.model(idx_vector, data[0])
        self.model(data)

    def test(self, data, param_idx):
        self.model.eval()
        idx_vector = torch.squeeze(self.params_data[param_idx])  # Pulling out the nested tensor
        param = self.model.get_param(param_idx)
        predicted_param, predicted_aux = self.model(idx_vector, data[0])
        self.model(data)

    def loss(self, predictions, targets):
        loss_sr[0] = (torch.linalg.norm(predictions["param"] - targets["param"], ord=2)) ** 2

        loss_task[0] = F.nll_loss(predicted_aux.unsqueeze(dim=0), data[1])
        loss_combined[0] = loss_sr[0] + lambda_val * loss_task[0]
        avg_relative_error[0] += formulas.relative_difference(predicted_param.item(), param.item())

        total_loss_sr[0] += loss_sr[0].item()
        total_loss_task[0] += loss_task[0].item()
        total_loss_combined[0] += loss_combined[0].item()

    def score(self):
        scores = Scores(self.config, self.device).get_scores()

        return scores

    def write(self, epoch: int):
        logger.info(f"Running epoch: #{epoch}")

    def run(self):
        for epoch in trange(0, self.run_config["num_epochs"], desc="Epochs"):
            logger.info(f"Epoch: {epoch}")
            if isinstance(self.dataset, DataLoader):
                for batch_idx, (data, param_idx) in enumerate(self.dataset[0]):
                    self.train(data.to(self.device), param_idx)
                    scores = self.score()

            if isinstance(self.dataset, DataLoader):
                for batch_idx, (data, param_idx) in enumerate(self.dataset[1]):
                    self.test(data.to(self.device), param_idx)
                    scores = self.score()

            self.write(epoch)


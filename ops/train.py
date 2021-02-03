import torch
import numpy as np

from typing import *
from tqdm import trange
from logzero import logger

from optim.algos import OptimizerObj, LRScheduler


class Trainer(object):
    # TODO: Consider designing Tuning and Benchmarking as subclasses of Trainer
    def __init__(self, config: Dict, model: torch.nn.Module, dataset, device):
        self.run_config = config["run_config"]
        self.model = model
        self.params = torch.nn.ParameterList(self.model.parameters())
        self.params_data = torch.eye(self.model.num_params, device=device)
        self.optimizer = OptimizerObj(config, self.params).optim_obj
        self.scheduler = LRScheduler(config, self.optimizer).schedule_obj
        self.dataset = dataset
        self.device = device

    def train(self, data, param_idx):
        self.model.train()
        self.optimizer.zero_grad()
        idx_vector = torch.squeeze(self.params_data[param_idx])  # Pulling out the nested tensor
        # param_idx should already be a tensor on the device when we initialized it using torch.eye
        param = self.model.get_param(param_idx)
        predicted_param, predicted_aux = self.model(idx_vector, data[0])
        self.model(data)

    def test(self):
        self.model.eval()
        self.model(self.dataset[1])

    def write(self, epoch):
        logger.info(f"Running epoch: #{epoch}")

    def run(self):
        for epoch in trange(0, self.run_config["num_epochs"], desc="Epochs"):
            logger.info(f"Epoch: {epoch}")
            if isinstance(self.dataset, torch.utils.data.DataLoader):
                for batch_idx, (data, param_idx) in enumerate(self.dataset[0]):
                    self.train(data.to(self.device), param_idx)
                    idx_vector = torch.squeeze(self.params_data[param_idx])  # Pulling out the nested tensor
            self.test()
            self.write(epoch)


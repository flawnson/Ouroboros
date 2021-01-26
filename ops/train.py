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
        self.optimizer = OptimizerObj(config, self.params).optim_obj
        self.scheduler = LRScheduler(config, self.optimizer).schedule_obj
        self.dataset = dataset
        self.device = device

    def train(self, data):
        self.model.train()
        self.optimizer.zero_grad()
        self.model(data)

    def test(self):
        self.model.eval()
        self.model(self.dataset[1])

    def write(self, epoch):
        logger.info(f"Running epoch: #{epoch}")

    def run(self):
        for epoch in trange(0, self.run_config["num_epochs"], desc="Epochs"):
            logger.info(f"Epoch: {epoch}")
            if isinstance(self.dataset[0], torch.utils.data.DataLoader):
                for batch_idx, (data, param_idx) in enumerate(self.dataset[0]):
                    self.train(data.to(self.device))
                    idx_vector = torch.squeeze(params_data[param_idx])  # Pulling out the nested tensor
            self.test()
            self.write(epoch)


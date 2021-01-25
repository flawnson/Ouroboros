import torch
import numpy as np

from typing import *
from logzero import logger


class Trainer(object):
    # TODO: Consider designing Tuning and Benchmarking as subclasses of Trainer
    def __init__(self, config, model, dataset, device):
        self.run_config = config["run_config"]
        self.model = model
        self.optimizer = Optimizer(self.train_config, self.params).optim_obj
        self.scheduler = LRScheduler(self.train_config, self.optimizer).schedule_obj
        self.dataset = dataset
        self.device = device

    def train(self):
        self.model.train()

    def test(self):
        pass

    def write(self):
        pass

    def run(self):
        for epoch in self.run_config["num_epochs"]:
            self.train()
            self.test()
            self.write()


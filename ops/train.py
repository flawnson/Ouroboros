import torch
import numpy as np

from typing import *
from logzero import logger


class Trainer(object):
    # TODO: Consider designing Tuning and Benchmarking as subclasses of Trainer
    def __init__(self, config, model, dataset, device):
        self.config = config
        self.model = model
        self.dataset = dataset
        self.device = device

    def train(self):
        pass

    def test(self):
        pass

    def write(self):
        pass

    def run(self):
        for epoch in self.config["num_epochs"]:
            self.train()
            self.test()
            self.write()


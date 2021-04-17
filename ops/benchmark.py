import torch

from torch.nn import Module
from torch.utils.data import DataLoader

from ops.trainers.trainer import AuxiliaryTrainer
from ops.tune import Tuner

from typing import *


class Benchmarker:
    def __init__(self, config: Dict, model: Module, dataset: Union[DataLoader], device: torch.device):
        self.config = config
        self.model = model
        self.dataset = dataset
        self.device = device

    def run_tune(self) -> Dict:
        return Tuner(self.config, self.model, self.dataset, self.device).run()

    def run_train(self, best_config: Dict):
        return AuxiliaryTrainer(best_config, self.model, self.dataset, self.device).run()

    def run(self):
        return self.run_train(self.run_tune())




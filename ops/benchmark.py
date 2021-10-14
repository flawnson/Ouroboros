import torch

from torch.nn import Module

from optim.parameters import ModelParameters
from ops.trainers.trainer import get_trainer
from ops.tune import Tuner

from typing import *


class Benchmarker:
    def __init__(self, config: Dict, model: Module, param_data: ModelParameters, dataset: Dict[str, list], device: torch.device):
        super(Benchmarker, self).__init__(config, model, param_data, dataset, device)
        self.config = config
        self.model = model
        self.param_data = param_data
        self.dataset = dataset
        self.device = device

    def run_tune(self) -> Dict:
        return Tuner(self.config, self.model, self.dataset, self.device).run()

    def run_train(self, best_config: Dict):
        return get_trainer(best_config, self.model, self.dataset, self.device)

    def run(self):
        return self.run_train(self.run_tune())




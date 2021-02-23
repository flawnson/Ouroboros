import torch
import os.path as osp
import multiprocessing

from typing import *
from logzero import logger
from torch.utils.data import DataLoader

from ops.train import AbstractTrainer
from torch.nn import Module

try:
    import ray
    from ray import tune
except ModuleNotFoundError:
    logger.info("Ray is not available, continuing run without benchmarking")


class Tuner(AbstractTrainer):
    def __init__(self, config: Dict, model: Module, dataset: Union[DataLoader], split_masks: List, device: torch.device):
        super(Tuner, self).__init__(config, model, dataset, split_masks, device)
        self.tuning_config = config["tuning_config"]
        self.model = model
        self.dataset = dataset
        self.split_masks = split_masks
        self.device = device

    def run_tune(self):
        cpus = int(multiprocessing.cpu_count())
        gpus = 1 if torch.cuda.device_count() >= 1 else 0

        ray.init()
        analysis = tune.run(
            self.run_train,  # Use the training pipeline to iterate through grid search
            config=self.tuning_config,
            num_samples=1,
            local_dir=osp.join(osp.dirname(osp.dirname(__file__)),
                               "logs",
                               self.tuning_config.get("model_config")["model"] + "_tuning"),
            resources_per_trial={"cpu": cpus, "gpu": gpus},
            loggers=tune.logger.DEFAULT_LOGGERS,
        )

        df = analysis.dataframe()
        df.to_csv(path_or_buf=osp.join(osp.dirname(osp.dirname(__file__)),
                                       "logs",
                                       "x_classification" + "_experiment.csv"))

        return analysis.get_best_config(metric="train_f1_score")  # needs to return config for best model




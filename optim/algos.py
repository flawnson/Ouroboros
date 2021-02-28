import torch

from typing import *
from logzero import logger
from torch.optim import Optimizer


class OptimizerObj(Optimizer):
    def __init__(self, config: Dict, params: torch.nn.ParameterList):
        """
        Consider switching to __call__ method instead of __init__

        Args:
            config: Configuration dictionary
            params: Pytorch parameter list of model parameters
        """
        super(OptimizerObj, self).__init__(params, config)
        self.optim_config = config["optim_config"]
        self.param_groups = None
        self.params = params

        if self.optim_config["optimizer"].casefold() == "adam":
            self.optim_obj = torch.optim.Adam(params, **self.optim_config["optim_kwargs"])
        elif self.optim_config["optimizer"].casefold() == "sgd":
            self.optim_obj = torch.optim.SGD(params, **self.optim_config["optim_kwargs"])
        elif self.optim_config["optimizer"].casefold() == "adagrad":
            self.optim_obj = torch.optim.Adagrad(params, **self.optim_config["optim_kwargs"])
        elif self.optim_config["optimizer"].casefold() == "rmsprop":
            self.optim_obj = torch.optim.RMSprop(params, **self.optim_config["optim_kwargs"])
        elif self.optim_config["optimizer"].casefold() == "adadelta":
            self.optim_obj = torch.optim.Adadelta(params, **self.optim_config["optim_kwargs"])
        else:
            logger.info(f"Optimizer {self.optim_config['optim']} not understood")
            raise NotImplementedError(f"Optimizer {self.optim_config['optim']} not implemented")


class LRScheduler(object):
    def __init__(self, config: Dict, optim_obj: Optimizer):
        """
        Consider switching to __call__ method instead of __init__

        Args:
            config: Configuration dictionary
            optim_obj: Pytorch optimizer object
        """
        self.optim_config = config["optim_config"]
        try:
            if self.optim_config["scheduler"].casefold() == "cawr":
                self.schedule_obj = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim_obj, **self.optim_config["scheduler_kwargs"])
            elif self.optim_config["scheduler"].casefold() == "multistep":
                self.schedule_obj = torch.optim.lr_scheduler.MultiStepLR(optim_obj, **self.optim_config["scheduler_kwargs"])
            elif self.optim_config["scheduler"].casefold() == "cyclic":
                self.schedule_obj = torch.optim.lr_scheduler.CyclicLR(optim_obj, **self.optim_config["scheduler_kwargs"])
            elif self.optim_config["scheduler"].casefold() is None:
                self.schedule_obj = None
            else:
                self.schedule_obj = None
        except AttributeError:
            logger.info(f"Scheduler {self.optim_config['scheduler']} not provided or not understood")
            self.schedule_obj = None
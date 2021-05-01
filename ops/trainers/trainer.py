import torch

from typing import *
from logzero import logger

from torch.utils.data import DataLoader

from models.standard.mlp_model import MLPModel
from models.augmented.quine import Auxiliary, Vanilla
from models.augmented.ouroboros import Godel
from models.augmented.classical import Classical
from models.augmented.hypernetwork import PrimaryNetwork
from .abstract_trainer import AbstractTrainer
from .classical_trainer import ClassicalTrainer
from .hypernetwork_trainer import HyperNetworkTrainer, DualHyperNetworkTrainer
from .quine_trainer import AuxiliaryTrainer, VanillaTrainer


def trainer(config: Dict, model: torch.nn.Module, param_data: torch.nn.Module, dataloaders: List[torch.utils.data.DataLoader], device: torch.device):
    """
    Initializes and returns a Trainer object based on model type.

    Returns:
        A trainer instance.
    """
    if isinstance(model, (Classical, MLPModel)):
        return ClassicalTrainer(config, model, dataloaders, device).run_train()
    elif isinstance(model, Auxiliary):
        return AuxiliaryTrainer(config, param_data, dataloaders, device).run_train()
    elif isinstance(model, Vanilla):
        return VanillaTrainer(config, param_data, dataloaders, device).run_train()
    elif isinstance(model, PrimaryNetwork):
        return HyperNetworkTrainer(config, model, dataloaders, device).run_train()
    elif isinstance(model, Godel):
        return DualHyperNetworkTrainer(config, model, dataloaders, device).run_train()
    else:
        try:
            return AbstractTrainer(config, model, dataloaders, device).run_train()
        except Exception as e:
            logger.info(e)
            raise NotImplementedError(f"Model {model} has no trainer pipeline")

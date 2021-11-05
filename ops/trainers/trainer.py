import torch

from typing import *
from logzero import logger

from torch.utils.data import DataLoader

from models.standard.linear_model import LinearModel
from models.augmented.quine import Vanilla, Auxiliary, SequentialAuxiliary, GraphAuxiliary
from models.augmented.ouroboros import Godel
from models.augmented.classical import Classical
from models.augmented.hypernetwork import ResNetPrimaryNetwork
from optim.parameters import ModelParameters
from .abstract_trainer import AbstractTrainer
from .classical_trainer import ClassicalTrainer
from .hypernetwork_trainer import HyperNetworkTrainer, DualHyperNetworkTrainer
from .aux_trainer import AuxiliaryTrainer
from .sequential_aux_trainer import SequentialAuxiliaryTrainer
from .graph_aux_trainer import GraphAuxiliaryTrainer
from .vanilla_trainer import VanillaTrainer


def get_trainer(config: Dict, model: torch.nn.Module, param_data: ModelParameters, dataloaders: Dict[str, torch.utils.data.DataLoader], device: torch.device):
    """
    Initializes and returns a Trainer object based on model type.

    Returns:
        A trainer instance.
    """
    if isinstance(model, (Classical, LinearModel)):
        return ClassicalTrainer(config, model, dataloaders, device).run_train()
    elif isinstance(model, Vanilla):
        return VanillaTrainer(config, param_data, dataloaders, device).run_train()
    elif isinstance(model, Auxiliary):
        return AuxiliaryTrainer(config, param_data, dataloaders, device).run_train()
    elif isinstance(model, SequentialAuxiliary):
        return SequentialAuxiliaryTrainer(config, model, dataloaders, device).run_train()
    elif isinstance(model, GraphAuxiliary):
        return GraphAuxiliaryTrainer(config, model, dataloaders, device).run_train()
    elif isinstance(model, ResNetPrimaryNetwork):
        return HyperNetworkTrainer(config, model, dataloaders, device).run_train()
    elif isinstance(model, Godel):
        return DualHyperNetworkTrainer(config, model, dataloaders, device).run_train()
    else:
        try:
            return AbstractTrainer(config, model, dataloaders, device).run_train()
        except Exception as e:
            logger.info(e)
            raise NotImplementedError(f"Model {model} has no trainer pipeline")

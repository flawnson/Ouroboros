import os
import json
import torch
import pytest
import logzero
import logging
import argparse
import numpy as np

from torch.utils.data import DataLoader
from typing import *
from logzero import logger

from utils.scores import scores
from utils.utilities import timed
from data.graph_preprocessing import PrimaryLabelset
from data.linear_preprocessing import HousingDataset
from data.data_preprocessing import get_image_data, get_graph_data, get_text_data
from models.standard.graph_model import GNNModel
from models.standard.linear_model import LinearModel
from models.standard.transformer_model import TransformerModel
from utils.checkpoint import load
from models.augmented.quine import get_auxiliary, Vanilla
from models.augmented.hypernetwork import MLPHyperNetwork, LinearHyperNetwork, ResNetPrimaryNetwork
from models.augmented.classical import Classical
from models.augmented.ouroboros import Ouroboros
from optim.parameters import ModelParameters
from utils.splitting import get_image_data_split, get_text_data_split


@pytest.fixture
def config():
    file = os.path.join("..", "configs", "linear_aux_demo_small.json")
    config: Dict = json.load(open(file))
    device = torch.device("cuda" if config["device"] == "cuda" and torch.cuda.is_available() else "cpu")
    logzero.loglevel(eval(config["log_level"]))
    logger.info(f"Successfully retrieved config json. Running {config['run_name']} on {device}.")

    return config


@pytest.fixture  # Can pass params directly as a list but would rather use parameterize
def device(config):
    device = torch.device("cuda" if config["device"] == "cuda" and torch.cuda.is_available() else "cpu")
    logger.info(f"Running {config['run_name']} on {device}.")

    return device


@pytest.fixture
def datasets(config):
    ### Aux Data preprocessing ###
    datasets: Union[torch.utils.data.Dataset, List] = None
    if config["data_config"]["dataset"].casefold() == "primary_labelset":
        datasets = PrimaryLabelset(config).dataset.to(device)
    elif config["data_config"]["dataset"].casefold() == "house":
        datasets = HousingDataset(config).dataset.to(device)
    elif config["data_config"]["dataset"].casefold() == "cora":
        datasets = get_graph_data(config)  # Cora only has one graph (index must be 0)
    elif config["data_config"]["dataset"].casefold() in ("mnist", "cifar10", "imagenet"):
        datasets = get_image_data(config)
    elif config["data_config"]["dataset"].casefold() in ("wikitext2", "amazonreviewfull"):
        datasets = get_text_data(config)
    else:
        raise NotImplementedError(f"{config['dataset']} is not a dataset")
    logger.info(f"Successfully built the {config['data_config']['dataset']} dataset")

    return datasets


@pytest.fixture
def model(config, datasets, device):
    ### Model preparation ###
    model: torch.nn.Module = None
    if config["model_config"]["model_type"].casefold() == "linear":
        model = LinearModel(config, device).to(device)
    elif config["model_config"]["model_type"].casefold() == "graph":
        model = GNNModel(config, datasets, device).to(device)
    elif config["model_config"]["model_type"].casefold() == "vision":
        pass
    elif config["model_config"]["model_type"].casefold() == "sequential":
        model = TransformerModel(config, datasets, device).to(device)
    elif config["model_config"]["model_type"].casefold() == "hypernetwork":
        pass
    elif config["model_config"]["load_dir"]:
        model = load(config["model_config"]["load_model"])
    else:
        raise NotImplementedError(f"{config['model_config']['model_type']} is not a model type")
    logger.info(f"Successfully built the {config['model_config']['model_type']} model type")

    return model


@pytest.fixture
def aug_model(config, model, datasets, device):
    ### Model augmentation ### (for none, use classical, all augmentations are model agnostic)
    aug_model: torch.nn.Module = None
    if config["model_aug_config"]["model_augmentation"].casefold() == "classical":
        aug_model = Classical(config, model, device).to(device)
    elif config["model_aug_config"]["model_augmentation"].casefold() == "ouroboros":
        aug_model = Ouroboros(config, model, device).to(device)
    elif config["model_aug_config"]["model_augmentation"].casefold() == "auxiliary":
        aug_model = get_auxiliary(config, model, datasets, device).to(device)
    elif config["model_aug_config"]["model_augmentation"].casefold() == "vanilla":
        aug_model = Vanilla(config, model, device).to(device)
    elif config["model_aug_config"]["model_augmentation"].casefold() == "hypernetwork":
        if config["model_config"]["model_type"].casefold() == "hypernetwork":
            aug_model = ResNetPrimaryNetwork(config, device=device).to(device)
        if config["model_config"]["model_type"].casefold() == "linear":
            aug_model = LinearHyperNetwork(config, device=device).to(device)
        if config["model_config"]["model_type"].casefold() == "image":
            aug_model = MLPHyperNetwork(config, model, device).to(device)
    else:
        raise NotImplementedError(f"{config['model_aug_config']['model_augmentation']} is not a model augmentation")
    logger.info(f"Successfully built the {config['model_aug_config']['model_augmentation']} augmentation")

    return aug_model


@pytest.fixture
def param_data(config, aug_model, datasets, device):
    ### Param data preprocessing ###
    param_data: ModelParameters = None
    if config["model_aug_config"]["model_augmentation"].casefold() == "classical":
        pass
    elif config["model_aug_config"]["model_augmentation"].casefold() == "vanilla":
        param_data = ModelParameters(config, aug_model, device)
    elif config["model_aug_config"]["model_augmentation"].casefold() == "auxiliary":
        param_data = ModelParameters(config, aug_model, device)
    else:
        logger.info(f"{config['model_aug_config']['model_augmentation']} does not require param data")
    logger.info(f"Successfully generated parameter data")

    return param_data


@pytest.fixture
def dataloaders(config, model, aug_model, datasets, param_data, device):
    ### Splitting dataset and parameters ###
    dataloaders: Dict[str, DataLoader] = None
    if config["data_config"]["dataset"] == "primary_labelset":
        pass
    elif config["data_config"]["dataset"].casefold() == "house":
        pass
    elif config["data_config"]["dataset"].casefold() in ("cora", "reddit"):
        pass
    elif config["data_config"]["dataset"].casefold() in ("mnist", "cifar10"):
        dataloaders = get_image_data_split(config, datasets, param_data, device)
    elif config["data_config"]["dataset"].casefold() in ("wikitext2", "amazonreviewfull"):
        dataloaders = get_text_data_split(config, datasets, param_data, device)
    else:
        raise NotImplementedError(f"Either {config['data_config']['dataset']} or {config['model_config']['model_type']} "
                                  f"does not have a valid split")
    logger.info(f"Successfully split dataset and parameters")

    return dataloaders


@timed
def test_regenerating(config, device, aug_model, dataloaders):
    param_idx_map = dict({})  # Maps param_idx to value, to be used in regeneration
    logits = torch.rand(1, 1)

    for batch_idx, (data, param_idxs) in enumerate(dataloaders[list(dataloaders)[0]]):
        for i, param_idx in enumerate(param_idxs):
            param_idx_map[param_idx.item()] = logits[0][i]

    aug_model().regenerate(param_idx_map)





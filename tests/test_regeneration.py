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

@pytest.fixture
def aug_model(config, datasets, device):
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


@timed
def test_regenerating(config, device, aug_model, datasets):
    for batch_idx, (data, param_idxs) in enumerate(datasets[list(datasets)[0]]):
        for i, param_idx in enumerate(param_idxs):
            param_idx_map[param_idx.item()] = logits[0][i]
    aug_model().regenerate(param_idx_map)
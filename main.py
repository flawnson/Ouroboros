import argparse
import logging
import logzero
import torch
import json
import dgl
import os

from typing import *
from logzero import logger
from models.graph_model import GNNModel
from models.mlp_model import MLPModel
from utils.quine import Auxiliary, Vanilla
from utils.classical import Classical
from utils.ouroboros import Ouroboros
from data.graph_preprocessing import AbstractGraphDataset, PrimaryLabelset
from data.linear_preprocessing import HousingDataset, get_aux_data

if __name__ == "__main__":
    ### Configuring ###
    parser = argparse.ArgumentParser(description="Config file parser")
    parser.add_argument("-c", "--config", help="json config file", type=str)
    parser.add_argument("-s", "--scheme", help="json scheme file", type=str)
    args = parser.parse_args()

    config: dict = json.load(open(args.config))
    device = torch.device("cuda" if config["device"] == "cuda" and torch.cuda.is_available() else "cpu")
    logzero.loglevel(eval(config["logging"]))

    ### Data preprocessing ###
    if config["data_config"]["dataset"] == "primary_labelset":
        dataset = PrimaryLabelset(config).dataset.to(device)
    elif config["data_config"]["dataset"].casefold() == "house":
        dataset = HousingDataset(config).dataset.to(device)
    elif config["data_config"]["dataset"].casefold() == "cora":
        dataset = dgl.data.CoraFull()[0]  # Cora only has one graph (index must be 0)
    elif config["data_config"]["dataset"].casefold() == "mnist":
        dataset = get_aux_data(config)  # Two dataloaders in a list, for training and testing
    else:
        raise NotImplementedError(f"{config['dataset']} is not a dataset")  # Add to logger when implemented
    logger.info(f"Successfully built the {config['data_config']['dataset']} dataset")

    ### Model preparation ###
    # Model selection
    if config["model_config"]["model_type"] == "linear":
        model = MLPModel(config, dataset, device).to(device)
    elif config["model_config"]["model_type"] == "graph":
        model = GNNModel(config, dataset, device).to(device)
    elif config["model_config"]["model_type"] == "vision":
        pass
    elif config["model_config"]["model_type"] == "language":
        pass
    else:
        raise NotImplementedError(f"{config['model_config']['model_type']} is not a model type")
    logger.info(f"Successfully built the {config['model_config']['model_type']} model type")

    # Model augmentation (for none, use classical, all augmentations are model agnostic)
    if config["model_aug_config"]["model_augmentation"] == "classical":
        aug_model = Classical(config["model_aug_config"], model, device).to(device)
    elif config["model_aug_config"]["model_augmentation"] == "ouroboros":
        aug_model = Ouroboros(config["model_aug_config"], model, device).to(device)
    elif config["model_aug_config"]["model_augmentation"] == "auxiliary":
        aug_model = Auxiliary(config["model_aug_config"], model, dataset, device).to(device)
    elif config["model_aug_config"]["model_augmentation"] == "vanilla":
        aug_model = Vanilla(config["model_aug_config"], model, device).to(device)
    else:
        raise NotImplementedError(f"{config['model_aug_config']['model_augmentation']} is not a model augmentation")
    logger.info(f"Successfully built the {config['model_aug_config']['model_augmentation']} augmentation")

    ### Pipeline ###
    if config["run_type"] == "demo":
        Trainer()
    if config["run_type"] == "tune":
        Tuner()
    if config["run_type"] == "benchmark":
        Benchmarker()

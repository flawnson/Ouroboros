import argparse
import logging
import logzero
import torch
import json
import dgl

from typing import *
from logzero import logger
from torch.utils.data import DataLoader

from models.standard.graph_model import GNNModel
from models.standard.mlp_model import MLPModel
from models.augmented.quine import Auxiliary, Vanilla
from models.augmented.classical import Classical
from models.augmented.ouroboros import Ouroboros
from data.graph_preprocessing import PrimaryLabelset
from data.linear_preprocessing import HousingDataset, get_aux_data
from utils.holdout import MNISTSplit, QuineSplit
from optim.parameters import ModelParameters
from ops.train import trainer
from ops.tune import Tuner
from ops.benchmark import Benchmarker


def main():
    ### Configuring ###
    parser = argparse.ArgumentParser(description="Config file parser")
    parser.add_argument("-c", "--config", help="json config file", type=str)
    parser.add_argument("-s", "--scheme", help="json scheme file", type=str)
    args = parser.parse_args()

    config: Dict = json.load(open(args.config))
    device = torch.device("cuda" if config["device"] == "cuda" and torch.cuda.is_available() else "cpu")
    logzero.loglevel(eval(config["logging"]))
    logger.info(f"Successfully retrieved config json. Running {config['run_name']} on {device}.")

    ### Aux Data preprocessing ###
    datasets: Union[torch.utils.data.Dataset, List] = None
    if config["data_config"]["dataset"].casefold() == "primary_labelset":
        datasets = PrimaryLabelset(config).dataset.to(device)
    elif config["data_config"]["dataset"].casefold() == "house":
        datasets = HousingDataset(config).dataset.to(device)
    elif config["data_config"]["dataset"].casefold() == "cora":
        datasets = dgl.data.CoraFull()[0]  # Cora only has one graph (index must be 0)
    elif config["data_config"]["dataset"].casefold() == "mnist":
        datasets = get_aux_data(config)
    else:
        raise NotImplementedError(f"{config['dataset']} is not a dataset")
    logger.info(f"Successfully built the {config['data_config']['dataset']} dataset")

    ### Model preparation ###
    model: torch.nn.Module = None
    if config["model_config"]["model_type"].casefold() == "linear":
        model = MLPModel(config, datasets, device).to(device)
    elif config["model_config"]["model_type"].casefold() == "graph":
        model = GNNModel(config, datasets, device).to(device)
    elif config["model_config"]["model_type"].casefold() == "vision":
        pass
    elif config["model_config"]["model_type"].casefold() == "language":
        pass
    else:
        raise NotImplementedError(f"{config['model_config']['model_type']} is not a model type")
    logger.info(f"Successfully built the {config['model_config']['model_type']} model type")

    ### Model augmentation ### (for none, use classical, all augmentations are model agnostic)
    aug_model: torch.nn.Module = None
    if config["model_aug_config"]["model_augmentation"].casefold() == "classical":
        aug_model = Classical(config, model, datasets, device).to(device)
    elif config["model_aug_config"]["model_augmentation"].casefold() == "ouroboros":
        aug_model = Ouroboros(config, model, device).to(device)
    elif config["model_aug_config"]["model_augmentation"].casefold() == "auxiliary":
        aug_model = Auxiliary(config, model, datasets, device).to(device)
    elif config["model_aug_config"]["model_augmentation"].casefold() == "vanilla":
        aug_model = Vanilla(config, model, device).to(device)
    else:
        raise NotImplementedError(f"{config['model_aug_config']['model_augmentation']} is not a model augmentation")
    logger.info(f"Successfully built the {config['model_aug_config']['model_augmentation']} augmentation")

    ### Param data preprocessing ###
    param_data: Union[torch.utils.data.Dataset, List] = None
    if config["model_aug_config"]["model_augmentation"].casefold() == "classical":
        pass
    elif config["model_aug_config"]["model_augmentation"].casefold() == "auxiliary":
        param_data = ModelParameters(config, aug_model, device)
    else:
        raise NotImplementedError(f"{config['model_aug_config']['model_augmentation']} does not require param data")
    logger.info(f"Successfully generated parameter data")

    ### Splitting dataset and parameters ###
    dataloaders: List[DataLoader] = None
    if config["data_config"]["dataset"] == "primary_labelset":
        pass
    elif config["data_config"]["dataset"].casefold() == "house":
        pass
    elif config["data_config"]["dataset"].casefold() == "cora":
        pass
    elif config["data_config"]["dataset"].casefold() == "mnist":
        dataloaders = MNISTSplit(config, datasets, param_data, device).partition()
        if param_data is not None:
            if len(datasets) < len(param_data):
                dataloaders = QuineSplit(config, param_data, device).partition()
    else:
        raise NotImplementedError(f"{config['dataset']} is not a valid split")
    logger.info(f"Successfully split dataset and parameters")

    ### Pipeline ###
    if config["run_type"].casefold() == "demo":
        trainer(config, aug_model, param_data, dataloaders, device)
    if config["run_type"].casefold() == "tune":
        Tuner(config, aug_model, param_data, dataloaders, device)
    if config["run_type"].casefold() == "benchmark":
        Benchmarker(config, aug_model, param_data, dataloaders, device)


if __name__ == "__main__":
    main()

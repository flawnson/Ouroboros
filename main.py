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
from data.combine_preprocessing import CombineDataset
from utils.holdout import DataHoldout, ModelHoldout
from optim.parameters import ModelParameters
from ops.train import Trainer


def main():
    ### Configuring ###
    parser = argparse.ArgumentParser(description="Config file parser")
    parser.add_argument("-c", "--config", help="json config file", type=str)
    parser.add_argument("-s", "--scheme", help="json scheme file", type=str)
    args = parser.parse_args()

    config: Dict = json.load(open(args.config))
    #print(config["logging"])
    device = torch.device("cuda" if config["device"] == "cuda" and torch.cuda.is_available() else "cpu")
    logzero.loglevel(eval(config["logging"]))

    ### Aux Data preprocessing ###
    datasets: Union[torch.utils.data.Dataset, List] = None
    if config["data_config"]["dataset"] == "primary_labelset":
        datasets = PrimaryLabelset(config).dataset.to(device)
    elif config["data_config"]["dataset"].casefold() == "house":
        datasets = HousingDataset(config).dataset.to(device)
    elif config["data_config"]["dataset"].casefold() == "cora":
        datasets = dgl.data.CoraFull()[0]  # Cora only has one graph (index must be 0)
    elif config["data_config"]["dataset"].casefold() == "mnist":
        datasets = get_aux_data(config)
    else:
        raise NotImplementedError(f"{config['dataset']} is not a dataset")  # Add to logger when implemented
    logger.info(f"Successfully built the {config['data_config']['dataset']} dataset")

    ### Model preparation ###
    model: torch.nn.Module = None
    if config["model_config"]["model_type"] == "linear":
        model = MLPModel(config, datasets, device).to(device)
    elif config["model_config"]["model_type"] == "graph":
        model = GNNModel(config, datasets, device).to(device)
    elif config["model_config"]["model_type"] == "vision":
        pass
    elif config["model_config"]["model_type"] == "language":
        pass
    else:
        raise NotImplementedError(f"{config['model_config']['model_type']} is not a model type")
    logger.info(f"Successfully built the {config['model_config']['model_type']} model type")

    ### Model augmentation ### (for none, use classical, all augmentations are model agnostic)
    aug_model: torch.nn.Module = None
    if config["model_aug_config"]["model_augmentation"] == "classical":
        aug_model = Classical(config, model, device).to(device)
    elif config["model_aug_config"]["model_augmentation"] == "ouroboros":
        aug_model = Ouroboros(config, model, device).to(device)
    elif config["model_aug_config"]["model_augmentation"] == "auxiliary":
        aug_model = Auxiliary(config, model, datasets, device).to(device)
    elif config["model_aug_config"]["model_augmentation"] == "vanilla":
        aug_model = Vanilla(config, model, device).to(device)
    else:
        raise NotImplementedError(f"{config['model_aug_config']['model_augmentation']} is not a model augmentation")
    logger.info(f"Successfully built the {config['model_aug_config']['model_augmentation']} augmentation")

    ### Param data preprocessing ###
    param_data: Union[torch.utils.data.Dataset, List] = None
    if config["data_config"]["dataset"] == "primary_labelset":
        pass
    elif config["data_config"]["dataset"].casefold() == "house":
        pass
    elif config["data_config"]["dataset"].casefold() == "cora":
        pass
    elif config["data_config"]["dataset"].casefold() == "mnist":
        param_data = ModelParameters(config, aug_model, device)
    else:
        raise NotImplementedError(f"{config['dataset']} is not a dataset")  # Add to logger when implemented
    logger.info(f"Successfully generated parameter data")

    ### Splitting dataset and parameters ###
    input_data: Optional = None #do we need this?
    dataloaders = [] #will contain a dataloader for each split
    if config["data_config"]["dataset"] == "primary_labelset":
        pass
    elif config["data_config"]["dataset"].casefold() == "house":
        pass
    elif config["data_config"]["dataset"].casefold() == "cora":
        pass
    elif config["data_config"]["dataset"].casefold() == "mnist":
        data_split = DataHoldout(config, datasets, model, device)
        model_split = ModelHoldout(config, datasets, model, device)
        # XXX: NEED TO FIX SPLITTING METHODS (CURRENTLY USING MASKS)
        # split_masks = [DataLoader(CombineDataset(dataset, params)) for
        #               dataset, params in zip(data_split.split(datasets).values(), model_split.split(param_data).values())]
        split_masks = zip(data_split.split(datasets).values(), model_split.split(param_data).values())
        logger.info(f"Split masks in Main: {split_masks}")
        for split_dataset, split_params in split_masks:
            combined = CombineDataset(datasets, param_data, splits=[split_dataset, split_params])
            dloader = DataLoader(combined)
            dataloaders.append(dloader)

    else:
        raise NotImplementedError(f"{config['dataset']} is not a valid split")  # Add to logger when implemented
    logger.info(f"Successfully split dataset and parameters")

    ### Pipeline ###
    if config["run_type"] == "demo":
        #Run the trainer on every split
        #We can treat each split as a separate model in an ensemble
        #Each split can fully have their own charts, logging, checkpointing etc..
        for i, split_dataloader in enumerate(dataloaders):
            print("Split: ", i)
            print("Dataloader: ", split)
            #NOTE: Would need to modify Trainer class...
            Trainer(config, aug_model, input_data, split_dataloader, device).run_train() #may want to remove input_data since split contains the dataloader??
    if config["run_type"] == "tune":
        pass
    if config["run_type"] == "benchmark":
        pass


if __name__ == "__main__":
    main()

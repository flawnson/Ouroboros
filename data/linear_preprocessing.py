import os
import torch
import numpy as np
import pandas as pd
import torchvision as tv

from logzero import logger
from typing import Dict, List
from torch.utils.data import Dataset, ConcatDataset, Subset


class HousingDataset(Dataset):
    def __init__(self, config: Dict):
        super(HousingDataset).__init__()
        self.config = config["data_config"]
        self.dataset = self.get_data()

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

    def get_data(self):
        return None


def get_data(config: Dict) -> ConcatDataset:
    """
    Load torchvision data, both training and tuning, and return a concatenated Dataset object.
    Splitting occurs further downstream (in holdout class methods)

    Args:
        config: Configuration dictionary

    Returns:
        torch.utils.data.Datasets
    """
    logger.info(f"Downloading {config['data_config']['dataset']} data to {config['data_config']['data_dir']}")
    transform = tv.transforms.Compose([tv.transforms.ToTensor()])

    #If specified, select only a subset for faster running (TAKES DOUBLE THE NUMBER IN CONFIG)
    subset = config["data_config"].get("subset", None)
    if isinstance(subset, int):
        subset_indices = list(range(subset))
        logger.info(f"Using a subset of the dataset sized: {subset}")
    else:
        subset_indices = []

    if config["data_config"]["dataset"].casefold() == "mnist":
        tv_dataset = tv.datasets.MNIST
    elif config["data_config"]["dataset"].casefold() == "cifar":
        tv_dataset = tv.datasets.CIFAR10
    else:
        raise NotImplementedError(f"{config['dataset']} is not a dataset")

    to_concat = []
    for x in [True, False]:
        if isinstance(subset, int):
            to_concat.append(Subset(tv_dataset(os.path.join(config['data_config']['data_dir']),
                                               train=x,
                                               download=True,
                                               transform=transform),
                                    subset_indices))
        else:
            to_concat.append(tv_dataset(os.path.join(config['data_config']['data_dir']),
                                        train=x,
                                        download=True,
                                        transform=transform))
    dataset = ConcatDataset(to_concat)

    to_concat_targets = []
    for x in [True, False]:
        if isinstance(subset, int):
            to_concat_targets.append(tv_dataset(os.path.join(config['data_config']['data_dir']),
                                                train=x,
                                                download=True,
                                                transform=transform).targets[:subset])
        else:
            to_concat_targets.append(tv_dataset(os.path.join(config['data_config']['data_dir']),
                                                train=x,
                                                download=True,
                                                transform=transform).targets)

    # In case targets are not a tensor (like in CIFAR10)
    to_concat_targets = [torch.tensor(x) for x in to_concat_targets]

    dataset.targets = torch.cat(to_concat_targets)

    return dataset

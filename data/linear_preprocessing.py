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
    logger.info(f"Downloading MNIST data to {config['data_config']['data_dir']}")
    transform = tv.transforms.Compose([tv.transforms.ToTensor()])

    #If specified, select only a subset for faster running (TAKES DOUBLE THE NUMBER IN CONFIG)
    subset = config["data_config"]["subset"]
    subset_indices = []
    if isinstance(subset, int):
        subset_indices = list(range(subset))

    to_concat = []

    for x in [True, False]:
        if isinstance(subset, int):
            to_concat.append(Subset(tv.datasets.MNIST(os.path.join(config['data_config']['data_dir']), train=x, download=True, transform=transform),
                                    subset_indices))
        else:
            to_concat.append(tv.datasets.MNIST(os.path.join(config['data_config']['data_dir']), train=x, download=True, transform=transform))
    dataset = ConcatDataset(to_concat)

    to_concat_targets = []
    for x in [True, False]:
        if isinstance(subset, int):
            to_concat_targets.append(tv.datasets.MNIST(os.path.join(config['data_config']['data_dir']), train=x, download=True, transform=transform).targets[:subset])
        else:
            to_concat_targets.append(tv.datasets.MNIST(os.path.join(config['data_config']['data_dir']), train=x, download=True, transform=transform).targets)
    dataset.targets = torch.cat(to_concat_targets)
    # dataset = ConcatDataset([tv.datasets.MNIST(os.path.join(config['data_config']['data_dir']),
    #                          train=x,
    #                          download=True,
    #                          transform=transform) for x in [True, False]])
    # dataset.targets = torch.cat([tv.datasets.MNIST(os.path.join(config['data_config']['data_dir']),
    #                              train=x,
    #                              download=True,
    #                              transform=transform).targets for x in [True, False]])
    return dataset

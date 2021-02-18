import os
import torch
import numpy as np
import pandas as pd
import torchvision as tv

from logzero import logger
from typing import Dict, List
from torch.utils.data import Dataset, ConcatDataset


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


def get_aux_data(config: Dict) -> Dataset:
    logger.info(f"Downloading MNIST data to {config['data_config']['data_dir']}")
    transform = tv.transforms.Compose([tv.transforms.ToTensor()])
    dataset = ConcatDataset([tv.datasets.MNIST(os.path.join(config['data_config']['data_dir']),
                              train=x,
                              download=True,
                              transform=transform) for x in [True, False]])
    dataset.targets = torch.cat([tv.datasets.MNIST(os.path.join(config['data_config']['data_dir']),
                              train=x,
                              download=True,
                              transform=transform).targets for x in [True, False]])
    return dataset

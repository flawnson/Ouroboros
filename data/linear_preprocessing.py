import os
import torch
import numpy as np
import pandas as pd
import torchvision as tv

from logzero import logger
from typing import Dict, List
from torch.utils.data import Dataset, DataLoader


class HousingDataset(Dataset):
    def __init__(self, config):
        super(HousingDataset).__init__()
        self.config = config["data_config"]
        self.dataset = self.get_data()

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

    def get_data(self):
        return None


def get_aux_data(configs: Dict) -> List:
    logger.info(f"Downloading MNIST data to {configs['data_dir']}")
    transform = tv.transforms.Compose([
        tv.transforms.ToTensor()
    ])
    train_loader = torch.utils.data.DataLoader(tv.datasets.MNIST(os.path.join(configs['data_dir']),
                                                                 train=True,
                                                                 download=True,
                                                                 transform=transform))
    test_loader = torch.utils.data.DataLoader(tv.datasets.MNIST(os.path.join(configs['data_dir']),
                                                                train=False,
                                                                download=True,
                                                                transform=transform))

    return [train_loader, test_loader]

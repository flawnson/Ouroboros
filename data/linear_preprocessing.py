import os
import torch
import numpy as np
import pandas as pd
import torchvision as tv

from logzero import logger
from typing import Dict, List
from torch.utils.data import Dataset, DataLoader


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


def get_aux_data(config: Dict) -> List[DataLoader]:
    logger.info(f"Downloading MNIST data to {config['data_config']['data_dir']}")
    transform = tv.transforms.Compose([tv.transforms.ToTensor()])
    return [torch.utils.data.DataLoader(tv.datasets.MNIST(os.path.join(config['data_config']['data_dir']),
                                                          train=x,
                                                          download=True,
                                                          transform=transform)) for x in [True, False]]

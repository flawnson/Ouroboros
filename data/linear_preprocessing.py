import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader


class HousingDataset(Dataset):
    def __init__(self, config):
        super(HousingDataset).__init__()
        self.config = config
        self.dataset = self.get_data()

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

    def get_data(self):
        pd.read_csv(self.config["data_path"])

        return 0

from typing import Dict, List
from torch.utils.data import Dataset


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

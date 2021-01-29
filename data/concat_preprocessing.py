import torch


class CombineDataset(torch.utils.data.Dataset):
    def __init__(self, config, *datasets, aug_model, device):
        self.config = config
        self.datasets = datasets
        self.aug_model = aug_model
        self.device = device

    def __getitem__(self, i):
        return tuple(d[i %len(d)] for d in self.datasets) #all items in tuple should already be tensors

    def __len__(self):
        return max(len(d) for d in self.datasets)

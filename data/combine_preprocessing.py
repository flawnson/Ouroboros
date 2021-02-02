import torch


class CombineDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i %len(d)] for d in self.datasets) #all items in tuple should already be tensors

    def __len__(self):
        return max(len(d) for d in self.datasets)

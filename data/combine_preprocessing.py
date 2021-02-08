import torch


class CombineDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets, splits=None):
        self.datasets = list(datasets)
        #Only take elements in dataset that are defined in the split mask
        for i, mask in enumerate(splits):
            self.datasets[i] = [self.datasets[i][j] for j in range(len(self.datasets[i])) if mask[j]]

    def __getitem__(self, i):
        return tuple(d[i %len(d)] for d in self.datasets) #all items in tuple should already be tensors

    def __len__(self):
        return max(len(d) for d in self.datasets)

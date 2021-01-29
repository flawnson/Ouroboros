import torch
import random


class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, config, *datasets, aug_model, device):
        self.config = config
        self.datasets = datasets
        self.aug_model = aug_model
        self.device = device

    def __getitem__(self, i):
        return tuple(d[i %len(d)] for d in self.datasets) #all items in tuple should already be tensors

    def __len__(self):
        return max(len(d) for d in self.datasets)

    def processing(self):
        params_data = torch.eye(self.aug_model.num_params, device=self.device)
        index_list = list(range(self.aug_model.num_params))
        random.shuffle(params_data)
        # divide into training/val
        split = int(len(params_data) * self.configs["train_size"])
        train_params = params_data[:split]
        train_idx = index_list[:split]
        test_params = params_data[split:]
        test_idx = index_list[split:]

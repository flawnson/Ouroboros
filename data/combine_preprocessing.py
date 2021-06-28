import torch

import optim


class CombineImageDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):

        #Extract the relevant indices
        # Assumes that aux data is first and param data is second
        # Iterate over MNIST and Model Params to return a sample of each (image and intger index)

        out = []
        for idx in range(len(self.datasets)):
            data_idx = i % len(self.datasets[idx])
            if isinstance(self.datasets[idx], torch.utils.data.Dataset):
                out.append(self.datasets[idx][data_idx])  # return an integer index
            elif isinstance(self.datasets[idx], optim.parameters.ModelParameters):
                out.append(self.datasets[idx].params[data_idx])  # return an integer index
            else:
                raise TypeError(f"Provided dataset {self.dataset[i]} cannot be combined")

        return out #all items in tuple should already be tensors

    def __len__(self):
        return max(len(self.datasets[0]), len(self.datasets[1]))

    def get_aux_data_len(self):
        return len(self.datasets[0])

    def get_param_data_len(self):
        return len(self.datasets[1])


class TextDataset(torch.utils.data.IterableDataset):
    def __init__(self, subset, datasets):
        self.subset = subset
        self.datasets = datasets

    def __iter__(self):
        return iter(self.datasets)

    def __len__(self):
        return len(self.datasets)


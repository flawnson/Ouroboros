import torch


class CombineDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):

        #Extract the relevant indices
        # Assumes that aux data is first and param data is second

        #MNIST
        aux_data_idx = i % len(self.datasets[0])
        a = self.datasets[0][aux_data_idx] #aux data

        #Model Param (Quine)
        param_data_idx = i % len(self.datasets[1])
        b = self.datasets[1].params[param_data_idx] #return an integer index
        
        return (a, b) #all items in tuple should already be tensors

    def __len__(self):
        return max(len(self.datasets[0]), len(self.datasets[1]))

#
# class CombineDataset(torch.utils.data.Dataset):
#     def __init__(self, *datasets):
#         self.datasets = datasets
#
#     def __getitem__(self, i):
#         return tuple(d[i %len(d)] for d in self.datasets) #all items in tuple should already be tensors
#
#     def __len__(self):
#         return max(len(d) for d in self.datasets)

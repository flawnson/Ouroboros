import torch


class CombineDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets, splits=None, mode="train"):
        self.datasets = list(datasets)
        self.splits = splits
        #hardcoded as if it was only 2 datasets
        #Convert mask array to array of indices
        #I'm not too sure how to use the mask array directly without converting it
        if mode == "train":
            self.splits[0] = [i for i,elem in enumerate(self.splits[0]) if elem == True]
            self.splits[1] = [i for i,elem in enumerate(self.splits[1]) if elem == True]
        else:
            self.splits[0] = [i for i,elem in enumerate(self.splits[0]) if elem == False]
            self.splits[1] = [i for i,elem in enumerate(self.splits[1]) if elem == False]

        # #Only take elements in dataset that are defined in the split mask
        # for i, mask in enumerate(splits):
        #     self.datasets[i] = [self.datasets[i][j] for j in range(len(self.datasets[i])) if mask[j]]

    def __getitem__(self, i):
        #Extract the relevant indices
        aux_data_idx = self.splits[0][i % len(self.datasets[0])]
        a = self.datasets[0][aux_data_idx] #aux data
        param_data_idx = self.splits[1][i % len(self.datasets[1])]
        b = self.datasets[1].get_param(param_data_idx) #param data
        return tuple(a,b) #all items in tuple should already be tensors

    def __len__(self):
        return max(len(self.splits[0]), len(self.splits[1]))

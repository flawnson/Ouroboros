import torch
import random

from models.augmented.quine import Quine
from sklearn.model_selection import StratifiedKFold


class Holdout(object):
    def __init__(self, config, dataset, model, device):
        self.config = config
        self.dataset = dataset
        self.model = model
        self.device = device

    def data_split(self):
        # See SciKitLearn's documentation for implementation details (note that this method enforces same size splits):
        # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
        split = StratifiedKFold(n_splits=len(self.data_config["splits"]), shuffle=self.data_config["shuffle"])
        # split = StratifiedShuffleSplit(n_splits=len(self.data_config["splits"]))
        masks = list(split._iter_test_masks(self.dataset.ndata["x"], self.dataset.ndata["y"]))

        return dict(zip(self.data_config["splits"].keys(), masks))

    def aug_model_split(self):
        params_data = torch.eye(self.model.num_params, device=self.device)
        index_list = list(range(self.model.num_params))
        random.shuffle(params_data)
        # divide into training/val
        split = int(len(params_data) * self.config["train_size"])
        train_params = params_data[:split]
        train_idx = index_list[:split]
        test_params = params_data[split:]
        test_idx = index_list[split:]

    def split(self):
        self.data_split()
        if isinstance(self.model, Quine):
            self.aug_model_split()

        return

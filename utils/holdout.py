import torch
import random
import logzero

import numpy as np

from typing import *
from abc import ABC, abstractmethod
from logzero import logger
from torch.nn import Module
from sklearn.model_selection import StratifiedKFold, train_test_split, ShuffleSplit
from torch.utils.data import Dataset, TensorDataset, DataLoader

from data.combine_preprocessing import CombineDataset
from models.augmented.quine import Quine

DEFAULT_SPLIT = 0.70


class AbstractSplit(ABC):
    def __init__(self, config, dataset, device):
        self.data_config = config["data_config"]
        self.dataset = dataset
        self.device = device

    @abstractmethod
    def holdout(self):
        pass

    @abstractmethod
    def kfold(self):
        # See SciKitLearn's documentation for implementation details (note that this method enforces same size splits):
        # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
        splits = StratifiedKFold(n_splits=len(self.data_config["splits"]), shuffle=self.data_config["shuffle"])
        # split = StratifiedShuffleSplit(n_splits=len(self.data_config["splits"]))
        y = [self.dataset[y][1] for y, d in enumerate(self.dataset)]
        masks = list(splits._iter_test_masks(self.dataset, y))

        return dict(zip(self.data_config["splits"].keys(), masks))

    @staticmethod
    @abstractmethod
    def type_check(subject):
        pass

    def partition(self):
        self.type_check(self.dataset)
        if self.data_config["split_type"] == "stratified":
            return self.kfold()
        elif self.data_config["split_type"] == "holdout":
            return self.holdout()
        else:
            raise NotImplementedError(f"Split-type: {self.data_config['split_type']} not understood")


class MNISTSplit(AbstractSplit):
    def __init__(self, config, dataset, param_data, device):
        super(MNISTSplit, self).__init__(config, dataset, device)
        self.config = config
        self.data_config = config["data_config"]
        self.dataset = dataset
        self.param_data = param_data.params
        self.device = device

    def holdout(self) -> Dict[str, DataLoader]:
        try:
            split_size = self.data_config["splits"]["size"]
            logger.info(f"Splitting dataset into {self.data_config['splits']['size']}")
        except KeyError:
            split_size = DEFAULT_SPLIT
            logger.info(f"Could not find split size in config, splitting dataset into {DEFAULT_SPLIT}")

        # train_x, test_x, train_y, test_y = train_test_split(self.dataset, self.dataset.targets, train_size=split_size, random_state=self.config["seed"])
        split_idx = list(ShuffleSplit(n_splits=1, train_size=split_size, random_state=self.config["seed"]).split(self.dataset, self.dataset.targets))
        samplers = [torch.utils.data.SubsetRandomSampler(idx_array) for idx_array in split_idx]
        dataloaders = [DataLoader(self.dataset, sampler=sampler) for sampler in samplers]
        return dict(zip([f"split_{x}" for x in range(1, self.data_config["num_splits"])], dataloaders))

    def kfold(self) -> Dict[str, DataLoader]:
        # See SciKitLearn's documentation for implementation details (note that this method enforces same size splits):
        # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
        splits = StratifiedKFold(n_splits=len(self.data_config["splits"]), shuffle=self.data_config["shuffle"])
        # split = StratifiedShuffleSplit(n_splits=len(self.data_config["splits"]))
        # The target labels (stratified k fold needs the labels to preserve label distributions in each split)
        # The .split() method from SKLearn returns a generator that generates 2 index arrays (for training and testing)
        samplers = [torch.utils.data.SubsetRandomSampler(idx) for idx in splits.split(self.dataset, self.dataset.targets)]
        if self.config["model_aug_config"]["model_augmentation"] == "auxiliary":
            dataloaders = [DataLoader(CombineDataset(self.dataset, self.param_data), sampler=sampler) for sampler in samplers]
        else:
            dataloaders = [DataLoader(self.dataset, sampler=sampler) for sampler in samplers]

        return dict(zip([f"split_{x}" for x in range(1, self.data_config["num_splits"])], dataloaders))

    @staticmethod
    def type_check(subject):
        assert isinstance(subject, torch.utils.data.Dataset), f"Subject: {type(subject)} is not a splittable type"


class GraphSplit(AbstractSplit):
    def __init__(self, config, dataset, model, device):
        super(GraphSplit, self).__init__(config, dataset, device)
        self.data_config = config["data_config"]
        self.dataset = dataset
        self.model = model
        self.device = device

    def holdout(self):
        pass

    def kfold(self):
        # See SciKitLearn's documentation for implementation details (note that this method enforces same size splits):
        # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
        splits = StratifiedKFold(n_splits=len(self.data_config["splits"]), shuffle=self.data_config["shuffle"])
        # split = StratifiedShuffleSplit(n_splits=len(self.data_config["splits"]))
        # The target labels (stratified k fold needs the labels to preserve label distributions in each split
        y = [self.dataset[y][1] for y, d in enumerate(self.dataset)]
        masks = list(splits._iter_test_masks(self.dataset, y))

        return dict(zip(self.data_config["splits"].keys(), masks))

    @staticmethod
    def type_check(subject):
        assert isinstance(subject, torch.utils.data.Dataset), f"Subject: {type(subject)} is not a splittable type"


class QuineSplit(AbstractSplit):
    def __init__(self, config, dataset, device):
        super(QuineSplit, self).__init__(config, dataset, device)
        self.data_config = config["data_config"]
        self.dataset = dataset
        self.device = device

    def holdout(self):
        pass

    def kfold(self):
        # See SciKitLearn's documentation for implementation details (note that this method enforces same size splits):
        # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
        splits = StratifiedKFold(n_splits=len(self.data_config["splits"]), shuffle=self.data_config["shuffle"])
        # split = StratifiedShuffleSplit(n_splits=len(self.data_config["splits"]))

        #all labels are 0
        #Does subject need to be a torch tensor
        samplers = [torch.utils.data.SubsetRandomSampler(idx) for idx in splits.split(self.dataset, torch.zeros_like(torch.tensor(self.dataset)))]

        return dict(zip(self.data_config["splits"].keys(), samplers))

    @staticmethod
    def type_check(subject):
        pass
        # assert isinstance(subject, Quine) & isinstance(subject, Module), f"Subject: {type(subject)} is not a splittable type"

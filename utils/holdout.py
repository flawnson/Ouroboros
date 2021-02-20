import torch
import random
import logzero

import numpy as np

from abc import ABC, abstractmethod
from logzero import logger
from torch.nn import Module
from torch.utils.data import Dataset
from models.augmented.quine import Quine
from sklearn.model_selection import StratifiedKFold, train_test_split


class AbstractSplit(ABC):
    def __init__(self, config, dataset, model, device):
        self.data_config = config["data_config"]
        self.dataset = dataset
        self.model = model
        self.device = device

    @abstractmethod
    def holdout(self, subject):
        pass

    @abstractmethod
    def kfold(self, subject):
        # See SciKitLearn's documentation for implementation details (note that this method enforces same size splits):
        # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
        splits = StratifiedKFold(n_splits=len(self.data_config["splits"]), shuffle=self.data_config["shuffle"])
        # split = StratifiedShuffleSplit(n_splits=len(self.data_config["splits"]))
        y = [subject[y][1] for y, d in enumerate(subject)]
        masks = list(splits._iter_test_masks(subject, y))

        return dict(zip(self.data_config["splits"].keys(), masks))

    @staticmethod
    @abstractmethod
    def type_check(subject):
        pass

    def partition(self, subject):
        self.type_check(subject)
        if self.data_config["split_type"] == "stratified":
            return self.kfold(subject)
        elif self.data_config["split_type"] == "holdout":
            return self.holdout(subject)
        else:
            raise NotImplementedError(f"Split-type: {self.data_config['split_type']} not understood")


class MNISTSplit(AbstractSplit):
    def __init__(self, config, dataset, model, device):
        super(MNISTSplit, self).__init__(config, dataset, model, device)
        self.data_config = config["data_config"]
        self.dataset = dataset
        self.model = model
        self.device = device

    def holdout(self, subject):
        train_test_split(subject, subject.targets, train_size=self.data_config.get("splits", .70), random_state=42)

    def kfold(self, subject):
        # See SciKitLearn's documentation for implementation details (note that this method enforces same size splits):
        # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
        splits = StratifiedKFold(n_splits=len(self.data_config["splits"]), shuffle=self.data_config["shuffle"])
        # split = StratifiedShuffleSplit(n_splits=len(self.data_config["splits"]))
        # The target labels (stratified k fold needs the labels to preserve label distributions in each split)
        # The .split() method from SKLearn returns a generator that generates 2 index arrays (for training and testing)
        samplers = [torch.utils.data.SubsetRandomSampler(idx) for idx in splits.split(subject, subject.targets)]

        if len(mnist_samplers) > len(quine_samplers):
            dataloaders = [DataLoader(CombineDataset(datasets, param_data), sampler=sampler) for sampler in mnist_samplers.partition(datasets).values()]
        else:
            #When splitting/partition, we split the indices of the params (which are ints)
            #In combineDataset, the param_data indices will be passed to get_param() in get_item
            dataloaders = [DataLoader(CombineDataset(datasets, param_data), sampler=sampler) for sampler in quine_samplers.partition(param_data.params).values()]

        return dict(zip(self.data_config["splits"].keys(), samplers))

    @staticmethod
    def type_check(subject):
        assert isinstance(subject, torch.utils.data.Dataset), f"Subject: {type(subject)} is not a splittable type"


class GraphSplit(AbstractSplit):
    def __init__(self, config, dataset, model, device):
        super(MNISTSplit, self).__init__(config, dataset, model, device)
        self.data_config = config["data_config"]
        self.dataset = dataset
        self.model = model
        self.device = device

    def holdout(self, subject):
        pass

    def kfold(self, subject):
        # See SciKitLearn's documentation for implementation details (note that this method enforces same size splits):
        # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
        splits = StratifiedKFold(n_splits=len(self.data_config["splits"]), shuffle=self.data_config["shuffle"])
        # split = StratifiedShuffleSplit(n_splits=len(self.data_config["splits"]))
        # The target labels (stratified k fold needs the labels to preserve label distributions in each split
        y = [subject[y][1] for y, d in enumerate(subject)]
        masks = list(splits._iter_test_masks(subject, y))

        return dict(zip(self.data_config["splits"].keys(), masks))

    @staticmethod
    def type_check(subject):
        assert isinstance(subject, torch.utils.data.Dataset), f"Subject: {type(subject)} is not a splittable type"


class QuineSplit(AbstractSplit):
    def __init__(self, config, dataset, model, device):
        super(QuineSplit, self).__init__(config, dataset, model, device)
        self.data_config = config["data_config"]
        self.dataset = dataset
        self.model = model
        self.device = device

    def holdout(self, subject):
        pass

    def kfold(self, subject):
        # See SciKitLearn's documentation for implementation details (note that this method enforces same size splits):
        # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
        splits = StratifiedKFold(n_splits=len(self.data_config["splits"]), shuffle=self.data_config["shuffle"])
        # split = StratifiedShuffleSplit(n_splits=len(self.data_config["splits"]))

        #all labels are 0
        #Does subject need to be a torch tensor
        samplers = [torch.utils.data.SubsetRandomSampler(idx) for idx in splits.split(subject, torch.zeros_like(torch.tensor(subject)))]

        return dict(zip(self.data_config["splits"].keys(), samplers))

    @staticmethod
    def type_check(subject):
        pass
        # assert isinstance(subject, Quine) & isinstance(subject, Module), f"Subject: {type(subject)} is not a splittable type"

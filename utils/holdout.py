import torch
import random
import logzero

import numpy as np

from abc import ABC, abstractmethod
from logzero import logger
from torch.nn import Module
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader

from data.combine_preprocessing import CombineDataset
from models.augmented.quine import Quine

DEFAULT_SPLIT = 0.70


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
    def __init__(self, config, dataset, param_data, model, device):
        super(MNISTSplit, self).__init__(config, dataset, model, device)
        self.config = config
        self.data_config = config["data_config"]
        self.dataset = dataset
        self.param_data = param_data.params
        self.model = model
        self.device = device

    def holdout(self, subject) -> Dict[str, DataLoader]:
        try:
            logger.info(f"Splitting dataset into {self.data_config['splits']['size']}")
            train_x, test_x, train_y, test_y = train_test_split(subject, subject.targets, train_size=self.data_config["splits"]["size"], random_state=42)
        except KeyError:
            logger.info(f"Could not find split size in config, splitting dataset into {DEFAULT_SPLIT}")
            train_x, test_x, train_y, test_y = train_test_split(subject, subject.targets, train_size=DEFAULT_SPLIT, random_state=42)

        dataloaders = [DataLoader(Dataset(trainset, labelset)) for trainset, labelset in zip([[train_x, test_x], [train_y, test_y]])]

        return dict(zip(self.data_config["splits"].keys(), dataloaders))

    def kfold(self, subject) -> Dict[str, DataLoader]:
        # See SciKitLearn's documentation for implementation details (note that this method enforces same size splits):
        # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
        splits = StratifiedKFold(n_splits=len(self.data_config["splits"]), shuffle=self.data_config["shuffle"])
        # split = StratifiedShuffleSplit(n_splits=len(self.data_config["splits"]))
        # The target labels (stratified k fold needs the labels to preserve label distributions in each split)
        # The .split() method from SKLearn returns a generator that generates 2 index arrays (for training and testing)
        samplers = [torch.utils.data.SubsetRandomSampler(idx) for idx in splits.split(subject, subject.targets)]
        if self.config["model_aug_config"]["model_augmentation"] == "auxiliary":
            dataloaders = [DataLoader(CombineDataset(self.dataset, self.param_data), sampler=sampler) for sampler in samplers]
        else:
            dataloaders = [DataLoader(self.dataset, sampler=sampler) for sampler in samplers]

        return dict(zip(self.data_config["splits"].keys(), dataloaders))

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
        self.dataset = dataset.params
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

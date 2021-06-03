import torch
import random
import logzero

import numpy as np

from typing import *
from abc import ABC, abstractmethod
from logzero import logger
from torch.nn import Module
from sklearn.model_selection import StratifiedKFold, train_test_split, ShuffleSplit
from torch.utils.data import Dataset, DataLoader

from data.combine_preprocessing import CombineDataset
from models.augmented.quine import Quine

DEFAULT_SPLIT = 0.70


class AbstractSplit(ABC):
    def __init__(self, config, dataset, param_data, device):
        self.config = config
        self.data_config = config["data_config"]
        self.dataset = dataset
        self.param_data = param_data
        self.device = device

    @abstractmethod
    def holdout(self):
        try:
            split_size = self.data_config["splits"]["size"]
            logger.info(f"Splitting dataset into {self.data_config['splits']['size']}")
        except KeyError as e:
            split_size = DEFAULT_SPLIT
            logger.error(e)
            logger.info(f"Could not find split size in config, splitting dataset into {DEFAULT_SPLIT}")

        # train_x, test_x, train_y, test_y = train_test_split(self.dataset, self.dataset.targets, train_size=split_size, random_state=self.config["seed"])
        split_idx = list(ShuffleSplit(n_splits=1, train_size=split_size, random_state=self.config["seed"]).split(self.dataset, self.dataset.targets))
        samplers = [torch.utils.data.SubsetRandomSampler(idx_array) for idx_array in split_idx]
        dataloaders = [DataLoader(self.dataset, sampler=sampler) for sampler in samplers]
        return dict(zip([f"split_{x}" for x in range(1, self.data_config["num_splits"])], dataloaders))

    @abstractmethod
    def kfold(self):
        # See SciKitLearn's documentation for implementation details (note that this method enforces same size splits):
        # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
        splits = StratifiedKFold(n_splits=len(self.data_config["num_splits"]), shuffle=self.data_config["shuffle"])
        # split = StratifiedShuffleSplit(n_splits=len(self.data_config["splits"]))
        # The target labels (stratified k fold needs the labels to preserve label distributions in each split)
        # The .split() method from SKLearn returns a generator that generates 2 index arrays (for training and testing)
        samplers = [torch.utils.data.SubsetRandomSampler(idx) for idx in splits.split(self.dataset, self.dataset.targets)]
        dataloaders = self.get_dataloaders(samplers)

        return dict(zip([f"split_{x}" for x in range(1, self.data_config["num_splits"])], dataloaders))

    @staticmethod
    @abstractmethod
    def type_check(subject):
        pass

    def get_dataloaders(self, samplers: List[torch.utils.data.Sampler]) -> List:
        if self.config["model_aug_config"]["model_augmentation"] == "auxiliary":
            combined_dataset = CombineDataset(self.dataset, self.param_data)
            return [DataLoader(combined_dataset,
                                      batch_size=self.config["data_config"]["batch_size"],
                                      sampler=sampler) for sampler in samplers]
        elif self.config["model_aug_config"]["model_augmentation"] == "vanilla":
            return [DataLoader(self.param_data,
                               batch_size=self.config["data_config"]["batch_size"],
                               sampler=sampler) for sampler in samplers]
        else:
            return [DataLoader(self.dataset,
                               batch_size=self.config["data_config"]["batch_size"],
                               sampler=sampler) for sampler in samplers]

    def partition(self):
        self.type_check(self.dataset)
        if self.data_config["split_type"] == "stratified":
            return self.kfold()
        elif self.data_config["split_type"] == "holdout":
            return self.holdout()
        else:
            raise NotImplementedError(f"Split-type: {self.data_config['split_type']} not understood")


class MNISTSplit(AbstractSplit):
    def __init__(self, config, dataset, param_data, larger_dataset, device):
        super(MNISTSplit, self).__init__(config, dataset, param_data, device)
        self.config = config
        self.data_config = config["data_config"]
        self.dataset = dataset
        self.param_data = param_data  # Needed for Aux models
        self.larger_dataset = larger_dataset
        self.device = device

    def holdout(self) -> Dict[str, DataLoader]:
        try:
            split_size = self.data_config["splits"]["size"]
            logger.info(f"Splitting dataset into {self.data_config['splits']['size']}")
        except KeyError as e:
            split_size = DEFAULT_SPLIT
            logger.info(e)
            logger.info(f"Could not find split size in config, splitting dataset into {DEFAULT_SPLIT}")

        # train_x, test_x, train_y, test_y = train_test_split(self.dataset, self.dataset.targets, train_size=split_size, random_state=self.config["seed"])
        split_idx = None
        if self.larger_dataset == "aux_data":
            split_idx = list(ShuffleSplit(n_splits=1, train_size=split_size, random_state=self.config["seed"]).split(self.dataset, self.dataset.targets))
        elif self.larger_dataset == "param_data":
            split_idx = list(ShuffleSplit(n_splits=1, train_size=split_size, random_state=self.config["seed"]).split(self.param_data.params))
        samplers = [torch.utils.data.SubsetRandomSampler(idx_array) for idx_array in split_idx[0]]
        dataloaders = self.get_dataloaders(samplers)

        return dict(zip([f"split_{x}" for x in range(1, self.data_config["num_splits"])], dataloaders))

    def kfold(self) -> Dict[str, DataLoader]:
        # See SciKitLearn's documentation for implementation details (note that this method enforces same size splits):
        # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
        splits = StratifiedKFold(n_splits=len(self.data_config["num_splits"]), shuffle=self.data_config["shuffle"])
        # split = StratifiedShuffleSplit(n_splits=len(self.data_config["splits"]))
        # The target labels (stratified k fold needs the labels to preserve label distributions in each split)
        # The .split() method from SKLearn returns a generator that generates 2 index arrays (for training and testing)
        samplers = [torch.utils.data.SubsetRandomSampler(idx) for idx in splits.split(self.dataset, self.dataset.targets)]
        dataloaders = self.get_dataloaders(samplers)

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
        splits = StratifiedKFold(n_splits=len(self.data_config["num_splits"]), shuffle=self.data_config["shuffle"])
        # split = StratifiedShuffleSplit(n_splits=len(self.data_config["splits"]))
        # The target labels (stratified k fold needs the labels to preserve label distributions in each split
        y = [self.dataset[y][1] for y, d in enumerate(self.dataset)]
        masks = list(splits._iter_test_masks(self.dataset, y))

        return dict(zip(self.data_config["splits"].keys(), masks))

    @staticmethod
    def type_check(subject):
        assert isinstance(subject, torch.utils.data.Dataset), f"Subject: {type(subject)} is not a splittable type"


class QuineSplit(AbstractSplit):
    def __init__(self, config, dataset, param_data, device):
        super(QuineSplit, self).__init__(config, dataset, param_data, device)
        self.config = config
        self.data_config = config["data_config"]
        self.dataset = dataset
        self.param_data = param_data  # Needed for Aux models
        self.device = device

    def holdout(self):
        # When splitting/partition, we split the indices of the params (which are ints)
        # In combineDataset, the param_data indices will be passed to get_param() in get_item
        try:
            split_size = self.data_config["splits"]["size"]
            logger.info(f"Splitting dataset into {self.data_config['splits']['size']}")
        except KeyError as e:
            split_size = DEFAULT_SPLIT
            logger.error(e)
            logger.info(f"Could not find split size in config, splitting dataset into {DEFAULT_SPLIT}")

        # train_x, test_x, train_y, test_y = train_test_split(self.dataset, self.dataset.targets, train_size=split_size, random_state=self.config["seed"])
        split_idx = list(ShuffleSplit(n_splits=1, train_size=split_size, random_state=self.config["seed"]).split(self.param_data.params))
        split_idx = split_idx[0]
        samplers = [torch.utils.data.SubsetRandomSampler(idx_array) for idx_array in split_idx]
        dataloaders = self.get_dataloaders(samplers)

        return dict(zip([f"split_{x}" for x in range(1, self.data_config["num_splits"])], dataloaders))

    def kfold(self):
        # See SciKitLearn's documentation for implementation details (note that this method enforces same size splits):
        # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
        splits = StratifiedKFold(n_splits=len(self.data_config["num_splits"]), shuffle=self.data_config["shuffle"])
        # split = StratifiedShuffleSplit(n_splits=len(self.data_config["splits"]))
        # The target labels (stratified k fold needs the labels to preserve label distributions in each split)
        # The .split() method from SKLearn returns a generator that generates 2 index arrays (for training and testing)
        samplers = [torch.utils.data.SubsetRandomSampler(idx) for idx in splits.split(self.dataset, self.dataset.targets)]
        dataloaders = self.get_dataloaders(samplers)

        return dict(zip([f"split_{x}" for x in range(1, self.data_config["num_splits"])], dataloaders))

    @staticmethod
    def type_check(subject):
        pass
        # assert isinstance(subject, Quine) & isinstance(subject, Module), f"Subject: {type(subject)} is not a splittable type"


def get_image_data_split(config, datasets, param_data, device):
    # Function works for both MNIST and CIFAR10 (untested for other datasets)
    if (param_data is not None) and len(datasets) < len(param_data):
        dataloaders = MNISTSplit(config, datasets, param_data, "param_data",
                                 device).partition()  # MNIST split appears to work fine with CIFAR
    else:
        dataloaders = MNISTSplit(config, datasets, param_data, "aux_data", device).partition()

    # Special case if Vanilla
    if config["model_aug_config"]["model_augmentation"].casefold() == "vanilla":
        logger.info("Using QuineSplit for Vanilla")
        dataloaders = QuineSplit(config, datasets, param_data, device).partition()

    return dataloaders


def get_text_data_split(config, datasets, param_data, device):
    # Function works for both MNIST and CIFAR10 (untested for other datasets)
    if (param_data is not None) and len(datasets) < len(param_data):
        dataloaders = MNISTSplit(config, datasets, param_data, "param_data",
                                 device).partition()  # MNIST split appears to work fine with CIFAR
    else:
        dataloaders = MNISTSplit(config, datasets, param_data, "aux_data", device).partition()

    # Special case if Vanilla
    if config["model_aug_config"]["model_augmentation"].casefold() == "vanilla":
        logger.info("Using QuineSplit for Vanilla")
        dataloaders = QuineSplit(config, datasets, param_data, device).partition()

    return dataloaders





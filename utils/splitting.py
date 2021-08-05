import math
import torch
import random
import logzero

import numpy as np

from typing import *
from abc import ABC, abstractmethod
from logzero import logger
from torch.nn import Module
from sklearn.model_selection import StratifiedKFold, train_test_split, ShuffleSplit, LeavePOut
from torch.utils.data import Dataset, DataLoader, Subset

from data.combine_preprocessing import CombineImageDataset
from models.augmented.quine import Quine


class AbstractSplit(ABC):
    def __init__(self, config: Dict, dataset, param_data, device: torch.device):
        self.config = config
        self.data_config = config["data_config"]
        self.dataset = dataset
        self.param_data = param_data
        self.device = device

    def binary(self):
        # FIXME: NOT FUNCTIONAL
        datasets = [torch.utils.data.TensorDataset(torch.Tensor(list(zip(x, x)))) for x in iter(splits)]
        splits = train_test_split([self.dataset.datasets[x].data for x in range(len(self.dataset.datasets))],
                                  **self.data_config["split_kwargs"],
                                  random_state=self.config["seed"])
        dataloaders = [DataLoader(dataset, batch_size=self.config["data_config"]["batch_size"]) for dataset in datasets]
        # Organizing datalaoders into dictionary
        dataloaders = dict(zip([f"split_{x}" for x in range(1, len(dataloaders))], dataloaders))
        # Creating dataloaders for the param_data
        dataloaders = {name: [dataloader, DataLoader(self.param_data.params)] for
                       name, dataloader in dataloaders.items()}

        return dataloaders

    def holdout(self):
        aux_p = math.floor(self.data_config["split_kwargs"]["test_size"] * len(self.dataset))
        aux_split_idx = LeavePOut(aux_p).split(self.dataset, self.dataset.targets)
        param_p = self.data_config["split_kwargs"]["test_size"] * len(self.param_data)
        param_split_idx = LeavePOut(param_p).split(self.dataset, self.dataset.targets)

        aux_samplers = [torch.utils.data.SubsetRandomSampler(idx_array) for idx_array in aux_split_idx[0]]
        param_samplers = [torch.utils.data.SubsetRandomSampler(idx_array) for idx_array in param_split_idx[0]]
        # Getting datalaoders from the aux sampler
        dataloaders = self.get_dataloaders(subsets=[None] * len(aux_samplers), samplers=aux_samplers)
        # Organizing datalaoders into dictionary
        dataloaders = dict(zip([f"split_{x}" for x in range(1, len(dataloaders))], dataloaders))
        # Creating dataloaders for the param_data
        dataloaders = {name: [dataloader, DataLoader(self.param_data.params, sampler=param_samplers)] for
                       name, dataloader in dataloaders.items()}

        return dataloaders

    def shuffle(self):
        split_idx = list(ShuffleSplit(**self.data_config["split_kwargs"],
                                      random_state=self.config["seed"]).split(self.dataset,
                                                                              self.dataset.targets))
        samplers = [torch.utils.data.SubsetRandomSampler(idx_array) for idx_array in split_idx]
        dataloaders = [DataLoader(self.dataset, sampler=sampler) for sampler in samplers]
        return dict(zip([f"split_{x}" for x in range(1, len(dataloaders))], dataloaders))

    def kfold(self):
        # See SciKitLearn's documentation for implementation details (note that this method enforces same size splits):
        # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
        splits = StratifiedKFold(**self.data_config["split_kwargs"],
                                 random_state=self.config["seed"])
        # split = StratifiedShuffleSplit(n_splits=len(self.data_config["splits"]))
        # The target labels (stratified k fold needs the labels to preserve label distributions in each split)
        # The .split() method from SKLearn returns a generator that generates 2 index arrays (for training and testing)
        samplers = [torch.utils.data.SubsetRandomSampler(idx) for idx in splits.split(self.dataset, self.dataset.targets)]
        dataloaders = self.get_dataloaders(subsets=[None]*len(samplers), samplers=samplers)

        return dict(zip([f"split_{x}" for x in range(1, len(dataloaders))], dataloaders))

    @staticmethod
    @abstractmethod
    def type_check(subject):
        pass

    def get_datasets(self):
        if self.config["model_config"]["model_type"] in ("linear", "image"):
            return CombineImageDataset(self.dataset, self.param_data)
        elif self.config["model_config"]["model_type"] == "sequential":
            return list(zip(self.dataset, self.dataset.targets))
        else:
            raise TypeError(f"Model type: {self.config['model_config']['model_type']} cannot combine with param_data")

    def get_dataloaders(self,
                        samplers: List[Optional[torch.utils.data.Sampler]] = [None]) -> List[DataLoader]:
        if self.config["model_aug_config"]["model_augmentation"] == "auxiliary":
            return [DataLoader(self.get_datasets(),
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

    def partition(self) -> Dict:
        self.type_check(self.dataset)
        if self.data_config["split_type"] == "kfold":
            return self.kfold()  # Stratified k-fold
        elif self.data_config["split_type"] == "shuffle":
            return self.shuffle()  # Shuffle splitting
        elif self.data_config["split_type"] == "holdout":
            return self.holdout()  # Hold p out
        elif self.data_config["split_type"] == "binary":
            return self.binary()  # Train test splitting
        else:
            raise NotImplementedError(f"Split-type: {self.data_config['split_type']} not understood")


class ImageDataSplit(AbstractSplit):
    def __init__(self, config, dataset, param_data, larger_dataset, device):
        super(ImageDataSplit, self).__init__(config, dataset, param_data, device)
        self.config = config
        self.data_config = config["data_config"]
        self.dataset = dataset
        self.param_data = param_data  # Needed for Aux models
        self.larger_dataset = larger_dataset
        self.device = device

    def shuffle(self) -> Dict[str, DataLoader]:
        split_idx = None
        if self.larger_dataset == "aux_data":
            split_idx = list(ShuffleSplit(**self.data_config["split_kwargs"],
                                          random_state=self.config["seed"]).split(self.dataset,
                                                                                  self.dataset.targets))
        elif self.larger_dataset == "param_data":
            split_idx = list(ShuffleSplit(**self.data_config["split_kwargs"],
                                          random_state=self.config["seed"]).split(self.param_data.params))

        samplers = [torch.utils.data.SubsetRandomSampler(idx_array) for idx_array in split_idx[0]]
        dataloaders = self.get_dataloaders(samplers)

        return dict(zip([f"split_{x + 1}" for x in range(len(dataloaders))], dataloaders))

    def kfold(self) -> Dict[str, DataLoader]:
        # See SciKitLearn's documentation for implementation details (note that this method enforces same size splits):
        # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
        splits = StratifiedKFold(**self.data_config["split_kwargs"])
        # split = StratifiedShuffleSplit(n_splits=len(self.data_config["splits"]))
        # The target labels (stratified k fold needs the labels to preserve label distributions in each split)
        # The .split() method from SKLearn returns a generator that generates 2 index arrays (for training and testing)
        samplers = [torch.utils.data.SubsetRandomSampler(idx) for idx in splits.split(self.dataset, self.dataset.targets)]
        dataloaders = self.get_dataloaders(samplers)

        return dict(zip([f"split_{x + 1}" for x in range(len(dataloaders))], dataloaders))

    @staticmethod
    def type_check(subject):
        assert isinstance(subject, torch.utils.data.Dataset), f"Subject: {type(subject)} is not a splittable type"


class GraphDataSplit(AbstractSplit):
    def __init__(self, config, dataset, model, device):
        super(GraphDataSplit, self).__init__(config, dataset, device)
        self.data_config = config["data_config"]
        self.dataset = dataset
        self.model = model
        self.device = device

    def shuffle(self):
        pass

    def kfold(self):
        # See SciKitLearn's documentation for implementation details (note that this method enforces same size splits):
        # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
        splits = StratifiedKFold(**self.data_config["split_kwargs"],
                                 random_state=self.config["seed"])
        # split = StratifiedShuffleSplit(n_splits=len(self.data_config["splits"]))
        # The target labels (stratified k fold needs the labels to preserve label distributions in each split
        y = [self.dataset[y][1] for y, d in enumerate(self.dataset)]
        masks = list(splits._iter_test_masks(self.dataset, y))

        return dict(zip(self.data_config["splits"].keys(), masks))

    @staticmethod
    def type_check(subject):
        assert isinstance(subject, torch.utils.data.Dataset), f"Subject: {type(subject)} is not a splittable type"


class QuineDataSplit(AbstractSplit):
    def __init__(self, config, dataset, param_data, device):
        super(QuineDataSplit, self).__init__(config, dataset, param_data, device)
        self.config = config
        self.data_config = config["data_config"]
        self.dataset = dataset
        self.param_data = param_data  # Needed for Aux models
        self.device = device

    def shuffle(self):
        # When splitting/partition, we split the indices of the params (which are ints)
        # In combineDataset, the param_data indices will be passed to get_param() in get_item

        # train_x, test_x, train_y, test_y = train_test_split(self.dataset, self.dataset.targets, train_size=split_size, random_state=self.config["seed"])
        split_idx = list(ShuffleSplit(**self.data_config["split_kwargs"],
                                      random_state=self.config["seed"]).split(self.param_data.params))
        split_idx = split_idx[0]
        samplers = [torch.utils.data.SubsetRandomSampler(idx_array) for idx_array in split_idx]
        dataloaders = self.get_dataloaders(samplers)

        return dict(zip([f"split_{x + 1}" for x in range(len(dataloaders))], dataloaders))

    def kfold(self) -> Dict[str, DataLoader]:
        # See SciKitLearn's documentation for implementation details (note that this method enforces same size splits):
        # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
        splits = StratifiedKFold(**self.data_config["split_kwargs"],
                                 random_state=self.config["seed"])
        # split = StratifiedShuffleSplit(n_splits=len(self.data_config["splits"]))
        # The target labels (stratified k fold needs the labels to preserve label distributions in each split)
        # The .split() method from SKLearn returns a generator that generates 2 index arrays (for training and testing)
        samplers = [torch.utils.data.SubsetRandomSampler(idx) for idx in splits.split(self.dataset, self.dataset.targets)]
        dataloaders = self.get_dataloaders(samplers)

        return dict(zip([f"split_{x + 1}" for x in range(len(dataloaders))], dataloaders))

    @staticmethod
    def type_check(subject):
        pass
        # assert isinstance(subject, Quine) & isinstance(subject, Module), f"Subject: {type(subject)} is not a splittable type"


class TextDataSplit(AbstractSplit):
    def __init__(self, config, dataset, param_data, larger_dataset, device):
        super(TextDataSplit, self).__init__(config, dataset, param_data, device)
        self.config = config
        self.data_config = config["data_config"]
        self.dataset = dataset
        self.param_data = param_data  # Needed for Aux models
        self.larger_dataset = larger_dataset
        self.device = device

    def binary(self) -> Dict[str, List[DataLoader]]:
        splits = train_test_split(self.dataset, self.dataset.targets, self.data_config["split_kwargs"], random_state=self.config["seed"])
        datasets = [torch.utils.data.TensorDataset(torch.Tensor(list(zip(x, x)))) for x in iter(splits)]
        dataloaders = [DataLoader(dataset, batch_size=self.config["data_config"]["batch_size"]) for dataset in datasets]
        # Organizing datalaoders into dictionary
        dataloaders = dict(zip([f"split_{x + 1}" for x in range(len(dataloaders))], dataloaders))
        # Creating dataloaders for the param_data
        dataloaders = {name: [dataloader, DataLoader(self.param_data.params, sampler=param_samplers)] for
                       name, dataloader in dataloaders.items()}

        return dataloaders

    def holdout(self) -> Dict[str, List[DataLoader]]:
        aux_p = math.floor(self.data_config["split_kwargs"]["test_size"] * len(self.dataset))
        aux_split_idx = LeavePOut(aux_p).split(self.dataset, self.dataset.targets)
        param_p = self.data_config["split_kwargs"]["test_size"] * len(self.param_data)
        param_split_idx = LeavePOut(param_p).split(self.dataset, self.dataset.targets)

        aux_samplers = [torch.utils.data.SubsetRandomSampler(idx_array) for idx_array in aux_split_idx[0]]
        param_samplers = [torch.utils.data.SubsetRandomSampler(idx_array) for idx_array in param_split_idx[0]]
        # Getting datalaoders from the aux sampler
        dataloaders = self.get_dataloaders(aux_samplers)
        # Organizing datalaoders into dictionary
        dataloaders = dict(zip([f"split_{x + 1}" for x in range(len(dataloaders))], dataloaders))
        # Creating dataloaders for the param_data
        dataloaders = {name: [dataloader, DataLoader(self.param_data.params, sampler=param_samplers)] for
                       name, dataloader in dataloaders.items()}

        return dataloaders

    def shuffle(self) -> Dict[str, List[DataLoader]]:
        # When splitting/partition, we split the indices of the params (which are ints)
        # In combineDataset, the param_data indices will be passed to get_param() in get_item

        aux_split_idx = list(ShuffleSplit(**self.data_config["split_kwargs"],
                                          random_state=self.config["seed"]).split(self.dataset,
                                                                                  self.dataset.targets))
        param_split_idx = list(ShuffleSplit(**self.data_config["split_kwargs"],
                                            random_state=self.config["seed"]).split(self.param_data.params))

        aux_samplers = [torch.utils.data.SubsetRandomSampler(idx_array) for idx_array in aux_split_idx[0]]
        param_samplers = [torch.utils.data.SubsetRandomSampler(idx_array) for idx_array in param_split_idx[0]]
        dataloaders = self.get_dataloaders(aux_samplers)
        dataloaders = dict(zip([f"split_{x + 1}" for x in range(len(dataloaders) + 1)], dataloaders))
        dataloaders = {name: [dataloader, DataLoader(self.param_data.params, sampler=sampler)] for
                       (name, dataloader), sampler in zip(dataloaders.items(), param_samplers)}

        return dataloaders

    @staticmethod
    def type_check(subject):
        pass
        # assert isinstance(subject, Quine) & isinstance(subject, Module), f"Subject: {type(subject)} is not a splittable type"


def get_image_data_split(config, datasets, param_data, device):
    # Function works for both MNIST and CIFAR10 (untested for other datasets)
    larger_dataset = "param_data" if (param_data is not None) and len(datasets) < len(param_data) else "aux_data"
    dataloaders = ImageDataSplit(config, datasets, param_data, larger_dataset, device).partition()  # MNIST split appears to work fine with CIFAR

    # Special case if Vanilla
    if config["model_aug_config"]["model_augmentation"].casefold() == "vanilla":
        logger.info("Using QuineSplit for Vanilla")
        dataloaders = QuineDataSplit(config, datasets, param_data, device).partition()

    return dataloaders


def get_text_data_split(config, datasets, param_data, device) -> Dict[str, List]:
    # Function works for both MNIST and CIFAR10 (untested for other datasets)
    larger_dataset = "param_data" if (param_data is not None) and len(datasets) < len(param_data) else "aux_data"
    dataloaders = TextDataSplit(config, datasets, param_data, larger_dataset, device).partition()  # MNIST split appears to work fine with CIFAR

    # Special case if Vanilla
    if config["model_aug_config"]["model_augmentation"].casefold() == "vanilla":
        logger.info("Using QuineSplit for Vanilla")
        dataloaders = QuineDataSplit(config, datasets, param_data, device).partition()

    return dataloaders





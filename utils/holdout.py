import torch
import random
import logzero

import numpy as np

from typing import *
from abc import ABC, abstractmethod
from logzero import logger
from torch.nn import Module
from sklearn.model_selection import StratifiedKFold, train_test_split, ShuffleSplit
from torch.utils.data import Dataset, DataLoader, Subset

from data.combine_preprocessing import CombineImageDataset
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
            split_size = self.data_config["train_size"]
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
        dataloaders = self.get_dataloaders(subsets=[None]*len(samplers), samplers=samplers)

        return dict(zip([f"split_{x}" for x in range(1, self.data_config["num_splits"])], dataloaders))

    @staticmethod
    @abstractmethod
    def type_check(subject):
        pass

    def get_datasets(self, subsets: Optional[torch.utils.data.Subset], samplers):
        if self.config["model_config"]["model_type"] in ("linear", "image"):
            return CombineImageDataset(self.dataset, self.param_data)
        elif self.config["model_config"]["model_type"] == "sequential":
            # self.dataset.subsets = subsets
            self.dataset.samplers = samplers
            return self.dataset
        else:
            raise TypeError(f"Model type: {self.config['model_config']['model_type']} cannot combine with param_data")

    def get_dataloaders(self,
                        subsets: List[Optional[torch.utils.data.Subset]] = [None],
                        samplers: List[Optional[torch.utils.data.Sampler]] = [None]) -> List[DataLoader]:
        if self.config["model_aug_config"]["model_augmentation"] == "auxiliary":
            return [DataLoader(self.get_datasets(subset, sampler),
                               batch_size=self.config["data_config"]["batch_size"],
                               sampler=sampler) for subset, sampler in zip(subsets, samplers)]
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


class ImageDataSplit(AbstractSplit):
    def __init__(self, config, dataset, param_data, larger_dataset, device):
        super(ImageDataSplit, self).__init__(config, dataset, param_data, device)
        self.config = config
        self.data_config = config["data_config"]
        self.dataset = dataset
        self.param_data = param_data  # Needed for Aux models
        self.larger_dataset = larger_dataset
        self.device = device

    def holdout(self) -> Dict[str, DataLoader]:
        try:
            split_size = self.data_config["train_size"]
            logger.info(f"Splitting dataset into {self.data_config['splits']['size']}")
        except KeyError as e:
            split_size = DEFAULT_SPLIT
            logger.info(e)
            logger.info(f"Could not find split size in config, splitting dataset into {DEFAULT_SPLIT}")

        # train_x, test_x, train_y, test_y = train_test_split(self.dataset, self.dataset.targets, train_size=split_size, random_state=self.config["seed"])
        split_idx = None
        if self.larger_dataset == "aux_data":
            split_idx = list(ShuffleSplit(n_splits=1,
                                          train_size=split_size,
                                          random_state=self.config["seed"]).split(self.dataset,
                                                                                  self.dataset.targets))
        elif self.larger_dataset == "param_data":
            split_idx = list(ShuffleSplit(n_splits=1,
                                          train_size=split_size,
                                          random_state=self.config["seed"]).split(self.param_data.params))

        samplers = [torch.utils.data.SubsetRandomSampler(idx_array) for idx_array in split_idx[0]]
        dataloaders = self.get_dataloaders(subsets=[None]*len(samplers), samplers=samplers)

        return dict(zip([f"split_{x}" for x in range(1, self.data_config["num_splits"])], dataloaders))

    def kfold(self) -> Dict[str, DataLoader]:
        # See SciKitLearn's documentation for implementation details (note that this method enforces same size splits):
        # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
        splits = StratifiedKFold(n_splits=len(self.data_config["num_splits"]), shuffle=self.data_config["shuffle"])
        # split = StratifiedShuffleSplit(n_splits=len(self.data_config["splits"]))
        # The target labels (stratified k fold needs the labels to preserve label distributions in each split)
        # The .split() method from SKLearn returns a generator that generates 2 index arrays (for training and testing)
        samplers = [torch.utils.data.SubsetRandomSampler(idx) for idx in splits.split(self.dataset, self.dataset.targets)]
        dataloaders = self.get_dataloaders(subsets=[None]*len(samplers), samplers=samplers)

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


class QuineDataSplit(AbstractSplit):
    def __init__(self, config, dataset, param_data, device):
        super(QuineDataSplit, self).__init__(config, dataset, param_data, device)
        self.config = config
        self.data_config = config["data_config"]
        self.dataset = dataset
        self.param_data = param_data  # Needed for Aux models
        self.device = device

    def holdout(self):
        # When splitting/partition, we split the indices of the params (which are ints)
        # In combineDataset, the param_data indices will be passed to get_param() in get_item
        try:
            split_size = self.data_config["train_size"]
            logger.info(f"Splitting dataset into {self.data_config['splits']['size']}")
        except KeyError as e:
            split_size = DEFAULT_SPLIT
            logger.error(e)
            logger.info(f"Could not find split size in config, splitting dataset into {DEFAULT_SPLIT}")

        # train_x, test_x, train_y, test_y = train_test_split(self.dataset, self.dataset.targets, train_size=split_size, random_state=self.config["seed"])
        split_idx = list(ShuffleSplit(n_splits=1,
                                      train_size=split_size,
                                      random_state=self.config["seed"]).split(self.param_data.params))
        split_idx = split_idx[0]
        samplers = [torch.utils.data.SubsetRandomSampler(idx_array) for idx_array in split_idx]
        dataloaders = self.get_dataloaders(subsets=[None]*len(samplers), samplers=samplers)

        return dict(zip([f"split_{x}" for x in range(1, self.data_config["num_splits"])], dataloaders))

    def kfold(self) -> Dict[str, DataLoader]:
        # See SciKitLearn's documentation for implementation details (note that this method enforces same size splits):
        # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
        splits = StratifiedKFold(n_splits=len(self.data_config["num_splits"]), shuffle=self.data_config["shuffle"])
        # split = StratifiedShuffleSplit(n_splits=len(self.data_config["splits"]))
        # The target labels (stratified k fold needs the labels to preserve label distributions in each split)
        # The .split() method from SKLearn returns a generator that generates 2 index arrays (for training and testing)
        samplers = [torch.utils.data.SubsetRandomSampler(idx) for idx in splits.split(self.dataset, self.dataset.targets)]
        dataloaders = self.get_dataloaders(subsets=[None]*len(samplers), samplers=samplers)

        return dict(zip([f"split_{x}" for x in range(1, self.data_config["num_splits"])], dataloaders))

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

    def holdout(self):
        # When splitting/partition, we split the indices of the params (which are ints)
        # In combineDataset, the param_data indices will be passed to get_param() in get_item
        try:
            split_size = self.data_config["train_size"]
            logger.info(f"Splitting dataset into {self.data_config['splits']['size']}")
        except KeyError as e:
            split_size = DEFAULT_SPLIT
            logger.error(e)
            logger.info(f"Could not find split size in config, splitting dataset into {DEFAULT_SPLIT}")

        # if self.larger_dataset == "aux_data":
        #     subsets = torch.utils.data.dataset.random_split(self.dataset,
        #                                                     [split_size*int(len(self.dataset)),
        #                                                      len(self.dataset) - split_size*int(len(self.dataset))],
        #                                                     torch.Generator().manual_seed(self.config["seed"]))
        # elif self.larger_dataset == "param_data":
        #     subsets = torch.utils.data.dataset.random_split(self.param_data.params,
        #                                                     [int(split_size*len(self.param_data.params)),
        #                                                      int(len(self.param_data.params)) - int(split_size*len(self.param_data.params))],
        #                                                     torch.Generator().manual_seed(self.config["seed"]))
        #
        # dataloaders = self.get_dataloaders(subsets=subsets, samplers=[None] * len(subsets))

        aux_split_idx = list(ShuffleSplit(n_splits=1,
                                      train_size=split_size,
                                      random_state=self.config["seed"]).split(self.dataset, self.dataset.targets))
        param_split_idx = list(ShuffleSplit(n_splits=1,
                                      train_size=split_size,
                                      random_state=self.config["seed"]).split(self.param_data.params))

        aux_samplers = [torch.utils.data.SubsetRandomSampler(idx_array) for idx_array in aux_split_idx[0]]
        param_samplers = [torch.utils.data.SubsetRandomSampler(idx_array) for idx_array in param_split_idx[0]]
        dataloaders = self.get_dataloaders(subsets=[None] * len(aux_samplers), samplers=aux_samplers)
        dataloaders = dict(zip([f"split_{x}" for x in range(1, self.data_config["num_splits"])], dataloaders))
        dataloaders = {name: [dataloader, DataLoader(self.param_data.params, sampler=param_samplers)] for
                       name, dataloader in dataloaders.items()}

        return dataloaders

    def kfold(self) -> Dict[str, DataLoader]:
        # See SciKitLearn's documentation for implementation details (note that this method enforces same size splits):
        # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
        splits = StratifiedKFold(n_splits=len(self.data_config["num_splits"]), shuffle=self.data_config["shuffle"])
        # split = StratifiedShuffleSplit(n_splits=len(self.data_config["splits"]))
        # The target labels (stratified k fold needs the labels to preserve label distributions in each split)
        # The .split() method from SKLearn returns a generator that generates 2 index arrays (for training and testing)
        subsets = [torch.utils.data.Subset(self.dataset, idx) for idx in splits.split(self.dataset, self.dataset.targets)]
        dataloaders = self.get_dataloaders(subsets=subsets, samplers=[None]*len(subsets))

        return dict(zip([f"split_{x}" for x in range(1, self.data_config["num_splits"])], dataloaders))

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





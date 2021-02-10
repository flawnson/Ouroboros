import torch
import random
import logzero
from logzero import logger

from abc import ABC, abstractmethod
from torch.nn import Module
from torch.utils.data import Dataset
from models.augmented.quine import Quine
from sklearn.model_selection import StratifiedKFold


class AbstractHoldout(ABC):
    def __init__(self, config, dataset, model, device):
        self.data_config = config["data_config"]
        self.dataset = dataset
        self.model = model
        self.device = device

    @abstractmethod
    def tri(self, subject):
        pass

    @abstractmethod
    def holdout(self, subject):
        # See SciKitLearn's documentation for implementation details (note that this method enforces same size splits):
        # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold

        # Should k fold just be done on the indices and not the data....
        split = StratifiedKFold(n_splits=len(self.data_config["splits"]), shuffle=self.data_config["shuffle"])
        # split = StratifiedShuffleSplit(n_splits=len(self.data_config["splits"]))
        y = [subject[y][1] for y, d in enumerate(subject)]
        masks = list(split._iter_test_masks(subject, y))
        return dict(zip(self.data_config["splits"].keys(), masks)) #returns dictionary of masks based on splits in config

    @staticmethod
    @abstractmethod
    def type_check(subject):
        pass

    # @abstractmethod
    def split(self, subject):
        self.type_check(subject)
        if self.data_config["split_type"] == "stratified":
            return self.holdout(subject)
        elif self.data_config["split_type"] == "tri":
            return self.tri(subject)
        else:
            raise NotImplementedError(f"Split-type: {self.data_config['split_type']} not understood")


class DataHoldout(AbstractHoldout):
    def __init__(self, config, dataset, model, device):
        super(DataHoldout, self).__init__(config, dataset, model, device)
        self.data_config = config["data_config"]
        self.dataset = dataset
        self.model = model
        self.device = device

    def tri(self, subject):
        pass

    def holdout(self, subject):
        # See SciKitLearn's documentation for implementation details (note that this method enforces same size splits):
        # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
        split = StratifiedKFold(n_splits=self.data_config["num_splits"], shuffle=self.data_config["shuffle"])
        # split = StratifiedShuffleSplit(n_splits=len(self.data_config["splits"]))

        logger.info("Data holdout")
         #subject is a ConcatDataset

        y = [subject[y][1] for y, d in enumerate(subject)]

        masks = list(split._iter_test_masks(subject, y))

        logger.info(f"Masks length: {len(masks)}")
        logger.info(f"Masks: {masks}")
        index_arr = list(range(0, len(masks)))
        #split_dict = dict(zip(self.data_config["splits"].keys(), masks))
        split_dict = dict(zip(index_arr, masks))
        logger.info(f"Split Dictionary: {split_dict}")

        return split_dict

    @staticmethod
    def type_check(subject):
        assert isinstance(subject, torch.utils.data.Dataset), f"Subject: {type(subject)} is not a splittable type"

    # def split(self, subject):
    #     self.type_check(subject)
    #     if self.data_config["split_type"] == "stratified":
    #         return self.holdout(subject)
    #     elif self.data_config["split_type"] == "tri":
    #         return self.tri(subject)
    #     else:
    #         raise NotImplementedError(f"Split-type: {self.data_config['split_type']} not understood")


class ModelHoldout(AbstractHoldout):
    def __init__(self, config, dataset, model, device):
        super(ModelHoldout, self).__init__(config, dataset, model, device)
        self.data_config = config["data_config"]
        self.dataset = dataset
        self.model = model
        self.device = device

    def tri(self, subject):
        pass

    def holdout(self, subject):
        # See SciKitLearn's documentation for implementation details (note that this method enforces same size splits):
        # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
        split = StratifiedKFold(n_splits=self.data_config["num_splits"], shuffle=self.data_config["shuffle"])
        # split = StratifiedShuffleSplit(n_splits=len(self.data_config["splits"]))
        logger.info("Param holdout")
        masks = list(split._iter_test_masks(subject.params, torch.zeros_like(torch.tensor(subject.params))))


        logger.info(f"Masks length: {len(masks)}")
        logger.info(f"One mask: {len(masks[0])}")
        logger.info(f"Masks: {masks}")
        index_arr = list(range(0, len(masks)))
        #split_dict = dict(zip(self.data_config["splits"].keys(), masks))
        split_dict = dict(zip(index_arr, masks))
        logger.info(f"Split Dictionary: {split_dict}")

        return split_dict

    @staticmethod
    def type_check(subject):
        pass
        # assert isinstance(subject, Quine) & isinstance(subject, Module), f"Subject: {type(subject)} is not a splittable type"

    # def split(self, subject):
    #     self.type_check(subject)
    #     if self.data_config["split_type"] == "stratified":
    #         return self.holdout(subject)
    #     elif self.data_config["split_type"] == "tri":
    #         return self.tri(subject)
    #     else:
    #         raise NotImplementedError(f"Split-type: {self.data_config['split_type']} not understood")

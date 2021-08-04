import torch

import numpy as np

from typing import *
from logzero import logger
from abc import ABC, abstractmethod
from sklearn.metrics import f1_score, precision_score, recall_score, jaccard_score, confusion_matrix, roc_auc_score


class AbstractScores:
    def __init__(self, config: Dict, predictions, targets, total, correct, device: torch.device):
        self.score_config = config["score_config"]
        self.predictions = predictions
        self.targets = targets
        self.total = total
        self.correct = correct
        self.device = device

    @staticmethod
    def calculate_relative_difference(x, y):
        return abs(x - y) / max(abs(x), abs(y))

    def relative_difference(self) -> float:
        rel_diff = self.calculate_relative_difference(self.predictions["param"].item(), self.targets["param"].item())
        return rel_diff

    def accuracy(self):
        """ Loops over each set correct number to calculate accuracy"""
        try:
            return [self.correct[x] / self.total[x] for x in range(len(self.correct))]
        except ZeroDivisionError as e:
            logger.exception(e)
            logger.info("Could not calculate accuracy, returning 0")
            return 0

    def precision(self):
        return precision_score(self.targets, np.argmax(self.predictions, axis=1), **self.score_config["precision"])

    def recall(self):
        return recall_score(self.targets, np.argmax(self.predictions, axis=1), **self.score_config["recall"])

    def auroc(self):
        return roc_auc_score(self.targets, self.predictions, **self.score_config["auroc"])

    def f1_score(self):
        return f1_score(self.targets, np.argmax(self.predictions, axis=1), **self.score_config["f1_score"])

    def get_scores(self) -> Dict[str, List[float]]:
        scoreset = {"acc": self.accuracy,
                    "auroc": self.auroc,
                    "precision": self.precision,
                    "recall": self.recall,
                    "f1_score": self.f1_score}

        return {score_type: scoreset[score_type]() for score_type, score_args in self.score_config.items()
                if score_args is not None}


class GraphScores:
    # XXX: PLACEHOLDER; CODE IS NOT FUNCTIONAL
    def __init__(self, config: Dict, dataset, epoch_data, device: torch.device):
        self.score_config = config["score_config"]
        self.dataset = dataset
        self.epoch_data = epoch_data
        self.device = device

    def accuracy(self, params) -> float:
        return self.prediction.eq(self.dataset.ndata["y"][self.agg_mask]).sum().item() / self.agg_mask.sum().item()

    def f1_score(self, params):
        return f1_score(y_true=self.dataset.ndata["y"][self.agg_mask].to('cpu'),
                        y_pred=self.prediction.to('cpu'),
                        average=params[0])

    def auroc(self, params) -> float:
        return roc_auc_score(params=params,
                           dataset=self.dataset,
                           agg_mask=self.agg_mask,
                           split_mask=self.split_mask,
                           logits=self.logits,
                           s_logits=self.s_logits)

    def confusion_mat(self, params) -> np.ndarray:
        return confusion_matrix(y_true=self.dataset.ndata["y"][self.agg_mask].to('cpu'),
                                y_pred=self.prediction.to('cpu'),
                                labels=None,
                                sample_weight=None,
                                normalize=None)

    def precision(self, params) -> float:
        return precision_score(y_true=self.dataset.ndata["y"][self.agg_mask].to('cpu'),
                               y_pred=self.prediction.to('cpu'),
                               labels=None,
                               pos_label=1,
                               average=params[0],
                               sample_weight=None,
                               zero_division=params[1])

    def recall(self, params) -> float:
        return recall_score(y_true=self.dataset.ndata["y"][self.agg_mask].to('cpu'),
                            y_pred=self.prediction.to('cpu'),
                            labels=None,
                            pos_label=1,
                            average=params[0],
                            sample_weight=None,
                            zero_division=params[1])

    def jaccard(self, params) -> float:
        return jaccard_score(y_true=self.dataset.ndata["y"][self.agg_mask].to('cpu'),
                             y_pred=self.prediction.to('cpu'),
                             labels=None,
                             pos_label=1,
                             average=params[0],
                             sample_weight=None)

    def get_scores(self) -> Dict[str, object]:
        scoreset = {"acc": self.accuracy(self.score_config["acc"]),
                    "auc": self.auroc(self.score_config["auc"]),
                    "f1": self.f1_score(self.score_config["f1"]),
                    "con": self.confusion_mat(self.score_config["con"]),
                    "prec": self.precision(self.score_config["prec"]),
                    "rec": self.recall(self.score_config["rec"]),
                    "jac": self.jaccard(self.score_config["jac"])
                    }

        return {score_type: scoreset[score_type] for score_type in self.score_config.keys()}


def scores(config: Dict, dataset, accumulator, device: torch.device) -> Dict:
    """
    Function to call the correct score class

    Args:
        config: Configuration dict
        dataset: output from the model
        correct: labels from the dataset
        device: torch.device

    Returns:
        Score object corresponding to the type of data (which then returns a dictionary of scores)

    """
    predictions = accumulator["predictions"]
    targets = accumulator["targets"]
    total = accumulator["total"]
    correct = accumulator["correct"]
    if config["data_config"]["dataset"].casefold() == "mnist" or "cifar" or "cifar10":
        return AbstractScores(config, predictions, targets, total, correct, device).get_scores()
    elif config["data_config"]["dataset"].casefold() == "cora":
        return GraphScores(config, dataset, predictions, targets, correct, device).get_scores()
    else:
        raise NotImplementedError(f"{config['data_config']['dataset']} is not a data type")



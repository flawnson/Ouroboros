import torch

import numpy as np

from typing import *
from logzero import logger
from abc import ABC, abstractmethod
from sklearn.metrics import f1_score, precision_score, recall_score, jaccard_score, confusion_matrix


class MLPScores:
    def __init__(self, config: Dict, dataset, epoch_data, device: torch.device):
        self.score_config = config["score_config"]
        self.dataset = dataset
        self.epoch_data = epoch_data
        self.device = device

    def relative_error(self):
        # x = predicted param
        # y = actual param
        # return abs(x - y) / max(abs(x), abs(y))
        pass

    def accuracy(self):
        """ Loops over each set correct number to calculate accuracy"""
        # pred = self.predictions.argmax(keepdim=True)  # get the index of the max log-probability
        # self.correct += pred.eq(self.targets.view_as(pred)).sum().item()
        return [self.epoch_data[x] / len(self.dataset[list(self.dataset)[x]]) for x in range(len(self.epoch_data))]

    def get_scores(self) -> Dict[str, List[float]]:
        scoreset = {"acc": self.accuracy()}

        return {score_type: scoreset[score_type] for score_type in self.score_config.keys()}


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
        return auroc_score(params=params,
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


def scores(config: Dict, dataset, correct, device: torch.device):
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
    if config["model_config"]["model_type"] == "linear":
        return MLPScores(config, dataset, correct, device).get_scores()
    elif config["model_config"]["model_type"] == "graph":
        return GraphScores(config, dataset, correct, device).get_scores()
    elif config["model_config"]["model_type"] == "image":
        pass
    elif config["model_config"]["model_type"] == "language":
        pass
    else:
        raise NotImplementedError(f"{config['model_config']['model_type']} is not a model type")



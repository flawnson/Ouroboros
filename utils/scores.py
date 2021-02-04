import torch

import numpy as np

from typing import *
from logzero import logger
from sklearn.metrics import f1_score, precision_score, recall_score, jaccard_score, confusion_matrix


class Scores(object):
    def __init__(self, config, device):
        self.config = config
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

    def score(self) -> Dict[str, object]:
        scoreset = {"acc": self.accuracy(self.score_config["acc"]),
                    "auc": self.auroc(self.score_config["auc"]),
                    "f1": self.f1_score(self.score_config["f1"]),
                    "con": self.confusion_mat(self.score_config["con"]),
                    "prec": self.precision(self.score_config["prec"]),
                    "rec": self.recall(self.score_config["rec"]),
                    "jac": self.jaccard(self.score_config["jac"])
                    }

        return {score_type: scoreset[score_type] for score_type in self.score_config.keys()}
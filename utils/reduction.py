import torch
import numpy as np

from typing import *
from copy import deepcopy
from logzero import logger
from sklearn import random_projection
from sklearn.decomposition import PCA


class Reduction(object):
    def __init__(self, config, data):
        self.config = config
        self.data = data

    def pca(self):
        pass

    def random(self):
        X = np.random.rand(1, self.data)
        transformer = random_projection.GaussianRandomProjection(n_components=self.model_aug_config["n_hidden"])
        transformer.fit(X)
        rand_proj_matrix = transformer.components_

        return rand_proj_matrix

    def reduce(self):
        if self.config["reduction_method"] is None:
            logger.info("No dimension reduction method provided... Continuing without reducing")
        elif self.config["reduction_method"].casefold() == "random":
            return self.random()
        elif self.config["reduction_method"].casefold() == "pca":
            return self.pca()
        else:
            logger.info(f"Dimension reduction method {self.config['reduce_dimension']} not undersood... Continuing without reducing")
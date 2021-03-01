import torch
import numpy as np

from typing import *
from copy import deepcopy
from logzero import logger
from sklearn import random_projection
from sklearn.decomposition import PCA


class Reduction(object):
    def __init__(self, model_aug_config: Dict, data: torch.tensor):
        self.model_aug_config = model_aug_config
        self.data = data

    def pca(self):
        pass

    def random(self) -> np.array:
        """
        Random project method from Scikitlearn (as used in the Quine paper)

        Returns:
            A numpy array (matrix) of the projected values
        """
        X = np.random.rand(1, self.data)
        transformer = random_projection.GaussianRandomProjection(n_components=self.model_aug_config["n_hidden"])
        transformer.fit(X)
        rand_proj_matrix = transformer.components_

        return rand_proj_matrix

    def reduce(self):
        """
        Method to call the chosen dimensionality reduction algorithm from class methods

        Returns:
            The output from whichever dimensionality reduction algorithm was chosen
        """
        if self.model_aug_config["reduction_method"] is None:
            logger.info("No dimension reduction method provided... Continuing without reducing")
        elif self.model_aug_config["reduction_method"].casefold() == "random":
            return self.random()
        elif self.model_aug_config["reduction_method"].casefold() == "pca":
            return self.pca()
        else:
            logger.info(f"Dimension reduction method {self.model_aug_config['reduce_dimension']} not undersood..."
                        f"Continuing without reducing")




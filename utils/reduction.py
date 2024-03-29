import torch
import numpy as np

from typing import *
from logzero import logger
from sklearn import random_projection


class Reduction(object):
    def __init__(self, model_aug_config: Dict, data_size: int):
        self.model_aug_config = model_aug_config
        self.data_size = data_size

    def random(self) -> np.array:
        """
        Random project method from Scikitlearn (as used in the Quine paper)

        Returns:
            A numpy array (matrix) of the projected values
        """
        X = np.random.rand(1, self.data_size)
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
        else:
            logger.info(f"Dimension reduction method {self.model_aug_config['reduce_dimension']} not undersood..."
                        f"Continuing without reducing")




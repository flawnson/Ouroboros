import torch
import numpy as np

from typing import *


def get_example_size(dataset: torch) -> int:
    # A function to return the size of an example for any dataset and datatype
    # Currently hard coded to supply the model with the flattened size of an MNIST image example
    return 784


import torch
import time

from logzero import logger
from typing import *
from functools import wraps


def get_example_size(dataset: torch) -> int:
    # A function to return the size of an example for any dataset and datatype
    # Currently hard coded to supply the model with the flattened size of an MNIST image example
    return 784


def timed(func: Callable):
    """This decorator prints the execution time for the decorated function."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logger(__name__).info("{} ran in {}s".format(func.__name__, round(end - start, 2)))
        return result

    return wrapper
import os
import torch
import time

from functools import wraps
from logzero import logger
from typing import *


def get_example_size(dataset: torch) -> int:
    """
    This function returns the size of a single data example, agnostic of types

    Args:
        dataset: The entire dataset from which to identify the example size

    Return:
        The integer size of a single example
    """
    # A function to return the size of an example for any dataset and datatype
    # Might be a bit convoluted right now
    # if isinstance(dataset, ConcatDataset):
    #     try:
    #         dataset_idx = np.random.randint(0, len(dataset.datasets))
    #         subset_indices = [np.random.randint(len(dataset.datasets[0]))]  # select your indices here as a list
    #         subset = torch.utils.data.Subset(dataset.datasets[dataset_idx], subset_indices)
    #         example = next(iter(torch.utils.data.DataLoader(subset, batch_size=1, num_workers=0, shuffle=False)))
    #         example_size = example[0].reshape(-1).shape[0]
    #         return example_size
    #     except:
    #         logger.info(f"Could not determine size of data example for {dataset} dataset")
    return 784


def timed(func: Callable):
    """
    This decorator prints the execution time for the decorated function.

    Args:
        func: The function to call (name used for logging)

    Returns:
        The wrapped function executed
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logger.info(f"{func.__name__} ran in {round(end - start, 2)}s \n")
        return result

    return wrapper


def make_clean_directories(beta, root_folder, iteration):
    """
    This function receives a file name, a directory name, and iteration number to check and wipe clean

    Args:
        beta: Filler info for naming
        root_folder: The root directory to check for
        iteration: The iteration number for logging

    Returns:
        The name of the data directory that was cleaned and prepared

    """
    data_dir = root_folder + '/results_' + str(beta) + '_' + str(iteration)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    else:
        if len(os.listdir(data_dir)) > 0:
            os.system("rm -r %s/*" % data_dir)

    return data_dir


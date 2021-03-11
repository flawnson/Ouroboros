import os
import torch
import time
import numpy as np

from torch.utils.data import ConcatDataset
from functools import wraps
from logzero import logger
from typing import *


def get_example_size(dataset: torch) -> int:
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
    """This decorator prints the execution time for the decorated function."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logger(__name__).info(f"{func.__name__} ran in {round(end - start, 2)}s \n")
        return result

    return wrapper


def make_clean_directories(beta, root_folder, iteration):
    data_dir = root_folder + '/results_' + str(beta) + '_' + str(iteration)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    else:
        if len(os.listdir(data_dir)) > 0:
            os.system("rm -r %s/*" % (data_dir))

    return data_dir


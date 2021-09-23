import os
import json
import torch
import pytest
import logzero
import logging
import argparse
import numpy as np

from torch.utils.data import DataLoader
from typing import *
from logzero import logger

from utils.scores import scores
from utils.utilities import timed


@pytest.fixture
def config():
    file = os.path.join("..", "configs", "linear_aux_demo_small.json")
    config: Dict = json.load(open(file))
    device = torch.device("cuda" if config["device"] == "cuda" and torch.cuda.is_available() else "cpu")
    logzero.loglevel(eval(config["log_level"]))
    logger.info(f"Successfully retrieved config json. Running {config['run_name']} on {device}.")

    return config


@pytest.fixture  # Can pass params directly as a list but would rather use parameterize
def device(config):
    device = torch.device("cuda" if config["device"] == "cuda" and torch.cuda.is_available() else "cpu")
    logger.info(f"Running {config['run_name']} on {device}.")

    return device


@timed
def test_scoring(config, device):
    NUM_SAMPLES = 500
    NUM_CLASSES = 10

    epoch_data = {"predictions": [],
                  "targets": [],
                  "total": [0, 0],
                  "correct": [0, 0]}

    for i in range(NUM_SAMPLES):
        epoch_data["predictions"].append(np.random.dirichlet(np.ones(NUM_CLASSES), size=NUM_SAMPLES).tolist())
    for i in range(NUM_SAMPLES):
        epoch_data["targets"].append(np.random.randint(0, NUM_CLASSES, NUM_SAMPLES).tolist())
    epoch_data["total"].append(NUM_SAMPLES)
    epoch_data["correct"] += sum(np.equal(epoch_data["targets"], np.argmax(epoch_data["predictions"], axis=1)))

    score_dict = scores(config, None, epoch_data, device)  # No dataset needed for non-graph models
    logger.info(score_dict)

    assert isinstance(score_dict, dict)
    assert isinstance(score_dict["auroc"], float)




import os
import json
import torch
import pytest
import logzero
import logging
import argparse
import numpy as np

from typing import *
from logzero import logger
from utils.utilities import timed
from utils.logging import TFTBLogger, PTTBLogger

### Configuring ###


@pytest.fixture
def config():
    file = r"C:\Users\flawn\Documents\Projects\Project_Introversion\Ouroboros\configs\local_linear_aux_demo.json"
    config: Dict = json.load(open(file))
    device = torch.device("cuda" if config["device"] == "cuda" and torch.cuda.is_available() else "cpu")
    logzero.loglevel(eval(config["logging"]))
    logger.info(f"Successfully retrieved config json. Running {config['run_name']} on {device}.")

    return config


def test_tb_logging(config):
    tb_logger = TFTBLogger(config)
    for idx in range(0, 100):
        tb_logger.scalar_summary('Some metric', np.random.randint(0, 1000), idx)
        tb_logger.scalar_summary('Some other metric', np.random.randint(0, 1000), idx)

    assert os.path.exists(config["log_dir"])
    assert len(os.listdir(config["log_dir"])) > 0


def test_pt_logging(config):
    pt_logger = PTTBLogger(config)
    for idx in range(0, 100):
        pt_logger.scalar_summary('Some metric', np.random.randint(0, 1000), idx)
        pt_logger.scalar_summary('Some other metric', np.random.randint(0, 1000), idx)

    assert os.path.exists(config["log_dir"])
    assert len(os.listdir(config["log_dir"])) > 0


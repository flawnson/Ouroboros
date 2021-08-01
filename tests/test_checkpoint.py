import os
import json
import torch
import pytest
import logzero
import logging
import pathlib
import numpy as np

from typing import *
from logzero import logger
from optim.algos import OptimizerObj
from models.standard.linear_model import LinearModel
from utils.checkpoint import PTCheckpoint

### Configuring ###


DEFAULT_CONFIG = os.path.join("configs", "linear_aux_demo_small.json")


@pytest.fixture
def config():
    config: Dict = json.load(open(DEFAULT_CONFIG))
    logzero.loglevel(eval(config["log_level"]))
    logger.info(f"Successfully retrieved config json. Running {config['run_name']}.")

    return config

@pytest.fixture
def device():
    config: Dict = json.load(open(DEFAULT_CONFIG))
    device = torch.device("cuda" if config["device"] == "cuda" and torch.cuda.is_available() else "cpu")
    logger.info(f"Running {config['run_name']} on {device}.")


@timed
@pytest.mark.parametrize("config,device", [config, device])
def test_checkpoint(config, device):
    model = LinearModel(config, device).to(device)
    optimizer = OptimizerObj(config, model).optim_obj
    checkpoint = PTCheckpoint(config)
    for epoch in range(0, 1000):
        checkpoint.checkpoint(config=config,
                              epoch=epoch,
                              model=model,
                              loss=np.random.randint(0, 1),
                              optimizer=optimizer)

    assert os.path.exists(pathlib.Path(config["run_config"]["checkpoint_dir"]))
    assert len(os.listdir(config["run_config"]["checkpoint_dir"])) > 0
    assert sum([True if file.endswith(".pt") or file.endswith(".py") else False
                for file in os.listdir(config["run_config"]["checkpoint_dir"])])

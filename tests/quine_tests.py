import json
import torch
import pytest
import logzero
import argparse
import numpy as np

from typing import *
from logzero import logger
from utils.scores import scores
from utils.holdout import AbstractSplit
from utils.utilities import timed
from models.augmented.quine import Quine


if __name__ == "__main__":
    ### Configuring ###
    parser = argparse.ArgumentParser(description="Config file parser")
    parser.add_argument("-c", "--config", help="json config file", type=str)
    parser.add_argument("-s", "--scheme", help="json scheme file", type=str)
    args = parser.parse_args()

    config: Dict = json.load(open(args.config))
    device = torch.device("cuda" if config["device"] == "cuda" and torch.cuda.is_available() else "cpu")
    logzero.loglevel(eval(config["logging"]))
    logger.info(f"Successfully retrieved config json. Running {config['run_name']} on {device}.")

    # In json config file, provide the name of the testing function to execute and pass the compulsory kwargs arguments
    eval(config["test_type"])(config["test_params"])

    @pytest.fixture
    def output_logits():
        return torch.randn(20, 20)


    @timed
    @pytest.mark.parametrize("test_config", config)
    def test_holdout(test_config):
        dataset = []
        output = AbstractSplit(test_config, dataset, device)
        assert output == "test"


    @timed
    @pytest.mark.parametrize("test_config", config)
    def test_regenerate(test_config):
        model = Quine(test_config, model, device)
        model.regenerate()


    @timed
    @pytest.mark.parametrize("test_config", config)
    def test_scores(test_config):
        assert isinstance(scores(test_config, dataset, epoch_data["correct"], device), dict)

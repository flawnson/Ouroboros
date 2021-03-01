import json
import torch
import logzero
import argparse
import numpy as np

from typing import *
from logzero import logger
from utils.holdout import AbstractSplit


def test_holdout(**kwargs):
    dataset = []
    output = AbstractSplit(config, dataset, device)
    assert output == "test"


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



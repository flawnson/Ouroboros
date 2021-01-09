import argparse
import torch
import json
import os

from models.graph_model import GNNModel
from data.graph_preprocessing import GenericDataset

### Configuring ###
path = os.path.join('data', 'biogrid')
parser = argparse.ArgumentParser(description="Config file parser")
parser.add_argument("-c", "--config", help="json config file", type=str)
parser.add_argument("-s", "--scheme", help="json scheme file", type=str)
args = parser.parse_args()

config: dict = json.load(open(args.config))
device = torch.device("cuda" if config["cuda"] and torch.cuda.is_available() else "cpu")

### Data preprocessing ###
dataset: GenericDataset = None
    if config["dataset"] == "primary_labelset":
        dataset = PrimaryLabelset(config).dataset.to(device)
    else:
        raise NotImplementedError(f"{config['dataset']} is not a dataset")  # Add to logger when implemented


### Model preparation ###
GNNModel(config, data, device)

### Pipeline ###

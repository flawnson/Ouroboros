import torch
import os.path as osp

from typing import *
from logzero import logging


def checkpoint(config: Dict, epoch: int, model: torch.nn.Module, loss: float, optimizer: torch.optim):
    """
    This is an example of Google style.
    Checkpoint intervals must be a divisor of the total number of epochs

    Args:
        config: Configuration dictionary
        epoch: The current epoch number
        model: The Pytorch model to be checkpoint
        loss: The loss of the current epoch
        optimizer: The Pytorch optimizer object to be checkpoint

    Raises:
        Pytorch's error returned by a failed attempt at saving a model
    """
    if config["run_config"]["checkpoint_intervals"] is None:
        pass
    elif epoch % config["run_config"]["checkpoint_intervals"] == 0:
        try:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, osp.join(config["run_config"].get(['checkpoint_dir'], osp.join("saves", "checkpoints")),
                        f"{config['run_name']}_{config['model_config']['model_type']}_{config['run_type']}_{config['data_config']['dataset']}_epoch_{str(epoch)}.pt"))
            logging.info(f"Successfully saved model checkpoint at epoch{epoch}. File saved at {config['run_config']['checkpoint_dir']}")
        except:
            logging.info(f"Failed to save checkpoint... Continuing run")
    else:
        pass


def load(config, model: torch.nn.Module = None):
    if model is not None:
        model = model.load_state_dict(config["model_dir"])
    elif model is None:
        model = torch.load(config["model_dir"])
    else:
        raise NotImplementedError(f"Provided model {model} is not loadable, or {config['model_dir']} does not exist")

    return model



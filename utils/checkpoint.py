import os
import sys
import torch
import shutil
import pathlib
import os.path as osp

from typing import *
from logzero import logger


class PTCheckpoint(object):

    def __init__(self, config):
        self.config = config
        self.save_dir = config["run_config"]["checkpoint_dir"]
        pathlib.Path(self.save_dir).absolute().mkdir(parents=True, exist_ok=True)
        if config["clean_log_dir"] and len(os.listdir(self.save_dir)) > 0:  # Clears dir of old checkpoints
            for f in os.listdir(self.save_dir):
                try:
                    os.remove(os.path.join(self.save_dir, f))
                    logger.info("Successfully cleaned checkpoint directory")
                except Exception as e:
                    logger.exception(e)
                    logger.info("Continuing run without deleting some files from checkpoint dir")

    def checkpoint(self, config: Dict, epoch: int, model: torch.nn.Module, loss: float, optimizer: torch.optim):
        """
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
        try:
            if not config["run_config"]["checkpoint_intervals"]:
                pass
            elif epoch % config["run_config"]["checkpoint_intervals"] == 0:
                save_name = f"{config['run_name']}_{config['model_config']['model_type']}_{config['run_type']}_{config['data_config']['dataset']}_epoch_{str(epoch)}.pt"

                # Save the model file
                model_path = os.path.abspath(sys.modules[model.__module__].__file__)
                model_file_name = model_path.replace("/", "\\").split("\\")[-1]
                shutil.copy(model_path, os.path.join(self.save_dir, model_file_name))

                # Save the model state
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                }, os.path.join(self.save_dir, save_name))
                logger.info(f"Successfully saved model checkpoint at epoch {epoch}. File saved at {osp.join(config['log_dir'], 'checkpoints')}")
            else:
                pass
        except Exception as e:
            logger.exception(e)
            logger.info("Checkpointing failed... Continuing run")


def load(config, model: torch.nn.Module = None):
    if model is not None:
        model = model.load_state_dict(config["model_dir"])
    elif model is None:
        model = torch.load(config["model_dir"])
    else:
        raise NotImplementedError(f"Provided model {model} is not loadable, or {config['model_dir']} does not exist")

    return model



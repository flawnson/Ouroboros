import torch
import os.path as osp

from typing import *
from logzero import logging


def checkpoint(config: Dict, epoch: int, model: torch.nn.Module, loss: float, optimizer: torch.optim):
    # Checkpoint intervals must be a divisor of the total number of epochs
    if config["checkpoint_intervals"] is None:
        pass
    elif epoch % config["checkpoint_intervals"] == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, osp.join(config['checkpoint_dir'], f"{config['run_name']}_{config['run_type']}_epoch:{str(epoch)}.pt"))
        logging.info(f"Successfully saved model checkpoint at epoch{epoch}. File saved at {config['checkpoint_dir']}")
    else:
        pass


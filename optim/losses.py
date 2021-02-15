"File for all implemented loss functions"

import torch
import torch.functional as F

from typing import *
from logzero import logger

def sr_loss(config, prediction, target):
    loss_sr = (torch.linalg.norm(predictions["param"] - targets["param"], ord=2)) ** 2

    return loss_sr


def task_loss(config, prediction, target):
    loss_task[0] = F.nll_loss(pred_aux.unsqueeze(dim=0), data[1])


def quine_combined_loss(config, prediction, target):
    loss_combined[0] = loss_sr[0] + lambda_val * loss_task[0]




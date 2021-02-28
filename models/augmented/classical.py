import torch

from typing import *
from logzero import logger


class Classical(object):
    def __init__(self, config: Dict, model: torch.nn.Module):
        self.config = config
        self.model = model





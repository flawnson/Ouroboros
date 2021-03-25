"""Code named file for our augmented self-referential models"""
import torch

from typing import *
from logzero import logger
from abc import ABC, abstractmethod


class Ouroboros(ABC):
    """Abstract class for any novel ideas we implement"""
    @abstractmethod
    def __init__(self):
        super(Ouroboros, self).__init__()
        pass

    def get_param(self, idx):
        pass


class Jung(Ouroboros, torch.nn.Module):
    """Code-named class for MetaNetworks; Adversarial HyperNetworks"""
    def __init__(self, config, model):
        super(Jung, self).__init__()
        self.config = config
        self.model = model
        self.aux_model = self.get_aux()

    def get_aux(self):

        return model


class Kekule(Ouroboros):
    def __init__(self, config, model):
        super(Kekule, self).__init__()
        self.config = config
        self.model = model


class Godel:
    def __init__(self, config, model):
        super(Godel, self).__init__()
        self.config = config
        self.model = model


class Escher:
    def __init__(self, config, model):
        super(Escher, self).__init__()
        self.config = config
        self.model = model


class Bach:
    def __init__(self, config, model):
        super(Bach, self).__init__()
        self.config = config
        self.model = model

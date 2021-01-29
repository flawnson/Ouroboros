import torch
import typing


class Classical(object):
    def __init__(self, config, model):
        self.config = config
        self.model = model
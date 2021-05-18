import torch
import numpy as np

from typing import *
from logzero import logger


class TransformerModel(torch.nn.Module):
    def __init__(self, config, device):
        super(TransformerModel, self).__init__()
        self.config = config
        self.device = device

    def multi_head_attn(self, v, k, q):
        v = torch.nn.Linear()(v)
        k = torch.nn.Linear(k)
        q = torch.nn.Linear(q)

        qk = torch.matmul(q, k)
        scaled_qk = qk / torch.sqrt(torch.tensor(k.size()))
        masked_qk = scaled_qk
        agg_qk = torch.softmax(masked_qk)
        final = torch.matmul(agg_qk, v)

        return final

    def positional_encoding(self):
        pass


    def encoder(self, v, k, q):
        embedding = self.multi_head_attn(v, k, q)

        torch.layer_norm()

    def decoder(self):
        pass

    def forward(self, x):
        return x

import math
import torch
import numpy as np

from typing import *
from logzero import logger


class TransformerModel(torch.nn.Module):
    def __init__(self, config, device):
        super(TransformerModel, self).__init__()
        self.config = config
        self.device = device
        self.positional_encoding = PositionalEncoding(d_model, config["dropout"], config["max_len"])

    def multi_head_attn(self, v, k, q):
        v = torch.nn.Linear()(v)
        k = torch.nn.Linear()(k)
        q = torch.nn.Linear()(q)

        qk = torch.matmul(q, k)
        scaled_qk = qk / torch.sqrt(torch.tensor(k.size()))
        masked_qk = scaled_qk
        agg_qk = torch.softmax(masked_qk)
        out = torch.matmul(agg_qk, v)

        final = torch.cat(out)

        return final

    def encoder(self, v, k, q):
        embedding = self.multi_head_attn(v, k, q)

        torch.layer_norm()

    def decoder(self):
        pass

    def forward(self, x, mask):
        self.encoder()
        self.decoder()

        x = torch.nn.Linear()(x)
        torch.nn.Softmax()(x)
        return x


class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
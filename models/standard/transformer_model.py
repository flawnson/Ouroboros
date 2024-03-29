import math
import torch
import numpy as np
import torch.nn.functional as F

from typing import *
from logzero import logger
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer


class ManualTransformerModel(torch.nn.Module):
    def __init__(self, config, device):
        super(ManualTransformerModel, self).__init__()
        self.config = config
        self.device = device
        self.positional_encoding = PositionalEncoding(d_model, config["dropout"], config["max_len"])

    def multi_head_attn(self, v, k, q):
        v = torch.nn.Linear()(v)
        k = torch.nn.Linear()(k)
        q = torch.nn.Linear()(q)

        # Scaled Dot Product Attn
        qk = torch.matmul(q, k)
        scaled_qk = qk / torch.sqrt(torch.tensor(k.size()))
        masked_qk = scaled_qk
        agg_qk = torch.softmax(masked_qk, dim=-1)  # Or dim=2, softmax needs to be applied across the time dim
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


class TransformerModel(torch.nn.Module):

    def __init__(self, config, datasets, device):
        super(TransformerModel, self).__init__()
        model_config = config["model_config"]
        num_tokens = len(config["vocab"].get_stoi())  # Dictionary mapping tokens to indices
        input_size = model_config["input_size"]
        num_heads = model_config["num_heads"]
        hidden_size = model_config["hidden_size"]
        num_layers = model_config["num_layers"]
        dropout = model_config["dropout"]
        self.pos_encoder = PositionalEncoding(input_size, dropout)
        encoder_layers = TransformerEncoderLayer(input_size, num_heads, hidden_size, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.encoder = torch.nn.Embedding(num_tokens, input_size)
        if model_config["decoder_type"].casefold() == "linear":  # PyTorch's example in docs uses linear decoder
            self.transformer_decoder = torch.nn.Linear(input_size, num_tokens)
        elif model_config["decoder_type"].casefold() == "decoder":
            decoder_layers = TransformerDecoderLayer(input_size, num_heads, hidden_size, dropout, batch_first=True)
            self.transformer_decoder = TransformerDecoder(input_size, num_tokens)
        self.ninp = input_size
        self.init_weights()
        # For auxiliary model (technically positional encoder has no trainable parameters only PyTorch buffers)
        self.param_list = list(self.pos_encoder.parameters()) + list(self.transformer_encoder.parameters()) + list(self.transformer_decoder.parameters())

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.transformer_decoder.bias.data.zero_()
        self.transformer_decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask):
        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.transformer_decoder(output)
        return output


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
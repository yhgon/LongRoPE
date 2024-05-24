from transformers import LlamaForCausalLM
import transformers
import torch
import math
import os
import warnings
from typing import Optional, Tuple, Union, List, Dict

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss


import torch

def long_rope(x, pos, dim, base=500000.0, lambda_factors=None, n_prime=0):
    half_dim = dim // 2
    freq_seq = torch.arange(half_dim, dtype=torch.float32)
    inv_freq = 1.0 / (base ** (freq_seq / half_dim))
    sinusoid_inp = torch.einsum("i,j->ij", pos, inv_freq)

    if lambda_factors is None:
        lambda_factors = torch.ones_like(freq_seq)

    emb = torch.zeros_like(x)
    if n_prime > 0:
        emb[:n_prime] = torch.cat((sinusoid_inp[:n_prime].sin(), sinusoid_inp[:n_prime].cos()), dim=-1)

    for i in range(n_prime, x.size(0)):
        factor = lambda_factors if i >= n_prime else 1.0
        scaled_sinusoid_inp = sinusoid_inp[i] * factor
        emb[i] = torch.cat((scaled_sinusoid_inp.sin(), scaled_sinusoid_inp.cos()), dim=-1)

    return x * emb

from transformers import LlamaForCausalLM
import torch
import torch.nn as nn

class LongRoPEWrapper(nn.Module):
    def __init__(self, model, lambda_factors=None, n_prime=0):
        super(LongRoPEWrapper, self).__init__()
        self.model = model
        self.dim = model.config.hidden_size
        self.base = model.config.rope_theta  # Use the model's rope_theta
        self.lambda_factors = lambda_factors if lambda_factors is not None else torch.linspace(1.0, 1.5, self.dim // 2)
        self.n_prime = n_prime

    def forward(self, input_ids, attention_mask=None, position_ids=None, use_long_rope=False, **kwargs):
        if use_long_rope and position_ids is not None:
            position_embeds = long_rope(input_ids, position_ids, self.dim, base=self.base, lambda_factors=self.lambda_factors, n_prime=self.n_prime)
            kwargs['position_ids'] = position_embeds

        return self.model(input_ids, attention_mask=attention_mask, **kwargs)
    
    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)


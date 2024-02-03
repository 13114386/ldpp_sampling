from __future__ import unicode_literals, print_function, division
'''
    Based on https://github.com/KaiQiangSong/struct_infused_summ
    Reference to embedding_layer and dropout_layer
'''
import torch
import torch.nn as nn
import numpy as np
from utility.utility import *

srng = np.random.RandomState(seed = 19940609)

def dropout_calc(x, rate = 0.0):
    mask = srng.binomial(n = 1,
                        p = (1- rate),
                        size = list(x.shape))
    mask = torch.tensor(mask, dtype=torch.float32, device=x.device)
    return x * mask / (1. - rate)


class EmbeddingLayer(nn.Module):
    def __init__(self, opts, vocab, batch_first):
        super().__init__()
        self.opts = opts
        self.vocab = vocab
        self.batch_first = batch_first

    def forward(self, input, mask=None, training=False):
        n_dim = self.opts['dim']
        result = self.vocab[input.flatten()]
        if self.batch_first:
            n_samples = input.shape[0]
            n_steps = input.shape[1]
            result = result.reshape([n_samples, n_steps, n_dim])
        else:
            n_steps = input.shape[0]
            n_samples = input.shape[1]
            result = result.reshape([n_steps, n_samples, n_dim])
        if training:
            result = dropout_calc(result, self.opts['dropout'])
        if mask is not None:
            return result * mask[...,None]
        else:
            return result

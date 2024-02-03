from __future__ import unicode_literals, print_function, division
import numpy as np
import torch

srng = np.random.RandomState(seed = 20210919)

class DropoutDim():
    '''
        Dropout along a specified dimension
    '''
    def __call__(self, x, dim, p):
        ndim = len(x.shape)
        if dim < 0:
            dim = ndim + dim
        size = x.shape[:dim+1]
        mask = self.dropout_calc(size, p)
        mask = torch.tensor(mask, dtype=torch.float32, device=x.device)
        dims = list(range(ndim-dim-1))
        for _ in dims:
            mask = mask.unsqueeze(-1)
        x = x*mask
        return x

    def dropout_calc(self, size, rate = 0.0):
        mask = srng.binomial(n=1, p=(1-rate), size=list(size))
        return mask


def fill_ignore_value(x, mask, ignore_value=-100):
    ignores = (1. - mask) * ignore_value
    x_ = x*mask + ignores.long()
    x_ = x_.contiguous()
    return x_


def one_hot_by_mask(mask):
    s=torch.sum(mask, dim=1, keepdim=True) - 1 # zero-indexed
    m=torch.zeros_like(mask)
    src=torch.ones_like(s)
    m.scatter_(dim=1, index=s, src=src)
    return m

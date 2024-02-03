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


def get_pos_map(struct_feature):
    pos_tensor = struct_feature[...,0]
    return pos_tensor

def get_inarc_map(struct_feature):
    inarc = struct_feature[...,1] if struct_feature.shape[-1] > 1 else None
    return inarc

def get_depth_map(struct_feature):
    depth = struct_feature[...,2] if struct_feature.shape[-1] > 2 else None
    return depth


def mask2symmtrx(masks, device, dtype=np.float32, batch_first=True, zero_diag=False):
    '''
    masks: [nL, nB] / [nB, nL]
    '''
    ns = len(masks.shape)
    assert ns <= 2
    if not batch_first and ns > 1:
        masks = masks.transpose(dim0=0, dim1=1)
    if ns == 1:
        masks = masks[None,...]
    m_lens = torch.sum(masks, dim=1).long()
    _, nL = masks.size()
    symmtrx = []
    zero_pads = [0] * nL
    for m, l in zip(masks, m_lens):
        symmtrx.append([m.cpu().tolist()] * l)
        symmtrx[-1].extend([zero_pads]*(nL-l))
    symmtrx = np.transpose(np.array(symmtrx, dtype=dtype))
    if zero_diag:
        symmtrx = symmtrx - np.eye(nL, dtype=dtype).reshape((nL, nL, 1))
    symmtrx[symmtrx==-1] = 0
    symmtrx = torch.tensor(symmtrx, device=device)
    return symmtrx


def fill_ignore_value(x, mask, ignore_value=-100):
    ignores = (1. - mask) * ignore_value
    x_ = x*mask + ignores.long()
    x_ = x_.contiguous()
    return x_

def remove_end_padding(mask, last_n):
    slens = torch.sum(mask, dim=1, keepdim=True)
    slefts = slens - last_n
    slens = slens - 1  # zero-indexed
    index = torch.cat((slefts, slens), dim=1)
    val = torch.zeros_like(index)
    new_mask = mask.scatter(1, index, val)
    return new_mask


def lookahead_1(threshold=0.5):
    s = srng.uniform(0.,1.,1)
    return s <= threshold

def lookahead_n(max_n, start=2):
    assert start < max_n
    s = srng.randint(2, max_n)
    return s

def occurences(x, mask=None):
    if mask is not None:
        lens_per_samples = torch.sum(mask, dim=1)
    else:
        lens_per_samples=[x.shape[1]]*x.shape[0]
        lens_per_samples = torch.tensor(lens_per_samples, device=x.device)
    #counts = [torch.bincount(x[i,:l]) for i, l in enumerate(lens_per_samples)]
    #cs = []
    #for i in range(len(lens_per_samples)):
    #    c = torch.gather(counts[i], dim=0, index=x[i])
    #    cs.append(c)
    #counts = torch.stack(cs, dim=0)
    # counts = [torch.gather(counts[i], dim=0, index=x[i]) for i in range(len(lens_per_samples))]
    counts = [torch.gather(torch.bincount(x[i,:l]), dim=0, index=x[i]) \
                for i, l in enumerate(lens_per_samples)]
    counts = torch.stack(counts, dim=0)
    return counts

def triangular_mask(input, zero_diag=True, up_tri=True):
    mi = torch.triu(input, diagonal=1)
    if zero_diag:
        mi = mi*(torch.eye(mi.shape[1], device=mi.device) < 1).long()
    m = (mi > 0).long()
    if not up_tri:
        return mi.transpose(1,2), m.transpose(1,2)
    else:
        return mi, m

def one_hot_by_mask(mask):
    '''
        Convert to one hot using the last non-zero bit.
    '''
    s=torch.sum(mask, dim=1, keepdim=True) - 1 # zero-indexed
    m=torch.zeros_like(mask)
    src=torch.ones_like(s)
    m.scatter_(dim=1, index=s, src=src)
    return m

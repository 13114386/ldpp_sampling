from __future__ import unicode_literals, print_function, division
import torch

def target_ce_loss(probas, y, y_mask, eps=1e-8):
    '''
    Based on https://github.com/KaiQiangSong/struct_infused_summ
    Reference to build_model
    '''
    # Apply proba for future words
    # y: [y0 y1, y2,..., ym]
    # p: [   p1, p2,..., pm, pm+1]
    # So, match 1~m terms only
    p_flat = probas[:,:-1].flatten() # nB x nL x nV (e.g. 64x1x5000)
    select_std_flat = y[:,1:].flatten() # nB x nL (e.g. 64x23)
    y_mask_flat = y_mask[:,1:].flatten() # nB x nL (e.g. 64x23)
    idxs = torch.arange(select_std_flat.shape[0], device=probas.device) # Index for each word in flatten y
    # idxs * prob.shape[2] -> set offset position for each word by vocab size
    idxs = idxs * probas.shape[2]
    # + select_std_flat -> add each word index to its offsetted position
    idxs += select_std_flat
    # negative log likelihood of each target word.
    cost = -torch.log(p_flat[idxs] + eps)
    # Sum up CE costs
    masked_cost = cost * y_mask_flat
    cost = masked_cost.sum() / y_mask_flat.sum()
    return cost

def attn_cost(alphas, x_mask, y_mask, weight_lambda=1.):
    '''
    Based on https://github.com/KaiQiangSong/struct_infused_summ
    Reference to build_model
    '''
    # Cumsum for each step with adding start from zero
    alph_t_cumsum = torch.cat([torch.zeros_like(alphas[:1], dtype = torch.float32),
                               torch.cumsum(alphas, axis = 0)], axis = 0)[:-1]
    # Constraint
    # y:   [y0 y1, y2,..., ym]
    # att: [   a1, a2,..., am, am+1]
    # The last am+1 (produced by eos) should not be part of calculation.
    # On the other hand, the first y0 is the bos, should be excluded too.
    # So, match 1~m terms only
    cost_attn = torch.minimum(alph_t_cumsum, alphas)[:-1,...] # Last pre
    nT, nB, nS = cost_attn.size()
    # Mask struction for summations over source sentence and then over target sentence
    masks = torch.ones((nT, nB, nS), dtype=torch.float32, device=alphas.device)
    masks = (masks * y_mask[:,1:].transpose(0,1)[:,:,None]) * x_mask[None,:,:]
    masks_flat = masks.flatten()
    masked_cost_attn = cost_attn.flatten() * masks_flat
    cost = weight_lambda * (masked_cost_attn.sum() / masks_flat.sum())
    return cost

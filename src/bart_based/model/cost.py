from __future__ import unicode_literals, print_function, division
import torch
import torch.nn.functional as F
from model.datautil import fill_ignore_value

def target_ce_loss(probas, y, y_mask, ignore_value=-100):
    '''
        probas: Must be probabilities
        y:      The target classes
    '''
    y = fill_ignore_value(y, y_mask, ignore_value=ignore_value)
    log_probas = torch.log(probas) # Apply log to softmax probas
    # Flatten
    nB, nL, nD = log_probas.size()
    log_probas = log_probas.view(-1, nD) # [nBxnL, nV]
    y = y.view(-1) # nBxnL
    cost = F.nll_loss(log_probas, y, ignore_index=ignore_value)
    return cost

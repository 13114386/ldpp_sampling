from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from model.mlp import MLP
from common.metric_meter import EarlyStopConditionByCount
from model.datautil import one_hot_by_mask

class DPPSearch(nn.Module):
    def __init__(self, opt, embedding_layer):
        super().__init__()
        self.opt = opt
        if self.opt["input_method"]["choice"] == "mlp":
            self.mlp = MLP(self.opt["input_method"]["layers"],
                           activation=F.relu,
                           batch_norm=self.opt["input_method"]["batch_norm"])
        self.embedding_layer = embedding_layer

    def forward(self, inputs):
        '''
            new_probas: New probabilistic distribution based on DPP search.
            max_score: Diversity scores. (May be used to form a cost function)
        '''
        probas = inputs["probas"]
        h_d = inputs["h_d"]
        mask = inputs["mask"].long()
        batch_vocab = inputs["batch_vocab"]
        best_choice, max_score = \
            self.dpp_search(probas,
                            h_d=h_d,
                            mask=mask,
                            batch_vocab=batch_vocab,
                            top_k=self.opt["top_K"],
                            n_iterations=self.opt["n_iterations"],
                            early_stop_cond=self.opt["early_stop_cond"])
        new_probas = self.diverse_proba(probas, best_choice, mask, self.opt["redist_weight"])
        return new_probas, max_score

    def dpp_search(
        self,
        probas,
        h_d,
        mask,
        batch_vocab,
        top_k,
        n_iterations,
        early_stop_cond
    ):
        '''
            Search for the max dpp
        '''
        # Get top k probas
        MAP = torch.argmax(probas, dim=-1, keepdim=True)
        topk_x = torch.topk(probas, top_k, dim=-1, largest=True, sorted=False)
        topk_probas = topk_x[0]
        topk_indices = topk_x[1]
        topk_probas[mask<1] = 1. # For Categorical to work.
        # Search
        max_score = None
        best_choice = None
        nb, nl, _ = topk_probas.size()
        cond = EarlyStopConditionByCount(early_stop_cond)
        sampler = Categorical(probs=topk_probas)
        for _ in range(n_iterations):
            # Sampling within topk
            choice = sampler.sample() # indices to the chosen
            # Map back the index into topk indices
            index = torch.reshape(choice, (nb,nl,1))
            samples = topk_indices.gather(dim=-1, index=index)
            # For the EOSs, they are deterministic ending for each sentence
            # and don't change predictive probabilities on them.
            one_hot = one_hot_by_mask(mask)
            samples[one_hot==1] = MAP[one_hot==1]
            # Get their word embeddings
            word_indices = batch_vocab[samples.flatten()]
            word_indices = word_indices.reshape(samples.shape)
            embs = self.embedding_layer(word_indices, mask, training=False)
            if self.opt["input_method"]["choice"] == "mlp":
                new_embs = torch.cat((embs, h_d*mask[...,None].float()), dim=-1)
                new_embs = self.mlp(new_embs)
            elif self.opt["input_method"]["choice"] == "additive":
                new_embs = (embs + h_d*mask[...,None].float()) / 10.
            # Compute DPP
            K = torch.bmm(new_embs, new_embs.transpose(1,2))
            # Determinants
            sLens = torch.sum(mask, dim=1)
            score = torch.stack([torch.det(K[i,0:l,0:l]) for i, l in enumerate(sLens)])
            # Choose the best subset of words
            if best_choice is None:
                max_score = score
                best_choice = samples
            else:
                maxd_mask = max_score < score
                if not torch.any(maxd_mask):
                    cond.incr()
                else:
                    cond.reset()
                if cond():
                    break
                # Update the max diversity
                max_score[maxd_mask] = score[maxd_mask]
                # Update samples
                best_choice[maxd_mask] = samples[maxd_mask]
        return best_choice, max_score

    def diverse_proba(self, x, best_choice, mask, redist_weight=1.):
        '''
            Update probability distribution by the best choice
        '''
        # Redistribute
        src = torch.ones_like(best_choice, dtype=x.dtype) * redist_weight
        assign_mask = torch.zeros_like(x) + (1-redist_weight)
        assign_mask.scatter_(dim=-1, index=best_choice, src=src)
        x = x * assign_mask
        # Renormalise
        if True:
            nm_factor = torch.sum(x, dim=-1, keepdim=True)
            nm_factor[mask==0]=1e-10 # To void divide by zeros
            x = x / nm_factor
        else:
            x = F.softmax(x, dim=-1)
        return x

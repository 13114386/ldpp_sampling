from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from model.mlp import MLP
from common.earlystop_cond import EarlyStopConditionByCount
from model.datautil import one_hot_by_mask


class DPPSearch(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.mlp = MLP(self.cfg["input_method"]["layers"],
                        activation=F.relu,
                        batch_norm=self.cfg["input_method"]["batch_norm"]) \
                    if self.cfg["input_method"]["use_mlp"] else None

    def forward(self, inputs, embedding_layer, embed_scale):
        '''
            new_probas: New probabilistic distribution based on DPP search.
            max_score: Diversity scores. (May be used to form a cost function)
        '''
        probas = inputs["probas"]
        h_d = inputs["h_d"]
        mask = inputs["mask"].long()
        batch_vocab = inputs.get("batch_vocab", None)
        best_choice, max_score = \
            self.dpp_search(probas,
                            h_d=h_d,
                            mask=mask,
                            top_k=self.cfg["top_K"],
                            batch_vocab=batch_vocab,
                            n_iterations=self.cfg["n_iterations"],
                            early_stop_cond=self.cfg["early_stop_cond"],
                            embedding_layer=embedding_layer,
                            embed_scale=embed_scale)
        new_probas = self.diverse_proba(
                            probas,
                            best_choice=best_choice,
                            mask=mask,
                            redist_weight=self.cfg["redist_weight"],
                            renormalise=self.cfg["renormalise"])
        return new_probas, max_score

    def dpp_search(
        self,
        probas,
        h_d,
        mask,
        top_k,
        batch_vocab,
        n_iterations,
        early_stop_cond,
        embedding_layer,
        embed_scale
    ):
        '''
            Search for the max dpp
        '''
        # Get top k probas
        MAP = torch.argmax(probas, dim=-1, keepdim=True)
        topk_x = torch.topk(probas, top_k, dim=-1, largest=True, sorted=False)
        topk_probas = topk_x[0]
        topk_indices = topk_x[1]
        # For Categorical to work on mini-batch,
        # assign 1.0 to padding word likelihoods over vocabulary.
        topk_probas[mask<1] = 1.
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
            # For BOSs and EOSs, they are deterministic starting and ending of each sentence
            # and so use the predictive MLE for DPP and cross entropy later.
            if self.cfg["mask_bos"]:
                samples[:,0] = MAP[:,0]
            if self.cfg["mask_eos"]:
                one_hot = one_hot_by_mask(mask)
                samples[one_hot==1] = MAP[one_hot==1]
            if self.cfg["input_method"]["choice"] in ["combined", "embedding"]:
                # Get their word embeddings
                if batch_vocab:
                    token_indices = batch_vocab[samples.flatten()]
                    token_indices = token_indices.reshape(samples.shape)
                else:
                    token_indices = samples.squeeze(dim=-1)
                embs = embedding_layer(token_indices) * embed_scale
                if self.cfg["input_method"]["choice"] == "combined":
                    embs = torch.cat((embs, h_d), dim=-1)
            else: #elif self.cfg["input_method"]["choice"] == "latent":
                embs = h_d
            if self.mlp is not None:
                embs = self.mlp(embs)
            # Compute L-DPP
            K = torch.bmm(embs, embs.transpose(1,2))
            # Determinants
            sLens = torch.sum(mask, dim=1)
            score = torch.stack([torch.logdet(K[i,0:l,0:l]) for i, l in enumerate(sLens)])
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

    def diverse_proba(
        self,
        x,
        best_choice,
        mask,
        redist_weight=1.,
        renormalise="hard",
        epsilon=1e-12
    ):
        '''
            Update probability distribution by the best choice
        '''
        src = torch.ones_like(best_choice, dtype=x.dtype) * redist_weight
        assign_mask = torch.zeros_like(x) + (1-redist_weight)
        assign_mask.scatter_(dim=-1, index=best_choice, src=src)
        x = x * assign_mask
        # Renormalise
        if renormalise == "hard":
            nm_factor = torch.sum(x, dim=-1, keepdim=True)
            nm_factor[mask==0] = epsilon # To void divide by zeros
            x = x / nm_factor
        else:
            x = F.softmax(x, dim=-1)
        return x



from common.metric_meter import ProgressCounter
from model.cost import target_ce_loss
class DPPCoverage():
    def __init__(self, cfg, dpp_search):
        self.cfg = cfg
        self.dpp_search = dpp_search
        self.freq_counter = ProgressCounter()

    def __call__(self, decoder_outputs, y, y_mask, embedding_layer, embed_scale):
        pred_cost = decoder_outputs.loss
        pred_cost, pred_probas = \
            self.dpp_regularise(logits=decoder_outputs.logits,
                                h_state=decoder_outputs.decoder_hidden_states[-1],
                                pred_cost=pred_cost,
                                y=y,
                                y_mask=y_mask,
                                batch_vocab=None,
                                embedding_layer=embedding_layer,
                                embed_scale=embed_scale)
        return pred_cost

    def dpp_regularise(self, logits, h_state, pred_cost, y, y_mask, batch_vocab,
                       embedding_layer, embed_scale):
        # DPP search
        # MLE distribution
        probas = F.softmax(logits, dim=-1)
        self.freq_counter += 1 # Incremental
        if self.freq_counter.count % self.cfg["freq"] == 0:
            dpp_input = {"probas": probas,
                        "h_d": h_state,
                        "mask": y_mask,
                        "batch_vocab": batch_vocab}
            probas_new, _ = self.dpp_search.forward(dpp_input,
                                                    embedding_layer=embedding_layer,
                                                    embed_scale=embed_scale)
            cost_dpp = target_ce_loss(probas_new, y, y_mask.long())
            if "RL" in self.cfg["cost"]: # Reinforcement cost
                if "MSE" in self.cfg["rl_risk_func"]:
                    cost_rl = F.mse_loss(cost_dpp, pred_cost)
                elif "ME" in self.cfg["rl_risk_func"]:
                    cost_rl = cost_dpp - pred_cost
                elif "MAE" in self.cfg["rl_risk_func"]:
                    cost_rl = F.l1_loss(cost_dpp, pred_cost)
                beta1 = self.cfg["beta1"]
                beta2 = self.cfg["beta2"]
                pred_cost = beta1*pred_cost + beta2*cost_rl
            elif "MDPP" in self.cfg["cost"]:
                beta1 = self.cfg["beta1"]
                beta2 = self.cfg["beta2"]
                pred_cost = beta1*pred_cost + beta2*cost_dpp
            return pred_cost, probas_new
        else:
            return pred_cost, probas

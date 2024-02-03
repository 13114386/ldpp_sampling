from __future__ import unicode_literals, print_function, division
'''
    Based on https://github.com/KaiQiangSong/struct_infused_summ
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.mlp import MLP

class Switcher(nn.Module):
    '''
    Calculate generation/copy probabilities
    Note: 
        Reference to switcher_pointer_decoder_calc (line 4~6) of the KaiQiangSong github
    '''
    def __init__(self, opts):
        super().__init__()
        # tanh activated layer
        n_in = opts["_tanh"]["n_in"]
        n_out = opts["_tanh"]["n_out"]
        self.proj_1 = nn.Linear(n_in, n_out, bias=True)
        self.activ_1 = eval('torch.'+opts["_tanh"]["activation"])
        # softmax activated layer
        n_in = opts["_softmax"]["n_in"]
        n_out = opts["_softmax"]["n_out"]
        self.proj_2 = nn.Linear(n_in, n_out, bias=True)
        self.activ_2 = eval('torch.'+opts["_softmax"]["activation"])
        # sigmoid activated layer
        n_in = opts["_switcher"]["n_in"]
        n_out = opts["_switcher"]["n_out"]
        self.proj_3 = nn.Linear(n_in, n_out, bias=True)
        self.activ_3 = eval('torch.'+opts["_switcher"]["activation"])

    def forward(self, inputs):
        # hidden state
        h_t = inputs["h_t"]
        y_t = inputs["y_t"]
        c_t = inputs["attn_c_t"]
        h_c_t = torch.cat((h_t, c_t), dim=-1)
        h_c_t = self.proj_1(h_c_t)
        h_c_t = self.activ_1(h_c_t)
        # hidden state distribution
        dist_t = self.proj_2(h_c_t)
        dist_t = self.activ_2(dist_t, dim=-1)
        # p_gen (switcher)
        p_gen = torch.cat((h_t, c_t, y_t), dim=-1)
        p_gen = self.proj_3(p_gen)
        p_gen = self.activ_3(p_gen)
        return dist_t, p_gen, h_c_t

class CopyNet(nn.Module):
    '''
    Choose between copying a source word and generating a new word
    Note: 
        Reference to build_model (line 213~218) of the KaiQiangSong github
    '''
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        dist_t = inputs["dist_t"]
        p_gen = inputs["p_gen"]
        alph_t = inputs["alph_t"]
        batch_vocab = inputs["batch_vocab"]
        pointer = inputs["pointer"]
        proba_t = self.calc_copy_proba(p_gen, alph_t, dist_t, batch_vocab, pointer)
        return proba_t

    def calc_copy_proba(self, p_gen, alpha, vocab, batch_vocab, pointer):
        '''
            vocab: generative vocab distribution
            batch_vocab: a larger vocab including the encoder source vocab
        '''
        if p_gen.shape[-1] < 2:
            nb, nl, ndim = vocab.shape
            # Cover both vocabs (source, target)
            padding = torch.zeros((nb, nl, batch_vocab.shape[0]-ndim),
                                    dtype = torch.float32, device=vocab.device)
            # Proba of generating a target word.
            prob_vocab = p_gen * torch.cat([vocab, padding], dim=2)
            # Proba of copying a source word.
            shp = pointer.shape
            # Source word index to the vocab.
            pointer = torch.reshape(F.one_hot(pointer.flatten(), batch_vocab.shape[0]).float(),
                                    [shp[0], shp[1], batch_vocab.shape[0]])
            # Multiply the source word attention distributed to the indexed word position.
            copyp = torch.bmm(alpha.permute(0,2,1), pointer)
            prob_posi = (1. - p_gen) * copyp # Proba of copying a source word.
            # Add up probas.
            proba = prob_vocab + prob_posi
            return proba
        else:
            raise ValueError("Not support p_gen ndim > 1")


# Reference:
#   def attention_decoder_init
#   def attention_calc
class Attention(nn.Module):
    def __init__(self, opts):
        super().__init__()
        self.cover_dims = opts["cover_dims"]
        n_in = opts["n_e"] + opts["n_d"] + self.cover_dims[-1]
        n_out = opts["n_att"]
        self.proj_1 = nn.Linear(n_in, n_out, bias=True)
        self.v = nn.Linear(n_out, 1, bias=False) # vector
        self.coverage = None
        if len(self.cover_dims) > 0 and self.cover_dims[0] > 0:
            self.coverage = MLP(self.cover_dims, activation=None) \
                            if len(self.cover_dims) > 1 else None
        self.activation = eval('torch.'+opts["activation"])

    def forward(self, inputs):
        '''
        Reference:
            def attention_decoder_calc
        '''
        dec_h_t = inputs["dec_h_t"]
        encoder_outputs = inputs["encoder_outputs"]
        x_mask = inputs["x_mask"]
        alphas = inputs["alpha_ts"]
        nb, nl, ndim = list(encoder_outputs.size()) # batch, len, ndim
        # Concat decoder h_t with encoder outputs
        dec_h_t_rep = dec_h_t.repeat(1,nl,1) * x_mask[...,None]
        x = torch.cat((encoder_outputs, dec_h_t_rep), -1)
        # Coverage
        coverage = None
        if len(self.cover_dims) > 0 and self.cover_dims[0] > 0:
            coverage = self.calc_coverage(alphas, nl, nb, gpu=encoder_outputs.is_cuda)
        if coverage is not None:
            x = torch.cat((x, coverage), dim=-1)
        x = self.proj_1(x) # Linear project
        x = torch.tanh(x)
        x = self.v(x)
        x_mask = x_mask.unsqueeze(-1)
        alph_t = self.activation(x*x_mask, dim=1) * x_mask # B x n_tk x 1
        # Renormalise after removing those zero-marked
        nm_factor = torch.sum(alph_t, dim=1, keepdim=True)
        alph_t = alph_t / nm_factor
        # Context
        # c_t = torch.sum(encoder_outputs * alph_t, dim=1).unsqueeze(dim=1) # B x 2*hidden_dim
        c_t = torch.bmm(alph_t.permute(0,2,1), encoder_outputs)
        return c_t, alph_t

    def calc_coverage(self, alphas, slen, nbatch, gpu=True):
        if len(alphas) == 0:
            ndim = self.cover_dims[-1]
            size = (nbatch, slen, ndim)
            if gpu:
                cov = torch.zeros(size, dtype=torch.float, device=torch.device('cuda'))
            else:
                cov = torch.zeros(size, dtype=torch.float)
            return cov
        else:
            # Cumsum along sequence/step dim.
            dim = 0
            if isinstance(alphas, list):
                alphas = torch.stack(alphas, dim).squeeze(-1)
            # Cumsum for each step but need only last step
            coverage = torch.cumsum(alphas, axis=dim)
            coverage = coverage[-1].unsqueeze(-1)
            if self.coverage:
                coverage = self.coverage(coverage)
            return coverage

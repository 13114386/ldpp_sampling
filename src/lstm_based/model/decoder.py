from __future__ import unicode_literals, print_function, division
'''
    Based on https://github.com/KaiQiangSong/struct_infused_summ
    Reference to build_model
'''
import torch
import torch.nn as nn
from model.attention import Attention, CopyNet, Switcher
from model.cost import target_ce_loss, attn_cost
from model.datautil import lookahead_1

class Decoder(nn.Module):
    def __init__(self, opts, weight_lambda,
                 embedding_layer, forcing_method,
                 apply_attn_cost,
                 generator, use_copy=True):
        super().__init__()
        self.opts = opts
        self.weight_lambda = weight_lambda
        self.embedding_layer = embedding_layer
        self.generator = generator
        self.forcing_method = forcing_method
        embedding_dim = opts["_lstm"]["n_in"]
        hidden_dim = opts["_lstm"]["n_out"]
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1,
                            batch_first=True, bidirectional=False,
                            dropout=opts["_lstm"]["dropout"])

        self.attention = Attention(opts["_att_1"])
        self.switcher = Switcher(opts)
        self.copynet = CopyNet() if use_copy else None
        self.apply_attn_cost = apply_attn_cost

    def forward(self, inputs, training=False):
        '''
        Reference:
            def attention_decoder_calc
            alph_ts (i.e. posi_), dist_ts (i.e. vocab_)
        '''
        if training:
            return self.train_forward(inputs)
        else:
            return self.test_forward(inputs)

    def train_forward(self, inputs):
        y = inputs["y"]
        y_mask = inputs["y_mask"]
        x_mask = inputs["x_mask"]
        s_t_1 = inputs["s_t_1"]
        encoder_outputs = inputs["encoder_outputs"]
        batch_vocab = inputs["batch_vocab"]
        pointer = inputs["pointer"]
        y_embedding = self.embedding_layer(y, y_mask, training=True)
        lens = int(torch.max(y_mask.sum(dim=1)).item())
        alpha_ts = []
        dist_ts = []
        proba_ts = []
        swt_h_ts = []
        h_ts = []
        for t in range(lens):
            if t == 0 or len(self.forcing_method) == 0 or \
                (self.forcing_method[0] == "teacher"):
                # Teacher forcing
                y_emb = y_embedding[:,t]
            elif self.forcing_method[0] == "predictive":
                if lookahead_1(0.8):
                    y_emb = y_embedding[:,t]
                else:
                    # Predictive forcing
                    proba = dist_ts[-1] if len(proba_ts) == 0 else proba_ts[-1]
                    gs = [self.generator.search(p, batch_vocab) for p in proba]
                    y_t_1 = torch.tensor(gs, dtype=torch.long,
                                        device=encoder_outputs.device)[:,None]
                    y_emb = self.embedding_layer(y_t_1, None, training=True).squeeze()
            elif self.forcing_method[0] == "<unk>":
                if lookahead_1(0.75):
                    y_emb = y_embedding[:,t]
                else:
                    nB = y.shape[0]
                    gs = [self.forcing_method[1]]*nB
                    y_t_1 = torch.tensor(gs, dtype=torch.long,
                                        device=encoder_outputs.device)[:,None]
                    y_emb = self.embedding_layer(y_t_1, None, training=True).squeeze()

            outputs = self.step(y_t=y_emb[:,None,:],
                                s_t_1=s_t_1,
                                encoder_outputs=encoder_outputs,
                                batch_vocab=batch_vocab,
                                pointer=pointer,
                                x_mask=x_mask,
                                alpha_ts=alpha_ts)
            s_t_1, attn_c_t_1, alph_t_1, dist_t_1, proba_t_1, swt_h_t, h_t = \
                outputs["lstm_s_t"], outputs["attn_c_t"], \
                outputs["attn_alph_t"], outputs["attn_dist_t"], \
                outputs["proba_t"], outputs["swt_h_t"], outputs["h_t"]
            # Book keeping
            alpha_ts.append(alph_t_1)
            dist_ts.append(dist_t_1)
            h_ts.append(h_t)
            swt_h_ts.append(swt_h_t)
            if proba_t_1 is not None:
                proba_ts.append(proba_t_1)
        cat_dim = 1
        h_ts = torch.cat(h_ts, dim=cat_dim)*y_mask[...,None]
        swt_h_ts = torch.cat(swt_h_ts, dim=cat_dim)*y_mask[...,None]
        probas = dist_ts if len(proba_ts) == 0 else proba_ts
        probas = torch.cat(probas, dim=cat_dim)*y_mask[...,None]
        alphas = torch.stack(alpha_ts, dim=0).squeeze(-1) # [nTarget, nBatch, nSrc]
        # MLE probability cost
        pred_cost = target_ce_loss(probas, y, y_mask)
        # Attentive cost
        att_cost = torch.tensor(0.0, dtype=torch.float32)
        if self.apply_attn_cost:
            att_cost = attn_cost(alphas=alphas,
                                x_mask=x_mask, y_mask=y_mask,
                                weight_lambda=self.weight_lambda)
        return {"alpha": alphas, "proba": probas, "h_ts": h_ts, "last_h_ts": swt_h_ts},\
               {"pred_cost": pred_cost, "att_cost": att_cost}

    def test_forward(self, inputs):
        x_mask = inputs["x_mask"]
        s_t_1 = inputs["s_t_1"]
        encoder_outputs = inputs["encoder_outputs"]
        batch_vocab = inputs["batch_vocab"]
        pointer = inputs["pointer"]
        max_step = inputs['max_len']
        alpha_ts = []
        dist_ts = []
        proba_ts = []
        swt_h_ts = []
        self.generator.reset((0, 0))
        for t in range(max_step):
            y_t_1 = self.generator.last()[0]
            y_t_1 = torch.tensor(y_t_1, dtype=torch.long,
                                 device=encoder_outputs.device)
            y_t_1 = y_t_1.reshape([1,1])
            y_emb = self.embedding_layer(y_t_1, None, training=False)
            outputs = self.step(y_t=y_emb,
                                s_t_1=s_t_1,
                                encoder_outputs=encoder_outputs,
                                batch_vocab=batch_vocab,
                                pointer=pointer,
                                x_mask=x_mask,
                                alpha_ts=alpha_ts)
            s_t_1, attn_c_t_1, alph_t_1, dist_t_1, proba_t_1, swt_h_t = \
                outputs["lstm_s_t"], outputs["attn_c_t"], \
                outputs["attn_alph_t"], outputs["attn_dist_t"], \
                outputs["proba_t"], outputs["swt_h_t"]
            # Book keeping
            alpha_ts.append(alph_t_1)
            dist_ts.append(dist_t_1)
            swt_h_ts.append(swt_h_t)
            if proba_t_1 is not None:
                proba_ts.append(proba_t_1)
            proba = dist_t_1 if proba_t_1 is None else proba_t_1
            wi = self.generator.search(proba, batch_vocab)
            self.generator.append(wi, alph_t_1)
        cat_dim = 1
        probas = dist_ts if len(proba_ts) == 0 else proba_ts
        probas = torch.cat(probas, cat_dim)
        alphas = torch.stack(alpha_ts, 0).squeeze(-1) # [nTarget, nBatch, nSrc]
        swt_h_ts = torch.cat(swt_h_ts, dim=cat_dim)
        return {"alpha": alphas, "proba": probas,
                "last_h_ts": swt_h_ts,
                "genwords": self.generator.genwords}

    def step(self, y_t, s_t_1, encoder_outputs,
            batch_vocab, pointer, x_mask, alpha_ts):
        # LSTM
        lstm_out, s_t = self.lstm(y_t, s_t_1)
        attn_inputs = {"dec_h_t": lstm_out,
                        "encoder_outputs": encoder_outputs,
                        "alpha_ts": alpha_ts,
                        "x_mask": x_mask}
        attn_c_t, alph_t = self.attention(attn_inputs)
        swt_inputs = {"y_t": y_t,
                      "attn_c_t": attn_c_t,
                      "h_t": lstm_out}
        dist_t, p_gen, swt_h_t = self.switcher(swt_inputs)
        proba_t = None
        if self.copynet:
            cpnet_inputs = {"dist_t": dist_t,
                            "p_gen": p_gen,
                            "alph_t": alph_t,
                            "batch_vocab": batch_vocab,
                            "pointer": pointer}
            proba_t = self.copynet(cpnet_inputs)
        return {"lstm_s_t": s_t,
                "h_t": lstm_out,
                "attn_c_t": attn_c_t,
                "attn_alph_t": alph_t,
                "attn_dist_t": dist_t,
                "swt_h_t": swt_h_t,
                "proba_t": proba_t}

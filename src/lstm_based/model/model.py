from __future__ import unicode_literals, print_function, division
import torch.nn.functional as F
import torch.nn as nn
from model.embedding import EmbeddingLayer
from model.encoder import Encoder, AgvEnc2Dec
from model.decoder import Decoder
from model.datautil import remove_end_padding
from model.cost import target_ce_loss
from model.dpp_search import DPPSearch
from model.modelutil import *
from common.metric_meter import ProgressCounter

class Model(nn.Module):
    def __init__(self, options, Vocab, generator, training=False, gpu=True):
        super().__init__()
        self.options = options
        model_cfg = options["model_cfg"]
        self.embedding = EmbeddingLayer(model_cfg["embedding"]["vocab"], Vocab,
                                        batch_first=True)
        self.encoder = Encoder(model_cfg["encoder"],
                               embedding_layer=self.embedding)
        self.enc2dec_state_h = AgvEnc2Dec(model_cfg["enc2dec_state"])
        self.enc2dec_state_c = AgvEnc2Dec(model_cfg["enc2dec_state"])
        # Decoder
        self.decoder = Decoder(opts=model_cfg["decoder"],
                                weight_lambda=options.get("weight_lambda", 1.),
                                embedding_layer=self.embedding,
                                forcing_method=options.get("forcing_method", None),
                                apply_attn_cost=training and "ATTN" in options["decoder_search"],
                                generator=generator)
        # DPP search
        if "dpp_search" not in self.options["exclude_modules"]:
            self.dpp_search = DPPSearch(opt=model_cfg["dpp_search"],
                                        embedding_layer=self.embedding)
            self.freq_counter = ProgressCounter()

        if gpu:
            self.cuda()

    def forward(self, inputs, training=False, iepoch=0):
        # Encoder work
        enc_inputs = {"x": inputs["x"],
                      "x_mask": inputs["x_mask"]}
        encoder_outputs, encoder_hidden = self.encoder(enc_inputs, training=training)

        # Collapse encoder hidden states [Dxn_layers,N,H] -> [1,N,H]
        # as initial hidden states of the decoder.
        h_state = self.enc2dec_state_h(encoder_hidden[0].transpose(0,1)).transpose(0,1)
        c_state = self.enc2dec_state_c(encoder_hidden[1].transpose(0,1)).transpose(0,1)
        encoder_hidden = (h_state, c_state)

        # Decoder work
        if training:
            y = inputs["y"]
            y_mask = inputs["y_mask"]
            dec_inputs = {"batch_vocab": inputs["batch_vocab"],
                            "pointer": inputs["pointer"],
                            "y": y,
                            "y_mask": y_mask,
                            "x_mask": inputs["x_mask"],
                            "encoder_outputs": encoder_outputs,
                            "s_t_1": encoder_hidden}
            decoder_states, decoder_costs = self.decoder(dec_inputs, training=training)
            m_output = {}
            m_output["cost"] = decoder_costs["att_cost"].cpu()
        else:
            dec_inputs = {"batch_vocab": inputs["batch_vocab"],
                            "pointer": inputs["pointer"],
                            "x_mask": inputs["x_mask"],
                            "encoder_outputs": encoder_outputs,
                            "s_t_1": encoder_hidden,
                            'max_len': self.options['max_len']}
            m_output = self.decoder(dec_inputs, training=training)

        # DPP regularisation
        if training:
            pred_probas = decoder_states["proba"]
            pred_cost = decoder_costs["pred_cost"]
            if "dpp_search" not in self.options["exclude_modules"] and \
                self.options["warmup"]["dpp_search"] <= iepoch:
                pred_cost, pred_probas = self.dpp_regularise(probas=pred_probas,
                                                        last_h_s=decoder_states["last_h_ts"],
                                                        pred_cost=pred_cost,
                                                        y=y,
                                                        y_mask=y_mask,
                                                        batch_vocab=inputs["batch_vocab"])
            m_output["cost"] += pred_cost.cpu()

        return m_output

    def dpp_regularise(self, probas, last_h_s, pred_cost, y, y_mask, batch_vocab):
        # DPP search
        self.freq_counter += 1 # Incremental
        if self.freq_counter.count % self.dpp_search.opt["freq"] == 0:
            y_mask_bos_shiftout = remove_end_padding(y_mask.long(),
                                                     len(self.options["padding"][0]))
            dpp_input = {"probas": probas,
                        "h_d": last_h_s,
                        "mask": y_mask_bos_shiftout,
                        "batch_vocab": batch_vocab}
            probas_new, _ = self.dpp_search.forward(dpp_input)
            cost_dpp = target_ce_loss(probas_new, y, y_mask.long())
            decoder_search = self.options["decoder_search"]
            if "RL" in decoder_search["cost"]: # Reinforcement cost
                if "MSE" in decoder_search["rl_risk_func"]:
                    cost_rl = F.mse_loss(cost_dpp, pred_cost)
                elif "ME" in decoder_search["rl_risk_func"]:
                    cost_rl = cost_dpp - pred_cost
                elif "MAE" in decoder_search["rl_risk_func"]:
                    cost_rl = F.l1_loss(cost_dpp, pred_cost)
                beta = decoder_search["beta"]
                pred_cost = beta*pred_cost + (1.-beta)*cost_rl
            if "MDPP" in decoder_search["cost"]: # Choose as MLE
                beta = decoder_search["beta"]
                pred_cost = beta*pred_cost + (1.-beta)*cost_dpp
            return pred_cost, probas_new
        else:
            return pred_cost, probas

    def analyse_emb(self, inputs, analyser, tokenizer):
        enc_inputs = {"x": inputs["x"],
                      "x_mask": inputs["x_mask"]}
        encoder_outputs, encoder_hidden = self.encoder(enc_inputs, training=False)

        h_state = self.enc2dec_state_h(encoder_hidden[0].transpose(0,1)).transpose(0,1)
        c_state = self.enc2dec_state_c(encoder_hidden[1].transpose(0,1)).transpose(0,1)
        encoder_hidden = (h_state, c_state)
        dec_inputs = {"batch_vocab": inputs["batch_vocab"],
                        "pointer": inputs["pointer"],
                        "x_mask": inputs["x_mask"],
                        "encoder_outputs": encoder_outputs,
                        "s_t_1": encoder_hidden,
                        'max_len': self.options['max_len']}
        decoder_states = self.decoder(dec_inputs, training=False)
        y_mask = tokenizer.get_mask(decoder_states["genwords"])
        pred_probas = decoder_states["proba"]
        _, _ = self.dpp_search.dpp_search(probas=pred_probas,
                                        h_d=decoder_states["last_h_ts"],
                                        mask=y_mask[None,:],
                                        batch_vocab=inputs["batch_vocab"],
                                        top_k=self.dpp_search.opt["top_K"],
                                        n_iterations=self.dpp_search.opt["n_iterations"],
                                        early_stop_cond=self.dpp_search.opt["early_stop_cond"],
                                        analyser=analyser)

def build_model(options, Vocab, log, checkpoint, generator=None,
                training=True, gpu=True, old_ver=False):
    model = Model(options=options, Vocab=Vocab, generator=generator,
                  training=training, gpu=gpu)
    if checkpoint:
        if old_ver:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            load_model_checkpoint(model, checkpoint, options, log)
    # Set up optimizers
    optimizers = None
    if training:
        if len(options["optimizers"]["choice"]) > 1:
            optimizers = setup_multi_optimizers(model, options, checkpoint)
        else:
            optimizers = setup_single_optimizers(model, options, checkpoint)
    return model, optimizers

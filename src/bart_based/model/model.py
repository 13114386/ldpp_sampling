from __future__ import unicode_literals, print_function, division
from typing import Optional
import torch
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM, CONFIG_MAPPING, BartConfig
from model.dpp_search import DPPCoverage, DPPSearch


class Model(nn.Module):
    def __init__(self, args, options, logger):
        super().__init__()

        # Instantiate baseline module
        if args.base_model_pretrained_name is not None:
            logger.info("Creating seq2seq model from pretrained weights.")
            self.seq2seq = AutoModelForSeq2SeqLM.from_pretrained(args.base_model_pretrained_name)
        elif options.base_model is not None:
            config = options.base_model.to_dict()
            logger.info("Creating seq2seq model from scratch using customized configuration.")
            self.seq2seq = AutoModelForSeq2SeqLM.from_config(BartConfig(**config))
        else:
            config = CONFIG_MAPPING[args.model_type]()
            logger.warning("You are instantiating a new config instance from scratch.")
            logger.info("Creating seq2seq model from scratch.")
            self.seq2seq = AutoModelForSeq2SeqLM.from_config(config)

        # Instantiate auxiliary modules
        self.dpp_search = None
        self.dpp_coverage = None
        if "dpp_search" not in options.aux_model.exclude_modules:
            d_model = self.seq2seq.get_decoder().config.d_model
            options.aux_model.dpp_search["input_method"]["layers"][0] = d_model*2
            self.dpp_search = DPPSearch(options.aux_model.dpp_search)
            self.dpp_coverage = DPPCoverage(cfg=options.aux_model.dpp_coverage,
                                            dpp_search=self.dpp_search)
        else:
            logger.warning("L-DPP Sampling is excluded.")


    def forward(self, batch, options, iepoch):
        inputs = batch[0]
        m_output = {}

        outputs = self.seq2seq(**inputs,
                               output_attentions=False,
                               output_hidden_states=True)

        if self.dpp_coverage and \
            options.training.warmup["dpp_search"] <= iepoch:
            y = inputs["labels"]
            y_mask = inputs["decoder_attention_mask"]
            embedding_layer = self.seq2seq.get_decoder().get_input_embeddings()
            embed_scale = self.seq2seq.get_decoder().embed_scale
            loss = self.dpp_coverage(outputs,
                                    y=y,
                                    y_mask=y_mask,
                                    embedding_layer=embedding_layer,
                                    embed_scale=embed_scale)
        else:
            loss = outputs.loss

        m_output["cost"] = loss
        return m_output

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        **model_kwargs,
    ):
        return self.seq2seq.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **model_kwargs,
        )

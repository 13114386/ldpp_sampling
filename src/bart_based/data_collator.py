from __future__ import unicode_literals, print_function, division
'''
    Modified version of https://github.com/huggingface/transformers/blob/v4.15.0/src/transformers/data/data_collator.py#L513
'''
import numpy as np
from dataclasses import dataclass
from typing import Any, Optional, Union
from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

@dataclass
class DataCollatorForSeq2Seq:
    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, batch, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        batch = list(zip(*batch)) # [(inputs, label),(inputs, label)] => [(inputs, inputs),(label, label)]
        inputs = batch[0]
        labels = batch[1] if len(batch) >= 2 else None
        indexes = batch[2] if len(batch) >= 3 else None

        inputs = self.tokenizer.pad(
            inputs,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        if labels is not None:
            labels = self.tokenizer.pad(
                labels,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=return_tensors,
            )
            mask_ignores = labels["attention_mask"] == 0
            labels["input_ids"][mask_ignores] = self.label_pad_token_id
            inputs["labels"] = labels.pop("input_ids")
            inputs["decoder_attention_mask"] = labels.pop("attention_mask")

        return (inputs, indexes)

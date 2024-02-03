from __future__ import unicode_literals, print_function, division
from abc import ABC, abstractmethod
from transformers import AutoTokenizer

class ModelTokenizer(ABC):
    def __call__(self, args, logger):
        model_type = args.tokenizer_name.split("/")[-1].split("-")[0]
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
        for split_type in args.split_type:
            self.process(args=args,
                         split_type=split_type,
                         tokenizer=tokenizer,
                         model_type=model_type,
                         logger=logger)

    @abstractmethod
    def process(
        self,
        args,
        split_type,
        tokenizer,
        model_type,
        logger
    ):
        raise NotImplementedError("Abstract method.")

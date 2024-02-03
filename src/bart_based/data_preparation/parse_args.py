from __future__ import unicode_literals, print_function, division
from abc import ABC, abstractmethod
import argparse


class ArgsParseBase(ABC):
    def __call__(self):
        parser = self.pre_parse()
        args = self.parse_args(parser)
        args = self.post_parse(args)
        return args

    @abstractmethod
    def pre_parse(self):
        raise NotImplementedError("ArgsParseBase.parse_args not implemented.")

    def parse_args(self, parser):
        args = parser.parse_args()
        return args

    def post_parse(self, args):
        return args


class ModelTokenizeDataArgsParse(ArgsParseBase):
    def pre_parse(self):
        parser = argparse.ArgumentParser(description="Build dataset")
        parser.add_argument('--dataset_root', type=str, default=".",
                            help='The root directory in which all data are to load and save.')
        parser.add_argument('--datasource_name', type=str, required=True,
                            help='One of ["cnndm", "gigaword"].')
        parser.add_argument('--source_folder', type=str, default=None,
                            help='The folder in which raw text data files reside.')
        parser.add_argument('--output_folder', type=str, default=None,
                            help='The folder to which token data is saved.')
        parser.add_argument('--source_file_ext', type=str, default=".txt",
                            help='The raw source text data file name pattern.')
        parser.add_argument('--output_file_ext', type=str, default=".json",
                            help='The output data file name pattern.')
        parser.add_argument('--split_type', type=str, required=True,
                            help='The data split type. '
                                'CNNDM - one of [train, validation, test], and '
                                'Gigaword - one of [train, dev, test].')
        parser.add_argument('--column_names', type=str, required=True,
                            help='CNNDM - ["article", "highlights"], and '
                                'Gigaword - ["document", "summary"].')
        parser.add_argument('--pairing_names', type=str, default=None,
                            help='CNNDM - None, Gigaword - ["src", "tgt"]')
        parser.add_argument('--tokenizer_name', type=str, required=True,
                            help="Base model - facebook/bart-base"
                                "CNNDM - facebook/bart-large-cnn"
                                "Gigaword - a1noack/bart-large-gigaword.")
        parser.add_argument('--truncation', type=str, default=True,
                            help='Truncate tokens by model specified or by max_length.')
        parser.add_argument('--max_length', type=str, default=None,
                            help='Model allowed length.')
        parser.add_argument('--error_log', type=str, default="error.{split_type}.log")
        parser.add_argument('--chunk_size', type=int, default=100,
                            help='Save annotation data by the chunk size.')

        return parser

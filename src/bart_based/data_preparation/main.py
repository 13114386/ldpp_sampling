from __future__ import unicode_literals, print_function, division
from parse_args import ModelTokenizeDataArgsParse
from one_input_model_tokenizer import OneInputFileModelTokenizer
from two_input_model_tokenizer import TwoInputFileModelTokenizer


def main():
    import logging
    logging.basicConfig(level=logging.NOTSET)
    logger = logging.getLogger(__name__)

    args = ModelTokenizeDataArgsParse()()

    if args.split_type is not None:
        args.split_type = args.split_type.strip(" []").split(",")
    if args.pairing_names is not None:
        args.pairing_names = args.pairing_names.strip(" []").split(",")
    if args.column_names is not None:
        args.column_names = args.column_names.strip(" []").split(",")

    # Build dataset
    if args.datasource_name in ["cnndm"]:
        OneInputFileModelTokenizer()(args, logger)
    elif args.datasource_name in ["gigaword"]:
        TwoInputFileModelTokenizer()(args, logger)
    logger.info("Done!")


if __name__ == "__main__":
    main()

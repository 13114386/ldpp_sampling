from __future__ import unicode_literals, print_function, division
from dataset import TDataset


def build_training_dataset(args, config, accelerator, logger):
    train_name, vaid_name = args.split_type
    train_dataset = TDataset(args=args,
                                split_type=train_name,
                                config=config,
                                logger=logger,
                                accelerator=accelerator)
    valid_dataset = TDataset(args=args,
                                split_type=vaid_name,
                                config=config,
                                logger=logger,
                                accelerator=accelerator)
    return (train_dataset, valid_dataset)


def build_test_dataset(args, config, accelerator, logger):
    test_name = args.split_type[0]
    test_dataset = TDataset(args=args,
                                split_type=test_name,
                                config=config,
                                logger=logger,
                                accelerator=accelerator)
    return test_dataset

from __future__ import unicode_literals, print_function, division
import os
from mylog.mylog import mylog
from options_loader import *
from utility.utility import *
from data_processor.data_manager import *
from data_processor.dataset import TrainingDataFactory
from generation.greedy import GreediGen
from model.model import build_model
from trainer import Trainer
from validator import Validator
from model.modelutil import parse_reload_options

def load_override(options, log):
    override_name = ""
    if options['reload'] == True:
        override_name = os.path.join(options['modeldata_root'],
                                    options["model_path"],
                                    options["model_name"],
                                    options["reload_options"])
    override_name = parse_reload_options(override_name)
    log.log("Override option path: {}".format(override_name))
    if not os.path.isfile(override_name):
        options["train_state"] = {}
        options["train_state"]['start_epoch'] = 0
        options["train_state"]['best_score'] = 1e99
        options["train_state"]['batch_count'] = 0
        options['reload'] = False
    else:
        override = optionsLoader(mode="train", log=log, disp=False, reload=override_name)
        assert override["train_state"]['reload_options'] in override_name
        options["model_cfg"] = override["model_cfg"]
        options["exclude_modules"] = override["exclude_modules"]
        options["train_state"] = override["train_state"]
    return options

def load(args):
    '''
    Port from https://github.com/KaiQiangSong/struct_infused_summ
    '''
    log = mylog()
    options = optionsLoader(mode="train", log=log, disp=True)
    # Overwrite
    if isinstance(args.vocab_dim, int):
        options["vocab_emb_dim"] = args.vocab_dim
    options['vocab_emb_init_path'] = \
        options['vocab_emb_init_path'].format(data=args.vocab_dir,
                                            dim=options["vocab_emb_dim"])
    options['my_vocab_path'] = \
        options['my_vocab_path'].format(data=args.vocab_dir,
                                        dim=options["vocab_emb_dim"])

    if args.dataset_root is not None:
        options["dataset_root"]=args.dataset_root
    if args.modeldata_dir is not None:
        options["modeldata_root"]=args.modeldata_dir
    log.log("dataset_root: "+options["dataset_root"])
    log.log("modeldata_root: "+options["modeldata_root"])
    options = load_override(options, log)
    Vocab_Giga = loadFromPKL(options['my_vocab_path'])
    log.log(str(Vocab_Giga.full_size)+', '+str(Vocab_Giga.n_in) + ', ' + str(Vocab_Giga.n_out))

    Vocab = {
        'w2i':Vocab_Giga.w2i,
        'i2w':Vocab_Giga.i2w,
        'i2e':Vocab_Giga.i2e
    }

    vocab_config_sync_check(options, Vocab)

    return log, options, Vocab

def main(args):
    use_cuda = torch.cuda.is_available()
    log, options, Vocab = load(args)
    if use_cuda:
        Vocab_i2e = torch.tensor(Vocab["i2e"], device=torch.device('cuda'))
    factory = TrainingDataFactory()
    train_dataloader, val_dataloader = factory(options, Vocab, log)
    checkpoint = None
    if options["reload"] and "reload_model" in options["train_state"]:
        model_path = os.path.join(options["modeldata_root"],
                                    options["model_path"],
                                    options["model_name"],
                                    options["train_state"]["reload_model"])
        if os.path.isfile(model_path):
            checkpoint = torch.load(model_path)
            log.log("Checkpoint is reloaded from: {}".format(model_path))
    if len(options["forcing_method"]) == 0: options["forcing_method"] += ["teacher"]
    generator = GreediGen() if options["forcing_method"][0] == "predictive" else None
    if options["forcing_method"][0] == "<unk>":
        options["forcing_method"] += [Vocab["w2i"][options["forcing_method"][0]]]
    model, optimizers = build_model(options=options, Vocab=Vocab_i2e, log=log,
                                   checkpoint=checkpoint,
                                   generator=generator,
                                   gpu=use_cuda)

    # Training
    validate_processor = Validator('validSet')
    train_processor = Trainer(model, options, validate_processor, optimizers)
    train_processor(options, [train_dataloader, val_dataloader], log)
    log.log("Training is Done")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Training run...')
    parser.add_argument('--dataset_root', type=str, required=False, default=None,
                        help='directory to which training dataset reside.')
    parser.add_argument('--modeldata_dir', type=str, required=False, default=None,
                        help='directory to which trained model data reside.')
    parser.add_argument('--vocab_dir', type=str, required=False, default='data',
                        help='directory to which vocabulary data reside')
    parser.add_argument('--vocab_dim', type=int, required=False, default=None,
                        help='The dimension size of vocabulary.')
    args = parser.parse_args()
    print("Command line args:\n", args)
    main(args)

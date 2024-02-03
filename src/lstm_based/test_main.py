from __future__ import unicode_literals, print_function, division
'''
Port from https://github.com/KaiQiangSong/struct_infused_summ
Reference to generate.py
'''
import os
import codecs
from datetime import datetime
from datasets import load_metric
from mylog.mylog import mylog
from data_processor.data_manager import vocab_config_sync_check
from data_processor.dataset import TestDataFactory
from generation.greedy import GreediGen
from options_loader import *
from model.model import build_model
from tester import Tester
from model.modelutil import parse_reload_options


def export_file(filename, str_list):
    with codecs.open(filename,'w',encoding = 'utf-8') as f:
        f.writelines(str_list)

def loadfromfile(fName):
    f = codecs.open(fName,'r',encoding = 'utf-8')
    result = []
    for l in f:
        line = l.strip().split()
        result.append(line)
    return result


def prepare(options, override_optname, Vocab, log, use_cuda, old_ver):
    '''
    options: base options
    override_optname: override option file name
    '''
    override = optionsLoader(mode="test", log=log, disp=False, reload=override_optname)
    options["model_cfg"] = override["model_cfg"]
    options["exclude_modules"] = override["exclude_modules"]

    checkpoint = None
    model_path = os.path.join(options["modeldata_root"],
                              options["model_path"],
                              options["model_name"],
                              override["train_state"]["reload_model"])
    assert os.path.isfile(model_path)
    checkpoint = torch.load(model_path)
    log.log("Checkpoint is loaded from: {}".format(model_path))
    model, _ = build_model(options=options, Vocab=Vocab, log=log,
                            checkpoint=checkpoint, generator=GreediGen(),
                            training=False, gpu=use_cuda,
                            old_ver=old_ver)
    return model, options

def load(args):
    log = mylog()
    options = optionsLoader(mode="test", log=log, disp=True)
    if isinstance(args.vocab_dim, int):
        options["vocab_emb_dim"] = args.vocab_dim
    options['vocab_emb_init_path'] = \
        options['vocab_emb_init_path'].format(data=args.vocab_dir,
                                            dim=options["vocab_emb_dim"])
    options['my_vocab_path'] = \
        options['my_vocab_path'].format(data=args.vocab_dir,
                                        dim=options["vocab_emb_dim"])

    Vocab_Giga = loadFromPKL(options['my_vocab_path'])
    log.log(str(Vocab_Giga.full_size)+', '+str(Vocab_Giga.n_in) + ', ' + str(Vocab_Giga.n_out))

    Vocab = {
        'w2i':Vocab_Giga.w2i,
        'i2w':Vocab_Giga.i2w,
        'i2e':Vocab_Giga.i2e
    }

    vocab_config_sync_check(options, Vocab)

    return log, options, Vocab

def rouge_eval(folder, inputs, show=True):
    rouge = load_metric("rouge")
    rouge_output = rouge.compute(predictions=inputs["summary"],
                                references=inputs["ref"],
                                rouge_types=["rouge1", "rouge2", "rougeL"])
    save_dir = os.path.join(folder, 'evaluation')
    os.makedirs(save_dir, exist_ok=True)
    fpath = os.path.join(save_dir, 'rouge_result.json')
    with open(fpath, 'w') as fp:
        # json.dump(rouge_output, fp, indent=4)
        print(rouge_output, file=fp)
    if show:
        print(rouge_output)

def dup_stat(folder, srcname):
    from tool.dupcount import repeated_word_stats
    outputname = "{}.dup.stat".format(srcname)
    repeated_word_stats(folder, srcname, outputname)

def main(args):
    use_cuda = torch.cuda.is_available()
    log, base_options, Vocab = load(args)
    if use_cuda:
        Vocab_i2e = torch.from_numpy(Vocab["i2e"]).cuda()

    if args.modeldata_root:
        base_options['modeldata_root'] = args.modeldata_root
    if args.dataset_root:
        base_options['dataset_root'] = args.dataset_root

    optionName = os.path.join(base_options['modeldata_root'],
                                base_options["model_path"],
                                base_options["model_name"],
                                base_options["reload_options"])
    optionName = parse_reload_options(optionName)

    now = datetime.now()
    result_folder = "result_" + now.strftime("%Y_%m_%d_%H_%M")

    for index, part in enumerate(base_options['subsets']):
        # Model
        log.log('Testing %dth model'%(index))
        model, options = prepare(base_options, optionName,
                                Vocab_i2e, log, use_cuda, args.old_ver)

        # Dataset
        factory = TestDataFactory()
        dataloader = factory(part, options, Vocab, log)

        fpath = os.path.join(base_options['dataset_root'],
                             base_options[part]["folder"],
                             base_options[part]["name"]+'.Ndocument')
        docset = loadfromfile(fpath)
        tester = Tester()
        output = tester(model, dataloader, docset, Vocab, options, log)
        # Save
        folder = os.path.join(base_options['dataset_root'],
                                base_options[part]["folder"],
                                result_folder)
        os.makedirs(folder, exist_ok=True)
        prefix = base_options[part]["name"]+'.result'
        doc_fname = os.path.join(folder, prefix+'.document')
        ref_fname = os.path.join(folder, prefix+'.reference')
        smry_fname = os.path.join(folder, prefix+'.summary')
        export_file(doc_fname, output["doc"])
        export_file(ref_fname, output["ref"])
        export_file(smry_fname, output["summary"])
        # Eval
        if args.eval:
            eval_method = eval(args.eval_method)
            eval_method(folder, output)
        if args.dup_stat:
            dup_stat(folder, prefix+'.summary')
    log.log('Finish Testing')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Test/Eval run...')
    parser.add_argument('--modeldata_root', type=str, required=False, default=None,
                        help='root directory to the trained model data.')
    parser.add_argument('--dataset_root', type=str, required=False, default=None,
                        help='root directory to the dataset.')
    parser.add_argument('--old_ver', type=bool, required=False, default=False,
                        help='directory to which vocabulary data reside.')
    parser.add_argument('--vocab_dir', type=str, required=False, default='data',
                        help='directory to which vocabulary data reside.')
    parser.add_argument('--vocab_dim', type=int, required=False, default=None,
                        help='The dimension size of vocabulary.')
    parser.add_argument('--eval', type=bool, required=False, default=False,
                        help='Evaluate summary against reference.')
    parser.add_argument('--eval_method', type=str, required=False, default="rouge_eval",
                        help='The evaluation method.')
    parser.add_argument('--dup_stat', type=bool, required=False, default=True,
                        help='Count word replication statistics.')
    args = parser.parse_args()
    main(args)

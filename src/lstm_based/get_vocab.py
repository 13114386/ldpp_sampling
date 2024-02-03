from __future__ import unicode_literals, print_function, division
import codecs, os, re
from mylog.mylog import mylog
from vocabulary.vocabulary import Vocabulary, I2E
from utility.utility import *
from options_loader import *


def solve_filepaths(root_dir, folder, name, options,
                    doc_ext=".Ndocument", summary_ext=".Nsummary"):
    def fetch_files(folder, docpath_pat, summpath_pat):
        files = []
        for fpath_pat in [docpath_pat, summpath_pat]:
            flist = [os.path.join(folder, f) for f in os.listdir(folder) if re.match(fpath_pat, f)]
            if len(flist) > 0:
                files.append(flist)
            print("===============================================")
            print(*flist, sep="\n")
        return files
    # file paths
    folder = os.path.join(root_dir, folder)
    docpath_pat = name+"_[0-9]+"+doc_ext
    summpath_pat = name+"_[0-9]+"+summary_ext
    files = fetch_files(folder, docpath_pat, summpath_pat)
    if len(files) == 2:
        return files[0], files[1]
    # Try single files
    docpath = name+doc_ext
    summpath = name+summary_ext
    files = fetch_files(folder, docpath, summpath)
    return files[0], files[1]

def create_vocab(args):
    log_vocab = mylog(logFile = 'log/log_vocab')
    options = optionsLoader(mode="train", log=log_vocab, disp=True)
    # Overwrite
    if isinstance(args.vocab_dim, int):
        options["vocab_emb_dim"] = args.vocab_dim
    options['vocab_emb_init_path'] = options['vocab_emb_init_path'].format(data=args.dest_dir,
                                                                           dim=options["vocab_emb_dim"])

    docCorpus, summCorpus = solve_filepaths(args.src_dir,
                                            options['trainSet']["folder"],
                                            options['trainSet']["name"],
                                            options)

    Vocab = Vocabulary(options, inputCorpus=docCorpus, outputCorpus=summCorpus)

    log_vocab.log(str(Vocab.full_size)+', '+str(Vocab.n_in) + ', ' + str(Vocab.n_out))

    os.makedirs(args.dest_dir, exist_ok=True)
    saveToPKL(os.path.join(args.dest_dir, args.outfile+'.Vocab'), Vocab)
    
    f = codecs.open(os.path.join(args.dest_dir, args.outfile+'.i2w'), 'w', encoding ='utf-8')
    st_opt = options["special_tokens"]
    special_tokens = st_opt[st_opt["choice"]]
    for item in Vocab.i2w:
        if item in special_tokens:
            print(item, 'NAN', file=f)
        else:
            print(item, Vocab.typeFreq[item], file=f)
    
    FeatureEmbedding = {}
    for feat in options["featList"]:
        FeatureEmbedding[feat] = I2E(options, feat)
    
    log_vocab.log(str(options["featList"]))
    saveToPKL(os.path.join(args.dest_dir, 'features.Embedding'), FeatureEmbedding)

    options_vocab = loadFromJson('settings/vocabulary.json')
    options_vocab['vocab_size'] = Vocab.full_size
    options_vocab['vocab_full_size'] = Vocab.full_size
    options_vocab['vocab_input_size'] = Vocab.n_in
    options_vocab['vocab_output_size'] = Vocab.n_out
    saveToJson('settings/vocabulary.json', options_vocab)


def main():
    import argparse
    # default_src_dir = r"/home/chshen/Data/n210801.mate/src/ml/dataset/gigaword"
    default_src_dir = r"../dataset/gigaword"
    default_dest_dir = r"./data"
    default_outfile = "my_vocab.Vocab"
    parser = argparse.ArgumentParser(description='Extract vocabulary from dataset')
    parser.add_argument('--src_dir', type=str, required=False, default=default_src_dir,
                        help='The directory of dataset.')
    parser.add_argument('--dest_dir', type=str, required=False, default=default_dest_dir,
                        help='The directory of dataset.')
    parser.add_argument('--vocab_dim', type=int, required=False, default=None,
                        help='The dimension size of vocabulary.')
    parser.add_argument('--outfile', type=str, required=False, default=default_outfile,
                        help='The filename of vocabulary output.')
    args = parser.parse_args()
    print(args)
    create_vocab(args)

if __name__ == '__main__':
    main()
    print("Done")

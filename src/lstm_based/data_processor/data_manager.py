from __future__ import unicode_literals, print_function, division
'''
    Partially adopt from https://github.com/KaiQiangSong/struct_infused_summ
'''
import numpy as np
from utility.utility import *


def Index2Word(Index, Vocab, options):
    # For some corner cases
    return Vocab['i2w'][Index]

def Index2Embedding(Index, Vocab, options):
    # For some corner cases
    return Vocab['i2e'][Index]

def Word2Index(Word, Vocab, options, flag):
    if not (Word in Vocab['w2i']):
        Word = '<unk>'
    elif flag and (Vocab['w2i'][Word] >= options['vocab_size']):
        Word = '<unk>'
    return Vocab['w2i'][Word]

def Word2Embedding(Word, Vocab,options, flag):
    return Index2Embedding(Word2Index(Word, Vocab, options, flag), Vocab, options)

'''
    Sentence Level Operations:
        Sentence2ListOfWord
        ListOfWord2ListOfIndex
        Sentence2ListOfIndex
        ListOfIndex2Embedding
        Sentence2Embedding 
        
        ListOfIndex2ListOfWord
        ListOfWord2Sentence
        ListOfIndex2Sentence
inps
    Options:
        endOfSentence:
            (add </s> after sentence)
'''

def Sentence2ListOfWord(sentence):
    listOfWord = sentence.split()
    return listOfWord

def ListOfWord2ListOfIndex(listOfWord, Vocab, options, flag):
    listOfIndex = []
    for w in listOfWord:
        listOfIndex.append(Word2Index(w, Vocab, options, flag))
    return listOfIndex
    
def Sentence2ListOfIndex(sentence, Vocab, options, flag):
    return ListOfWord2ListOfIndex(Sentence2ListOfWord(sentence),Vocab, options, flag)

def ListOfIndex2Embedding(listOfIndex, Vocab, options):
    emb = np.empty((0, options['emb_dim']), dtype = eval(options['dtype_float_numpy']))
    for wi in listOfIndex:
        emb = np.append(emb, Index2Embedding(wi, Vocab, options)[None,:], axis = 0)
    return emb

def Sentence2Embedding(sentence, Vocab, options, flag):
    return ListOfIndex2Embedding(Sentence2ListOfIndex(sentence, Vocab, options, flag), Vocab, options)

def ListOfIndex2ListOfWord(listOfIndex, Vocab, options):
    return [Index2Word(Index, Vocab, options) for Index in listOfIndex]

def ListOfWord2Sentence(listOfWord):
    first = True
    sentence = ''
    for word in listOfWord:
        if first:
            first = False
            sentence += word
        else:
            sentence += ' ' + word
    return sentence

def ListOfIndex2Sentence(listOfIndex, Vocab, options):
    return ListOfWord2Sentence(ListOfIndex2ListOfWord(listOfIndex, Vocab, options))

def cutDown(listOfIndex, maxStep = None):
    if maxStep == None:
        if 1 in listOfIndex:
            maxStep = listOfIndex.index(1)
        else:
            maxStep = len(listOfIndex)
    result = listOfIndex[0:min(len(listOfIndex), maxStep)]
    
    return result

'''
    Batched Data Preparation
'''
def ListOfEmbedding2BatchedEmbedding(listOfEmbedding, options, maxStep = None):
    if maxStep == None:
        n_steps = [emb.shape[0] for emb in listOfEmbedding]
        maxStep = max(n_steps)
        
    n_dim = listOfEmbedding[0].shape[1]
    data = np.empty((maxStep, 0 , n_dim), dtype = eval(options['dtype_float_numpy']))
    mask = np.empty((maxStep, 0), dtype = eval(options['dtype_float_numpy']))
    
    for emb in listOfEmbedding:
        data_i = np.append(emb, np.zeros((maxStep - emb.shape[0], n_dim), dtype = eval(options['dtype_float_numpy'])), axis = 0)
        data = np.append(data, data_i.reshape(data_i.shape[0], 1, data_i.shape[1]), axis = 1)
        
        mask_i = np.append(np.ones((emb.shape[0],1), dtype = eval(options['dtype_float_numpy'])),
                           np.zeros((maxStep - emb.shape[0],1), dtype = eval(options['dtype_float_numpy'])),
                           axis = 0)
        
        mask = np.append(mask, mask_i, axis = 1)
    return data, mask

def listOfListOfIndex2BatchedAnnotation(listOfListOfIndex, options ,maxStep = None):
    '''
    Padding data to max length of the batch, and create its mask for the original data.
    Note: first dim (vertical) is the sentence and second dim is batch
    '''
    if maxStep == None:
        n_steps = [len(it) for it in listOfListOfIndex]
        maxStep = max(n_steps)
        
    data = np.empty((maxStep, 0), dtype = eval(options['dtype_int_numpy']))
    mask = np.empty((maxStep, 0), dtype = eval(options['dtype_float_numpy']))
    
    for listOfIndex in listOfListOfIndex:
        data_i = np.array(listOfIndex, dtype = data.dtype)
        data_i = data_i.reshape(data_i.shape[0], 1)
        data_i = np.append(data_i, np.zeros((maxStep - len(listOfIndex),1), dtype = data_i.dtype), axis = 0)
        data = np.append(data, data_i, axis = 1)
        
        mask_i = np.append(np.ones((len(listOfIndex),1), dtype = mask.dtype),
                           np.zeros((maxStep - len(listOfIndex),1), dtype = mask.dtype),
                           axis = 0)
        mask = np.append(mask, mask_i, axis = 1)
    if options["batch_first"]:
        data = np.transpose(data, (1,0))
        mask = np.transpose(mask, (1,0))
    return data, mask


'''
    Generate Batches
'''
def LVT(x, options):
    vocab_LVT = set(list(range(0,options['decoder']['_softmax']['n_out'])))
    for l in x:
        vocab_LVT = vocab_LVT.union(set(l))
    vocab_LVT = list(vocab_LVT)
    vocab_LVT = sorted(list(vocab_LVT))
    dict_LVT = {}
    Index = 0
    for w in vocab_LVT:
        dict_LVT[w] = Index
        Index += 1
    return vocab_LVT, dict_LVT

def get_pointer(x, dict):
    result = []
    for l in x:
        result.append([dict[w] for w in l])
    return result

def sharpLVT(x, dict, special_tokens):
    '''
    If a word (by index) is not in the vocab/dictionary (e.g. 5000 words),
    replace the index with zero ('unk').
    '''
    unk_idx = special_tokens[special_tokens["choice"]].index("<unk>")
    result = []
    for l in x:
        temp = []
        for w in l:
            if w in dict:
                temp.append(dict[w])
            else:
                temp.append(unk_idx)
        result.append(temp)
    return result


def batch2Inputs_new(batch, options):
    '''
    Return inputs have the following data of which some may be None if not configured.
    inps = [input, input_mask] +
           [batch_vocab, pointer] +
           [output, output_mask]
    '''
    x = batch[0]
    y = batch[1]
    inps = []
    # Document data
    special_tokens = options["special_tokens"]
    model_cfg = options["model_cfg"]
    input, input_mask = listOfListOfIndex2BatchedAnnotation(x, options)
    inps += [input, input_mask]
    if model_cfg['LVT_available']:
        batch_vocab, batch_dict = LVT(x, model_cfg)
        batch_vocab = np.asarray(batch_vocab, dtype=np.int64)
        pointer = get_pointer(x, batch_dict)
        pointer, _ = listOfListOfIndex2BatchedAnnotation(pointer, options)
        y = sharpLVT(y, batch_dict, special_tokens)
        inps += [batch_vocab, pointer]
    else:
        inps += [None, None]

    # Summary ground truth
    output, output_mask = listOfListOfIndex2BatchedAnnotation(y, options)
    inps += [output, output_mask]
    return inps


def collect_data(inputs):
    '''
    Refer to batch2Inputs_new for inputs
    '''
    inp_dict = {"x": inputs[0], "x_mask": inputs[1],
                "y": inputs[4], "y_mask": inputs[5]}
    # Document feature stuff
    if inputs[2] is not None and inputs[3] is not None:
        inp_dict["batch_vocab"] = inputs[2]
        inp_dict["pointer"] = inputs[3]
    return inp_dict


def vocab_config_sync_check(options, Vocab):
    '''
        Make sure use vocab matching configuration
    '''
    choice = options["special_tokens"]["choice"]
    special_tokens = options["special_tokens"][choice]
    for w in special_tokens:
        assert w in Vocab['i2w'], "Vocab mismatch special_token configuration"

from __future__ import unicode_literals, print_function, division
'''
    Adopt from https://github.com/KaiQiangSong/struct_infused_summ
'''

import numpy as np
import os
import copy

from utility.utility import *
from collections import OrderedDict

class Vocabulary(object):
    '''
    Vocabulary:
        Vocabulary: minFreq = 5, '<unk>' to replace, sorted by frequency in referenced summaries
        Input Vocabulary: Full_Size
        Output_Vocabulary: Output_Size (types in referenced summaries)
    
    Functions:
    '''
    def __init__(self, options, inputCorpus = None, outputCorpus = None):
        self.cased = options['vocab_cased']
        self.sortedBy = options['vocab_freq_sortedBy']
        self.minFreq = options['vocab_minFreq']
        self.dim = options['vocab_emb_dim']
        self.init_path = options['vocab_emb_init_path']
        self.dtype = eval(options['dtype_float_numpy'])
        special_tokens = options["special_tokens"]
        st_choice = special_tokens["choice"]
        self.special_tokens = special_tokens[st_choice]
        assert os.path.isfile(self.init_path)
        assert(inputCorpus != None)
        assert(outputCorpus != None)
            
        typeFreq = self.setup(inputCorpus, outputCorpus, self.minFreq, self.cased, self.sortedBy)
        number, n_in, n_out, w2i, i2w, i2e = self.initialize(typeFreq, init_path = self.init_path,
                                                             cased = self.cased, dim = self.dim,
                                                             special_tokens = self.special_tokens)
        self.full_size = number
        self.n_in = n_in
        self.n_out = n_out
        self.w2i = w2i
        self.i2w = i2w
        self.i2e = i2e
        self.typeFreq = typeFreq
    
    def accumulate(self, fileName, Counter, cased = False, rmDigit = True):
        '''
        Count the number of occurrence of each word
        '''
        f = codecs.open(fileName, 'r', encoding = 'utf-8')
        for line in f:
            tokens = line.split()
            for token in tokens:
                word = token
                if not cased:
                    word = word.lower()
                if rmDigit:
                    word = remove_digits(word)
                if word not in Counter:
                    Counter[word] = 1
                else:
                    Counter[word] += 1
        f.close()
        return Counter
    
    def setup(self, inputCorpus, outputCorpus, minFreq=5, cased=False, rmDigit=True, sortedBy='output'):
        '''
        Create an ordered list of tuple of each word appearing in 
        input doc and output target summary. The tuple is in format
        (input word count, output word count, total count)
        '''
        typeFreq_input = {}
        typeFreq_output = {}
        if (type(inputCorpus) == str):
            typeFreq_input = self.accumulate(inputCorpus, typeFreq_input, cased, rmDigit)
        else:
            for fileName in inputCorpus:
                typeFreq_input = self.accumulate(fileName, typeFreq_input, cased, rmDigit)
        if (type(outputCorpus) == str):
            typeFreq_output = self.accumulate(outputCorpus, typeFreq_output, cased, rmDigit)
        else:
            for fileName in outputCorpus:
                typeFreq_output = self.accumulate(fileName, typeFreq_output, cased, rmDigit)
        
        typeFreq_Full = {}
        '''
        for key, value in typeFreq_input.items():
            if (key in typeFreq_output) and (typeFreq_input[key] + typeFreq_output[key] >= minFreq):
                typeFreq_Full[key] = (typeFreq_input[key], typeFreq_output[key], typeFreq_input[key] + typeFreq_output[key])
            elif typeFreq_input[key] >= minFreq :
                typeFreq_Full[key] = (typeFreq_input[key], 0, typeFreq_input[key])
        
        for key, value in typeFreq_output.items():
            if (key not in typeFreq_input) and (typeFreq_output[key] >= minFreq):
                typeFreq_Full[key] = (0, typeFreq_output[key], typeFreq_output[key])
        '''
        for key, value in typeFreq_output.items():
            if (typeFreq_output[key] >= minFreq):
                if (key in typeFreq_input):
                    typeFreq_Full[key] = (typeFreq_input[key], typeFreq_output[key], typeFreq_input[key] + typeFreq_output[key])
                else:
                    typeFreq_Full[key] = (0, typeFreq_output[key], typeFreq_output[key])
        
        for key, value in typeFreq_input.items():
            if (key not in typeFreq_output) and (typeFreq_input[key] >= minFreq):
                typeFreq_Full[key] = (typeFreq_input[key], 0, typeFreq_input[key])
        
        if sortedBy == 'input':
            select = 0
            another = 1
        elif sortedBy == 'output':
            select = 1
            another = 0
        else:
            select = 2
            another = 1
            
        typeFreq = OrderedDict(sorted(typeFreq_Full.items(), key = lambda x:(-x[1][select], -x[1][another])))
        return typeFreq
    
    def loadEmbedding(self, init_path, cased=False, rmDigit=True):
        '''
        Create high dimensional embedding numpy vector of each word 
        in the pretrained GloVe.6B.xxx embedding file
        '''
        embedding = {}
        f = codecs.open(init_path, 'r', encoding = 'utf-8')
        for line in f:
            items = line.split()
            word = items[0]
            if not cased:
                word = word.lower()
            if rmDigit:
                word = remove_digits(word)

            emb = list(map(float, items[1:]))
            emb = np.array(emb, dtype = self.dtype)
            embedding[word] = emb

        return embedding
                
    def initialize(self, typeFreq, init_path=None, cased=False, dim=100, special_tokens=None):
        '''
        Find words (in both input and output) embeddings from the pretrained GloVe
        '''
        if (init_path != None) and (init_path != ''):
            pretrained = self.loadEmbedding(init_path, cased = cased)
        else:
            pretrained = {}
        assert special_tokens is not None, "Missing special token configuration"
        i2w = copy.deepcopy(special_tokens)
        w2i = {}
        n_sts = len(i2w)
        for t, i in zip(i2w, range(n_sts)):
            w2i[t] = i

        number = n_sts
        n_in = 0
        n_out = 0

        for key, value in typeFreq.items():
            if key not in w2i:
                i2w.append(key)
                w2i[key] = number
                number += 1
            if key not in i2w[:n_sts]:
                if (value[0] > 0):
                    n_in += 1
                if (value[1] > 0):
                    n_out += 1
        i2e = np.empty((0, dim), dtype = self.dtype)
        for Index in range(0, number):
            type = i2w[Index]
            if (type in pretrained):
                embedding = pretrained[type].reshape(1, dim)
            else:
                embedding = random_weights(1, dim, 0.01)
            i2e = np.append(i2e, embedding, axis = 0)
        
        return number, n_in+n_sts, n_out+n_sts, w2i, i2w, i2e
 
class I2E(object):
    '''
    POS tag embeddings
    '''
    def __init__(self, options, feat):
        self.feat = feat
        self.dtype = eval(options['dtype_float_numpy'])
        self.number = options[feat+'_num']
        
        self.i2e = self.initialize(self.number, options[feat+'_dim'], options[feat+'_init_method'])
        
    def initialize(self, total, dim, initial_method):
        if (initial_method == 'random'):
            Index = 0
            i2e = np.empty((0, dim), dtype = self.dtype)
            for Index in range(total):
                emb = random_weights(1, dim, 0.01)
                i2e = np.append(i2e, emb, axis = 0)
        return i2e
    
class Mapping(I2E):
    def __init__(self, map, options, feat):
        self.feat = feat
        self.dtype = eval(options['dtype_float_numpy'])
        self.number = options[feat+'_num']
        
        self.i2w = map["i2w"]
        self.w2i = map["w2i"]
        self.i2e = self.initialize(self.number, options[feat+'_dim'], options[feat+'_init_method'])
    
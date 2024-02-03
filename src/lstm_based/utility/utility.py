from __future__ import unicode_literals, print_function, division
'''
    Partially from https://github.com/KaiQiangSong/struct_infused_summ
'''
import codecs, json, re
import pickle as Pickle
import numpy as np
import torch


def random_weights(n_in,n_out, scale = None):
    if scale is None:
        scale = np.sqrt(2.0 / (n_in + n_out))
    W = scale * np.random.randn(n_in,n_out)
    return W.astype('float32')

# activation_Function
def softmax_mask(x, mask):
    x = torch.softmax(x)
    x = x * mask
    x = x / x.sum(0,keepdims=True)
    return x

# IO
def loadFromJson(filename):
    f = codecs.open(filename,'r',encoding = 'utf-8')
    data = json.load(f,strict = False)
    f.close()
    return data

def saveToJson(filename, data):
    f = codecs.open(filename,'w',encoding = 'utf-8')
    json.dump(data, f, indent=4)
    f.close()
    return True

def saveToPKL(filename, data):
    with open(filename,'wb')as f:
        Pickle.dump(data, f)
    return 

def loadFromPKL(filename):
    with open(filename,'rb') as f:
        data = Pickle.load(f, encoding="latin1")
    return data

def remove_digits(parse):
    return re.sub(r'\d', '#', parse)

def process_trailing_stop(line, trailing_stop):
    if not trailing_stop:
        if line[-1] == '.':
            line = line[:-1].rstrip()
    return line

from __future__ import unicode_literals, print_function, division
'''
    Adopt from https://github.com/KaiQiangSong/struct_infused_summ
'''
from utility.utility import *
from collections import OrderedDict

'''
    The Goal of this file is to setup all the options of the experiment
    and prepare for experiments
'''

train_optionsFrame = {
    'vocabulary': 'settings/vocabulary.json',
    'dataType':'settings/dtype.json',
    'training':'settings/training.json',
    'dataset':'settings/my_train_settings.json',
    'network':'settings/model_config.json',
    'saveLoad':'settings/save_load.json',
    'earlyStop':'settings/early_stop.json'
}

test_optionsFrame = {
    'vocabulary': 'settings/vocabulary.json',
    'dataType':'settings/dtype.json',
    'test':'settings/test.json',
    'dataset':'settings/my_test_settings.json',
    'network':'settings/model_config.json',
    'saveLoad':'settings/save_load.json'
}

def optionsLoader(mode, log, disp=False, reload=None):
    if reload == None:
        log.log('Start Loading Options')
        options = OrderedDict()
        if mode == "train":
            optionsFrame = train_optionsFrame
        else:
            optionsFrame = test_optionsFrame
        for k,v in optionsFrame.items():
            log.log(k)
            option = loadFromJson(v)
            for kk,vv in option.items():
                if not kk in options:
                    options[kk] = vv
                else:
                    log.log('Options Error: conflict with ' + kk)
        log.log('Stop Loading Options')
    else:
        log.log('Start Reloading Options')
        options = loadFromJson(reload)
        log.log('Stop Reloading Options')
    if disp:
        print('Options:')
        for kk,vv in options.items():
            print('\t',kk,':',vv)
    return options

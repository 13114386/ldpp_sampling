from __future__ import unicode_literals, print_function, division
'''
    Adopt from https://github.com/KaiQiangSong/struct_infused_summ
'''
import os
import codecs
from time import time
from datetime import datetime

class mylog(object):
    def __init__(self, logFile='log/logFile', fileOutput=True, screenOutput=True):
        self.__st = time()
        self.__logFile = logFile
        self.__fileOutput = fileOutput
        self.__screenOutput = screenOutput
        d, f = os.path.split(self.__logFile)
        if not os.path.isdir(d): 
            os.mkdir(d)

    def get_start(self):
        return self.__st
        
    def set_start(self, st):
        self.__st = st
        
    def reset(self):
        self.__st = time()
        
    def get_time(self):
        return time() - self.__st
    
    def set_fileOutput(self, fileOutput):
        self.__fileOutput = fileOutput
        
    def set_screenOutput(self, screenOutput):
        self.__screenOutput = screenOutput
    
    def set_logFile(self, logFile):
        self.__logFile = logFile
        
    def log(self, msg, fileOutput = None, screenOutput = None):
        if fileOutput == None:
            fileOutput = self.__fileOutput
        if screenOutput == None:
            screenOutput = self.__screenOutput
        
        currentTime = self.get_time()
        
        if fileOutput == True:
            f =  codecs.open(self.__logFile,'a+',encoding='utf-8')
            f.write(str(datetime.now()) +' : '+str(msg) +'\n')
            f.close()
        
        if screenOutput == True:
            print(currentTime, ':', msg)

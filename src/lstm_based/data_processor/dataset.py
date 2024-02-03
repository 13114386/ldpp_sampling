from __future__ import print_function, division
import os
import random
import glob
import regex
import codecs
import torch
from torch.utils.data import Dataset, DataLoader
from utility.utility import (loadFromPKL,
                            remove_digits,
                            saveToPKL,
                            process_trailing_stop)
from data_processor.data_manager import (Sentence2ListOfIndex,
                                        cutDown,
                                        batch2Inputs_new,
                                        collect_data)

class DotstopSearch():
    def __init__(self):
        self.pat = regex.compile(r"\.\s*\n*$")

    def __call__(self, text):
        m = self.pat.search(text)
        return m!=None

class Subset:
    '''
    Adopt from https://github.com/KaiQiangSong/struct_infused_summ
    '''
    def __init__(self, folder, name, limit_len, dotstop_required, Vocab, options, log):
        self.name = name
        self.limit_len = limit_len
        self.dotstop_required = dotstop_required
        self.log = log
        self.Vocab = Vocab
        self.options = options
        self.path = os.path.join(folder, name)
        self.applied = False
        self.Data = {}

    def loadFromFile(self, fName, Vocab, options):
        documentName = fName + '.Ndocument'
        summaryName = fName + '{}.Nsummary'.format(options["summary_dotstop"])

        df = codecs.open(documentName,'r', encoding = 'utf-8')
        sf = codecs.open(summaryName,'r', encoding = 'utf-8')

        data = []
        anno = []
        
        count = 0
        dotstop_search = DotstopSearch()
        while (True):
            count += 1
            dLine = df.readline()
            sLine = sf.readline()

            if (not dLine) or (not sLine):
                break

            if self.dotstop_required and not dotstop_search(dLine):
                continue

            dLine = remove_digits(dLine.strip()).lower()
            sLine = remove_digits(sLine.strip()).lower()

            if (len(dLine.split()) < 1) or(len(sLine.split()) < 1):
                continue

            dLine = process_trailing_stop(dLine, trailing_stop=options["trailing_stop"])
            sLine = process_trailing_stop(sLine, trailing_stop=options["trailing_stop"])

            # Root head uses self-loop instead of -1.
            document = Sentence2ListOfIndex(dLine, Vocab, options, False)
            if self.limit_len and len(document) > options['max_posi']:
                document = cutDown(document, options['max_posi'])

            summary = Sentence2ListOfIndex(sLine, Vocab, options, False)
            paddings = options["padding"]
            padding_size = sum([len(i) for i in paddings])
            if self.limit_len and len(summary) > options['max_len'] - padding_size:
                summary = cutDown(summary, options['max_len']-2)
            summary = paddings[0] + summary + paddings[1]

            data.append(document)
            anno.append(summary)
        print(len(data), len(anno))
        return len(anno), (data, anno)

    def apply(self):
        if self.options['dataset_loading_method'] == 'load' and \
            os.path.isfile(self.path+'.data'):
            self.log.log('Loading Subset %s from PKL File'%(self.name))
            self.Data = loadFromPKL(self.path+'.data')
            self.applied = True
            self.log.log('Finish Loading Subset %s'%(self.name))
        else: #if self.options['dataset_loading_method'] == 'build':
            self.log.log('Building Subset %s from original text documents'%(self.name))
            number, data = self.loadFromFile(self.path, self.Vocab, self.options)
            self.Data['number'] = number
            self.Data['data'] = data
            # Pickle for fast load
            saveToPKL(self.path+'.data',self.Data)
            self.applied = True
            self.log.log('Finish Building Subset %s'%(self.name))
        return

    def data(self):
        if not self.applied:
            self.apply()
        return self.Data['data']

    def __len__(self):
        if not self.applied:
            self.apply()
        return self.Data['number']

    def __getitem__(self, indexes):
        if not self.applied:
            self.apply()
        data = []
        for i in indexes:
            if i < self.Data['number']:
                data.append((self.Data['data'][0][i],
                             self.Data['data'][1][i]))
            else:
                index = random.randint(0, self.Data['number']-1)
                data.append((self.Data['data'][0][index],
                             self.Data['data'][1][index]))
        return data


class MultiSubsetDataHelper():
    @staticmethod
    def extract_maxN(alist):
        nums = MultiSubsetDataHelper.extract_filenum(alist)
        maxN = max(nums)
        return maxN

    @staticmethod
    def extract_filenum(alist, order="sort"):
        pat = regex.compile("\w+\_(\d+)")
        nums = [int(pat.findall(f)[-1]) for f in alist]
        if order == "sort":
            nums.sort()
        elif order == "random":
            random.shuffle(nums)
        return nums

    @staticmethod
    def sample_indexes(nrange, nsamples):
        sample_idxs = random.sample(range(0, nrange), nsamples)
        return sample_idxs

    @staticmethod
    def calc_offset_subset(indexes, subset_size):
        choices = []
        for idx in indexes:
            which = idx // subset_size
            offset = idx % subset_size
            choices.append((which, offset))
        return choices

class MultiDataset(Dataset):
    def __init__(self, setname, options, Vocab, log):
        set_opt = options[setname]
        self.subset_size = set_opt["subset_size"]
        self.subsets = self.load_data(set_opt, options, Vocab, log)

    def load_data(self, set_opt, options, Vocab, log):
        folder = os.path.join(options["dataset_root"], set_opt["folder"])
        log.log("Load data from folder: {}".format(folder))
        fpath_pat = os.path.join(folder, set_opt["name"]+"_*.Ndocument")
        flist = glob.glob(fpath_pat)
        fnums = MultiSubsetDataHelper.extract_filenum(flist, order=set_opt["order"])
        subsets = []
        limit_len = set_opt.get("limit_len", True)
        dotstop_required = set_opt.get("dotstop_required", False)
        if len(fnums) == 0:
            subset = Subset(folder, set_opt["name"],
                            limit_len, dotstop_required,
                            Vocab, options, log)
            subsets.append(subset)
        else:
            log.log('Dataset file suffix number list: %s'%(fnums))
            name_pat = set_opt["name"]+"_{}"
            for i in fnums:
                set_part_name = name_pat.format(i)
                subset = Subset(folder, set_part_name,
                                limit_len, dotstop_required,
                                Vocab, options, log)
                subsets.append(subset)
        return subsets

    def __len__(self):
        return len(self.subsets) * self.subset_size

    def __getitem__(self, idx):
        choice = MultiSubsetDataHelper.calc_offset_subset([idx], self.subset_size)
        nth, ith = choice[0]
        data = self.subsets[nth].__getitem__([ith]) # subset index, indexes within the subset
        return data[0]


class SimpleDataset(Dataset):
    def __init__(self, setname, options, Vocab, log):
        set_opt = options[setname]
        self.subset = self.load_data(set_opt, options, Vocab, log)

    def load_data(self, set_opt, options, Vocab, log):
        folder = os.path.join(options["dataset_root"], set_opt["folder"])
        limit_len = set_opt.get("limit_len", True)
        dotstop_required = set_opt.get("dotstop_required", False)
        subset = Subset(folder, set_opt["name"],
                        limit_len, dotstop_required,
                        Vocab, options, log)
        return subset

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        data = self.subset.__getitem__([idx])
        return data[0]


class DataBatch():
    def __init__(self, data):
        self.data = data

    def __call__(self, options):
        '''
        Convert to desired batch data structure
        '''
        if options["sortByLength"]:
            self.data.sort(key=lambda item: len(item[0]))

        doc, anno = [], []
        for d in self.data:
            doc.append(d[0])
            anno.append(d[1])

        inputs = batch2Inputs_new((doc, anno), options)
        inputs = [torch.tensor(item, device=torch.device('cuda')) \
                    if item is not None else None for item in inputs]
        samples = collect_data(inputs)
        return samples, (doc, anno)


def my_collate_fn(data):
    return DataBatch(data)

class TrainingDataFactory():
    def __call__(self, options, Vocab, log):
        train_dataset = MultiDataset("trainSet", options, Vocab, log)
        val_dataset = SimpleDataset("validSet", options, Vocab, log)
        # Data loader
        batch_size = options["batch_size"]
        shuffle = options["batchShuffle"]
        num_workers = options.get("num_workers", 0)
        pin_memory = options.get("pin_memory", False)
        train_dataloader = DataLoader(train_dataset,
                                    batch_size=batch_size,
                                    shuffle=shuffle,
                                    collate_fn=my_collate_fn,
                                    num_workers=num_workers,
                                    pin_memory=pin_memory)

        val_dataloader = DataLoader(val_dataset,
                                    batch_size=batch_size,
                                    shuffle=shuffle,
                                    collate_fn=my_collate_fn,
                                    num_workers=num_workers,
                                    pin_memory=pin_memory)
        return train_dataloader, val_dataloader


class TestDataFactory():
    def __call__(self, setname, options, Vocab, log):
        dataset = SimpleDataset(setname, options, Vocab, log)
        batch_size = options["batch_size"]
        num_workers = options.get("num_workers", 0)
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                collate_fn=my_collate_fn,
                                num_workers=num_workers)
        return dataloader


if __name__ == "__main__":
    from utility.utility import loadFromJson
    from mylog.mylog import mylog

    log = mylog()

    Vocab_Giga = loadFromPKL('data/my_vocab.Vocab')
    Vocab = {
        'w2i':Vocab_Giga.w2i,
        'i2w':Vocab_Giga.i2w,
        'i2e':Vocab_Giga.i2e
    }

    if True:
        cfgs = ["settings/dtype.json",
                "settings/my_train_settings.json",
                "settings/training.json"]
        options = [loadFromJson(v) for v in cfgs]
        options = {**options[0], **options[1], **options[2], **options[3]}
        fac = TrainingDataFactory()
        train_dataloader, val_dataloader = fac(options, Vocab, log)
        for ibatch, batch in enumerate(train_dataloader):
            samples = batch(sort_by_length=options["sortByLength"])
            print(ibatch)

    if False:
        cfgs = ["settings/my_test_settings.json",
                "settings/test.json"]
        options = [loadFromJson(v) for v in [cfgs[1], cfgs[2]]]
        options = {**options[0], **options[1], **options[2]}
        fac = TestDataFactory()
        test_dataloader = fac(options, Vocab, log)

    print("Done!")

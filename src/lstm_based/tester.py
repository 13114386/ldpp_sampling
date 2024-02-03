from __future__ import unicode_literals, print_function, division
'''
Port from https://github.com/KaiQiangSong/struct_infused_summ
Reference to generate.py
'''
import torch
from data_processor.data_manager import batch2Inputs_new, collect_data, \
                                        ListOfIndex2Sentence, cutDown
from generation.greedy import TokenDecoder

class Tester():
    def __init__(self):
        super().__init__()

    def __call__(self, model, dataloader, docset, Vocab, options, log):
        '''
        dataloader: Data loader
        docset:  Documents (in text form) corresponding to inputs in dataset
        Vocab: Full vocab of all documents
        '''
        model.eval()
        tkn_decoder = TokenDecoder(Vocab, options)
        documents = []
        summaries = []
        references = []
        ref_start_n = len(options["padding"][0])
        ref_end_n = len(options["padding"][-1])
        ref_limit_len = options["evalSet"].get("limit_len", True)
        for idx, batch in enumerate(dataloader):
            samples, (doc, anno) = batch(options)
            with torch.no_grad():
                output = model(samples, training=False)

            summary = tkn_decoder.decode(output["genwords"], docset[idx], Vocab)
            # Reference
            if ref_end_n > 0:
                ref_sample = anno[0][ref_start_n:-ref_end_n]
            else:
                ref_sample = anno[0][ref_start_n:]
            if ref_limit_len:
                ref_sample = cutDown(ref_sample)
            reference = ListOfIndex2Sentence(ref_sample, Vocab, options)
            document = ListOfIndex2Sentence(doc[0], Vocab, options)

            documents.append(document+"\n")
            summaries.append(summary+"\n")
            references.append(reference+"\n")
        return {"doc": documents,
                "ref": references,
                "summary": summaries}

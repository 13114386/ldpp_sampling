from __future__ import unicode_literals, print_function, division
import torch
from utility.utility import remove_digits

class GreediGen():
    def __init__(self):
        super().__init__()
        # [seqence token index, dist]
        self.candidates = []#[(0, 's')]

    def reset(self, candidate):
        self.candidates.clear()
        self.candidates.append(candidate)

    def last(self):
        return self.candidates[-1]

    @property
    def genwords(self):
        return self.candidates[1:]

    def search(self, dist, batch_vocab):
        index = torch.argmax(dist, dim=-1) # Index to the batch_vocab
        word_index = batch_vocab[index] # Get the actual word index from batch_vocab
        wi = word_index.item()
        return wi

    def append(self, wi, alpha):
        self.candidates.append((wi, alpha))


class TokenDecoder():
    def __init__(self, Vocab, options):
        # Get eos_id and unk_id according to config
        if options["special_tokens"]["choice"] == "t5_tokenizer":
            self.eos_id = Vocab["w2i"]["<eos>"]
            self.unk_id = Vocab["w2i"]["<unk>"]
        elif options["choice"] == "struct_tokenizer":
            self.eos_id = Vocab["w2i"]["<s>"]
            self.unk_id = Vocab["w2i"]["<unk>"]

    def get_mask(self, inputs):
        '''
            Create mask for generated sentence by setting mask boundary
            at first occurrence of EOS token.
        '''
        mask = [0]*len(inputs)
        for i, item in enumerate(inputs):
            if item[0] != self.eos_id: # Ending indicator
                mask[i] = 1
            else: # Stop at first sight
                mask[i] = 1 # Inclusive
                break
        mask = torch.tensor(mask, dtype=torch.int64, device=item[1].device)
        return mask

    def decode(self, inputs, docset, Vocab):
        '''
        inputs: candidate sequence, a list of items.
                item format: (vocab index, source attention)
        '''
        sentence = ''
        for item in inputs:
            if item[0] == self.eos_id: # Ending indicator 
                break
            if item[0] == self.unk_id: # unkown
                attn = int(item[1].argmax())
                word = docset[attn]
            else: # Vocab index
                word = Vocab['i2w'][item[0]]
                if '#' in word:
                    cands = []
                    for index, token in enumerate(docset):
                        if remove_digits(token).lower() == word:
                            cands += [index]
                    if len(cands) > 0:
                        cidxs = torch.tensor(cands, dtype=torch.int64, device=item[1].device)
                        selected = item[1].index_select(dim=1, index=cidxs)
                        best_ci = torch.argmax(selected)
                        word_ci = cands[best_ci.item()]
                        word = docset[word_ci]
            word = word.lower()
            sentence += word + ' '
        return sentence

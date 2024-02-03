from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class AgvEnc2Dec(nn.Module):
    def __init__(self, opts, n_layers=1):
        super().__init__()
        layers = []
        self.opts = opts
        in_dim = opts["n_in"]
        out_dim = opts["n_out"]
        self.act = eval('torch.'+opts["activation"])
        for i in range(n_layers):
            layers.append(nn.Linear(in_dim, out_dim, bias=True))
            in_dim = out_dim
        self.mlp = nn.Sequential(*layers)

    def forward(self, input):
        x = input
        x = torch.mean(x, dim=1, keepdim=True)
        x = self.mlp(x)
        x = self.act(x)
        return x

class Encoder(nn.Module):
    def __init__(self, opts, embedding_layer):
        super().__init__()
        self.opts = opts
        self.embedding_layer = embedding_layer
        # BiLSTM x 2 layers
        self.lstm = nn.LSTM(opts["encoder_0"]["n_in"],
                            opts["encoder_1"]["n_out"],
                            num_layers=2,
                            batch_first=True, bidirectional=True,
                            dropout=opts["encoder_1"]["dropout"])

    def forward(self, inputs, training=False):
        '''
        Input is sorted in ascending order if opts["sortByLength"] is True.
        Pytorch pack_padded_sequence expects descending order if enforce_sorted is True.
        So, we need to reverse order if using opts["sortByLength"] to control enforce_sorted.
        '''
        x = inputs["x"]
        x_mask = inputs["x_mask"].long()
        x = self.embedding_layer(x, x_mask, training=training)
        seq_lens = x_mask.sum(dim=1).cpu() # Along length dim
        packed = pack_padded_sequence(x, seq_lens,
                                      batch_first=True,
                                      enforce_sorted=False)
        output, hidden = self.lstm(packed)
        encoder_outputs, _ = pad_packed_sequence(output, batch_first=True)  # h dim = B x t_k x n
        encoder_outputs = encoder_outputs.contiguous()
        return encoder_outputs, hidden

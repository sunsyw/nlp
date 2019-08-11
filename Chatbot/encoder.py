import torch.nn as nn
"""
Convert word indexes to embeddings.
Pack padded batch of sequences for RNN module.
Forward pass through GRU.
Unpack padding.
Sum bidirectional GRU outputs.
Return output and final hidden state.
"""


class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding

        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), bidirectional=True)

    def forward(self, input_seq, input_lengths, hidden=None):
        embedded = self.embedding(input_seq)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        outputs, hidden = self.gru(packed, hidden)
        # [seq_len, batch, hidden_size*2], [n_layers*2, batch, hidden_size]
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        # Sum bidirectional GRU outputs
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        # [seq_len, batch, hidden_size]
        return outputs, hidden
        # [seq_len, batch, hidden_size], [n_layers*2, batch, hidden_size]

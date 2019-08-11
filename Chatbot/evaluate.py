import torch
import torch.nn as nn
from load import normalizeString
from prepare_data import indexesFromSentence


MAX_LENGTH = 10
SOS_token = 1
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")


def evaluate(searcher, voc, sentence, max_length=MAX_LENGTH):
    indexes_batch = [indexesFromSentence(voc, sentence)]

    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])

    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)

    input_batch = input_batch.to(device)
    lengths = lengths.to(device)

    tokens, scores = searcher(input_batch, lengths, max_length)

    decoded_words = [voc.index2word[token.item()] for token in tokens]
    return decoded_words


def evaluateInput(searcher, voc):
    for i in range(5):
        try:
            # Get input sentence
            input_sentence = input('> ')
            # Check if it is quit case
            if input_sentence == 'q' or input_sentence == 'quit':
                break
            # Normalize sentence
            input_sentence = normalizeString(input_sentence)
            # Evaluate sentence
            output_words = evaluate(searcher, voc, input_sentence)
            # Format and print response sentence
            output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
            print('Bot:', ' '.join(output_words))

        except KeyError:
            print("Error: Encountered unknown word.")


class GreedySearchDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, input_length, max_length):
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        # print(encoder_outputs.size())  where i am ?  4+eos
        # [seq_len, 1, hidden_size] [5, 1, 500], [n_layers*2, 1, hidden_size]  [4, 1, 500]
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]  # [n_layers, 1, hidden_size]  [2, 1, 500]
        decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * SOS_token  # [[1]]

        all_tokens = torch.zeros([0], device=device, dtype=torch.long)  # []
        all_scores = torch.zeros([0], device=device)

        for _ in range(max_length):
            # Forward pass through decoder
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            # [1, output_size] [1, 18008], [n_layers, 1, hidden] [2, 1, 500]

            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            # [1] 每行最大值, [1] 每行最大值对应的列的索引
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)  # [1]
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)  # [1]

            decoder_input = torch.unsqueeze(decoder_input, 0)  # [1, 1]

        return all_tokens, all_scores  # [1 * max_len], [1 * max_len]

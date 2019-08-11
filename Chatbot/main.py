"""
https://pytorch.org/tutorials/beginner/chatbot_tutorial.html
"""

import torch
import os
from torch import optim
from load import prepareData
import torch.nn as nn
from encoder import EncoderRNN
from decoder import LuongAttnDecoderRNN
from train import trainIters
from evaluate import evaluateInput, GreedySearchDecoder

try:
    print("Start loading training data ... \n")
    voc = torch.load('save/voc.pkl')
    pairs = torch.load('save/pairs.pkl')
    print("Load voc and pairs success ... \n")
except FileNotFoundError:
    print("Saved data not found, start preparing training data ... \n")
    voc, pairs = prepareData()

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

# Configure models
model_name = 'cb_model'
attn_model = 'dot'
# attn_model = 'general'
# attn_model = 'concat'
hidden_size = 500
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1
batch_size = 64

# Set checkpoint to load from; set to None if starting from scratch
# loadFilename = None
checkpoint_iter = 2000
loadFilename = os.path.join('save/model.pkl')

print('Building encoder and decoder ...\n')
# Initialize word embeddings
embedding = nn.Embedding(voc.num_words, hidden_size)
# Initialize encoder & decoder models
encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)
checkpoint = None

# Load model if a loadFilename is provided
if loadFilename:
    # If loading on same machine the model was trained on
    print('Loading model.pkl ...\n')
    checkpoint = torch.load(loadFilename)
    # If loading a model trained on GPU to CPU
    # checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    encoder_optimizer_sd = checkpoint['en_opt']
    decoder_optimizer_sd = checkpoint['de_opt']
    embedding_sd = checkpoint['embedding']
    voc.__dict__ = checkpoint['voc_dict']
    print('Loading model success ...\n')

    embedding.load_state_dict(embedding_sd)

    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)
# Use appropriate device
encoder = encoder.to(device)
decoder = decoder.to(device)

#################################################
# Configure training/optimization

clip = 50.0
teacher_forcing_ratio = 1.0
learning_rate = 0.0001
decoder_learning_ratio = 5.0
n_iteration = 2000


# Ensure dropout layers are in train mode
encoder.train()
decoder.train()

# Initialize optimizers
print('Building optimizers ...\n')
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
if loadFilename:
    encoder_optimizer.load_state_dict(encoder_optimizer_sd)
    decoder_optimizer.load_state_dict(decoder_optimizer_sd)

# Run training iterations
# print("Starting Training!")
# trainIters(voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer, embedding,batch_size, clip,
#            loadFilename, checkpoint)

#################################################

# Set dropout layers to eval mode
encoder.eval()
decoder.eval()

# Initialize search module
searcher = GreedySearchDecoder(encoder, decoder)

# Begin chatting (uncomment and run the following line to begin)
evaluateInput(searcher, voc)

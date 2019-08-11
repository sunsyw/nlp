
import tensorflow as tf
from data import preprocess_sentence
from coder import Encoder, Decoder
from data import load_dataset

tf.enable_eager_execution()

path_to_file = 'cmn-eng/cmn.txt'
num_examples = 30000
input_tensor, target_tensor, inp_lang, targ_lang, max_length_inp, max_length_targ = load_dataset(path_to_file,
                                                                                                 num_examples)

vocab_inp_size = len(inp_lang.word2idx)
vocab_tar_size = len(targ_lang.word2idx)

BATCH_SIZE = 64
embedding_dim = 256
units = 1024


def evaluate(sentence, encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ):
    sentence = preprocess_sentence(sentence)

    inputs = [inp_lang.word2idx[i] for i in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=max_length_inp, padding='post')
    inputs = tf.convert_to_tensor(inputs)

    result = ''

    hidden = [tf.zeros((1, units))]

    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([targ_lang.word2idx['<start>']], 0)

    for t in range(max_length_targ):
        predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_out)

        predicted_id = tf.argmax(predictions[0]).numpy()

        result += targ_lang.idx2word[predicted_id] + ' '

        if targ_lang.idx2word[predicted_id] == '<end>':
            return result, sentence

        dec_input = tf.expand_dims([predicted_id], 0)

    return result, sentence


def translate(sentence, encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ):
    result, sentence = evaluate(sentence, encoder, decoder, inp_lang, targ_lang, max_length_inp,
                                max_length_targ)

    print('Input: {}'.format(sentence))
    print('Predicted translation: {}'.format(result))


encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)
decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)
optimizer = tf.train.AdamOptimizer()

# restoring the latest checkpoint in checkpoint_dir
checkpoint_dir = './'
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

translate('我在哪里？', encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ)
translate('我沒事。', encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ)
translate('干的好！', encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ)
translate('太可惜了！', encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ)

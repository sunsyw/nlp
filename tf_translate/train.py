import numpy as np
import tensorflow as tf
import time
from coder import Encoder, Decoder
import pickle
tf.enable_eager_execution()
from tqdm import tqdm


data = pickle.load(open('data/data.pkl', 'rb'))

input_tensor_train = data['input_tensor_train']
target_tensor_train = data['target_tensor_train']
vocab_inp_size = data['vocab_inp_size']
vocab_tar_size = data['vocab_tar_size']

BUFFER_SIZE = len(input_tensor_train)
BATCH_SIZE = 64
N_BATCH = BUFFER_SIZE//BATCH_SIZE
embedding_dim = 256
units = 1024

dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)
decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)

optimizer = tf.train.AdamOptimizer()

checkpoint_dir = 'ckpt'
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)


def loss_function(real, pred):
    mask = 1 - np.equal(real, 0)
    loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real, logits=pred) * mask
    return tf.reduce_mean(loss_)


for epoch in range(1):
    start = time.time()

    hidden = encoder.initialize_hidden_state()
    total_loss = 0

    bar = tqdm(enumerate(dataset), total=N_BATCH)
    for (batch, (inp, targ)) in bar:
        loss = 0

        with tf.GradientTape() as tape:
            enc_output, enc_hidden = encoder(inp, hidden)
            dec_hidden = enc_hidden

            dec_input = tf.expand_dims([1] * BATCH_SIZE, 1)

            # Teacher forcing - feeding the target as the next input
            for t in range(1, targ.shape[1]):
                predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
                # [batch_size * 1, vocab_size]  [1, batch_size, hidden_size]

                loss += loss_function(targ[:, t], predictions)  # [batch_size * 1, vocab_size]

                # using teacher forcing
                dec_input = tf.expand_dims(targ[:, t], 1)
                # [batch_size] -> [batch, 1]

            batch_loss = (loss / int(targ.shape[1]))  # 列数 vocab_size
            total_loss += batch_loss

            variables = encoder.variables + decoder.variables

            gradients = tape.gradient(loss, variables)

            optimizer.apply_gradients(zip(gradients, variables))

            # if batch % 10 == 0:
            #     print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, batch, batch_loss.numpy()))
            bar.set_description('epoch:{}\t batch:{}\t loss:{:.4f}\t'.format(epoch + 1, batch, batch_loss.numpy()))

    # saving (checkpoint) the model every 2 epochs
    checkpoint.save(file_prefix=checkpoint_dir)

    print('Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss / N_BATCH))
    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

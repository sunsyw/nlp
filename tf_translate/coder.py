import tensorflow as tf


def gru(units):
    # if tf.keras.is_gpu_available():
    #     return tf.keras.layers.CuDNNGRU(units,
    #                                     return_sequences=True,
    #                                     return_state=True,
    #                                     recurrent_initializer='glorot_uniform')
    #
    # else:
    return tf.keras.layers.GRU(units,
                               return_sequences=True,
                               return_state=True,
                               recurrent_activation='sigmoid',
                               recurrent_initializer='glorot_uniform')


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = gru(self.enc_units)

    def call(self, x, hidden):
        x = self.embedding(x)  # [seq_len, batch, embedding_dim]
        output, state = self.gru(x, initial_state=hidden)
        # [batch_size, max_len, hidden_size], [batch_size, hidden_size]
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))  # [batch, hidden]


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = gru(self.dec_units)
        self.fc = tf.keras.layers.Dense(vocab_size)

        self.W1 = tf.keras.layers.Dense(self.dec_units)
        self.W2 = tf.keras.layers.Dense(self.dec_units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, x, hidden, enc_output):
        """

        :param x:
        :param hidden: (batch_size, hidden_size)
        :param enc_output: (batch_size, max_length, hidden_size)
        :return:
        """
        hidden_with_time_axis = tf.expand_dims(hidden, 1)  # [batch_size, 1, hidden_size]
        score = self.V(tf.nn.tanh(self.W1(enc_output) + self.W2(hidden_with_time_axis)))
        # (batch_size, max_length, dec_units) + (batch, 1, dec_units) = (batch_size, max_length, dec_units)
        # [batch_size, max_len, 1]
        attention_weights = tf.nn.softmax(score, axis=1)

        context_vector = attention_weights * enc_output
        # [batch_size, max_len, 1] * [batch_size, max_length, hidden_size] = batch_size, max_length, hidden_size]
        context_vector = tf.reduce_sum(context_vector, axis=1)  # [batch_size, hidden_size]
        x = self.embedding(x)  # (batch_size, 1, embedding_dim)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        # [batch_size, hidden_size] -> [batch_size, 1, hidden_size] -> (batch_size, 1, embedding_dim + hidden_size)
        output, state = self.gru(x)
        # [batch_size, 1, hidden_size], [1, batch_size, hidden_size]
        output = tf.reshape(output, (-1, output.shape[2]))
        # [batch_size * 1, hidden_size]
        x = self.fc(output)  # [batch_size * 1, vocab_size]
        return x, state, attention_weights

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.dec_units))

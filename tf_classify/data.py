from tensorflow import keras
import pickle


imdb = keras.datasets.imdb  # 加载数据

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
# print(train_data[0])

word_index = imdb.get_word_index()  # 加载字典

word_index = {k: (v+3) for k, v in word_index.items()}
word_index['<PAD>'] = 0
word_index['<START>'] = 1
word_index['<UNK>'] = 2
word_index['UNUSED'] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


# train_data = decode_review(train_data[0])
# print(train_data)

# 调整长度
train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index['<PAD>'],
                                                        padding='post',
                                                        maxlen=256)
test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index['<PAD>'],
                                                       padding='post',
                                                       maxlen=256)

# 创建验证集
x_val = train_data[:10000]
print(x_val)
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]


pickle.dump(x_val, open('data/x_val.pkl', 'wb'))

pickle.dump(partial_x_train, open('data/partial_x_train.pkl', 'wb'))

pickle.dump(y_val, open('data/y_val.pkl', 'wb'))

pickle.dump(partial_y_train, open('data/partial_y_train.pkl', 'wb'))

pickle.dump(test_data, open('data/test_data.pkl', 'wb'))
pickle.dump(test_labels, open('data/test_labels.pkl', 'wb'))


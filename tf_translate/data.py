import unicodedata
import re
import jieba
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pickle


def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')


def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())
    # print(w)

    if w >= u'\u4e00' and w <= u'\u9fa5':
        # print('hans', w)
        w_obj = jieba.cut(w)
        w = ' '.join(w_obj)
    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."

    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)

    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    # w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)

    w = w.rstrip().strip()

    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    # w = '<start> ' + w + ' <end>'
    return w


def create_dataset(path, num_examples):
    lines = open(path, encoding='UTF-8').read().strip().split('\n')

    words_pairs = [[preprocess_sentence(w) for w in l.split('\t')] for l in lines[:num_examples]]

    return words_pairs


class LanguageIndex:
    def __init__(self, lang):
        self.lang = lang
        self.word2idx = {'<pad>': 0, '<start>': 1, '<end>': 2}
        self.idx2word = {}
        self.vocab = set()

        self.create_index()

    def create_index(self):
        for phrase in self.lang:
            self.vocab.update(phrase.split(' '))

        self.vocab = sorted(self.vocab)

        for index, word in enumerate(self.vocab):
            self.word2idx[word] = index + 3

        for word, index in self.word2idx.items():
            self.idx2word[index] = word


def max_length(tensor):
    return max(len(t) for t in tensor)


def load_dataset(path, num_examples):
    pairs = create_dataset(path, num_examples)
    # print(pairs)
    inp_lang = LanguageIndex(cmn for en, cmn in pairs)
    # print(inp_lang.word2idx)
    target_lang = LanguageIndex(en for en, cmn in pairs)
    # print(target_lang.word2idx)

    input_tensor = [[inp_lang.word2idx[s] for s in cmn.split(' ')] for en, cmn in pairs]
    target_tensor = [[target_lang.word2idx[s] for s in en.split(' ')] for en, cmn in pairs]

    max_length_inp, max_length_tar = max_length(input_tensor), max_length(target_tensor)

    # Padding the input and output tensor to the maximum length
    input_tensor = tf.keras.preprocessing.sequence.pad_sequences(input_tensor, maxlen=max_length_inp, padding='post')
    target_tensor = tf.keras.preprocessing.sequence.pad_sequences(target_tensor, maxlen=max_length_tar, padding='post')

    return input_tensor, target_tensor, inp_lang, target_lang, max_length_inp, max_length_tar


if __name__ == '__main__':
    # a = create_dataset('cmn-eng/cmn.txt', 10)
    # print(a)

    # a = load_dataset('cmn-eng/cmn.txt', 10)
    # print(a)
    path_to_file = 'cmn-eng/cmn.txt'
    num_examples = 30000
    input_tensor, target_tensor, inp_lang, targ_lang, max_length_inp, max_length_targ = load_dataset(path_to_file,
                                                                                                     num_examples)

    input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor,
                                                                                                    target_tensor,
                                                                                                    test_size=0.2)

    print(len(input_tensor_train), len(target_tensor_train), len(input_tensor_val), len(target_tensor_val))

    # pickle.dump(input_tensor_train, open('data/input_tensor_train.pkl', 'wb'))
    # pickle.dump(input_tensor_val, open('data/input_tensor_val.pkl', 'wb'))
    # pickle.dump(target_tensor_train, open('data/target_tensor_train.pkl', 'wb'))
    # pickle.dump(target_tensor_val, open('data/target_tensor_val.pkl', 'wb'))
    #
    vocab_inp_size = len(inp_lang.word2idx)
    vocab_tar_size = len(targ_lang.word2idx)
    # print(vocab_inp_size)
    #
    # pickle.dump(vocab_inp_size, open('data/vocab_inp_size.pkl', 'wb'))
    # pickle.dump(vocab_tar_size, open('data/vocab_tar_size.pkl', 'wb'))

    data = {}
    data['input_tensor_train'] = input_tensor_train
    data['input_tensor_val'] = input_tensor_val
    data['target_tensor_train'] = target_tensor_train
    data['target_tensor_val'] = target_tensor_val
    data['vocab_inp_size'] = vocab_inp_size
    data['vocab_tar_size'] = vocab_tar_size

    pickle.dump(data, open('data/data.pkl', 'wb'))

    data1 = pickle.load(open('data/data.pkl', 'rb'))
    print(data1)

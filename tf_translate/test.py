import unicodedata
import jieba
from tqdm import tqdm


def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')


def is_chinese(w):
    """判断一个unicode是否是汉字"""

    w = unicode_to_ascii(w.lower().strip())

    if w >= u'\u4e00' and w <= u'\u9fa5':

        return True

    else:

        return False


# a = is_chinese('?hi')
# print(a)
#
# b = is_chinese('?你')
# print(b)

# seg_list = jieba.cut("你懂了吗？")
# seg = ' '.join(seg_list)
# print(seg)
# print(type(seg))
#
# phrase = seg
# vocab = set()
# print(phrase.split(' '))
# vocab.update(phrase.split(' '))
# vocab = sorted(vocab)
# print(vocab)


# a = [i for i in tqdm(range(10000))]
# print(a)

import pickle

a = pickle.load(open('data/input_tensor_train.pkl', 'rb'))
print(a)
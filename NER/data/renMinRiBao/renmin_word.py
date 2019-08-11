
import re

from more_itertools import flatten
import pandas as pd
import numpy as np


def originHandle():
    with open("./renmin.txt", "r") as inp, open("./renmin2.txt", "w") as outp:
        for line in inp.readlines():
            line = line.split("  ")
            i = 1
            while i < len(line) - 1:
                if line[i][0] == "[":  # [中央/n', '人民/n', '广播/vn', '电台/n]nt',
                    outp.write(line[i].split("/")[0][1:])
                    i += 1
                    while i < len(line) - 1 and line[i].find("]") == -1:  # 找不到 "]"
                        if line[i] != "":
                            outp.write(line[i].split("/")[0])  # 人民
                        i += 1
                    outp.write(line[i].split("/")[0].strip() + "/" + line[i].strip("/")[1][-2:] + " ")
                elif line[i].split("/")[1] == "nr":  # nr: 人名
                    word = line[i].split("/")[0]
                    i += 1
                    if i < len(line) - 1 and line[i].split("/")[1] == "nr":
                        outp.write(word+line[i].split('/')[0]+'/nr ')
                        i += 1
                    else:
                        outp.write(word + "/nr")
                        continue
                else:
                    outp.write(line[i] + " ")
                i += 1
            outp.write("\n")


def originHandle2():
    with open('./renmin2.txt', 'r') as inp, open('./renmin3.txt', 'w') as outp:
        for line in inp.readlines():
            line = line.split(' ')
            i = 0
            while i < len(line)-1:
                if line[i] == '':
                    i += 1
                    continue
                word = line[i].split('/')[0]
                tag = line[i].split('/')[1]
                if tag == 'nr' or tag == 'ns' or tag == 'nt':  # 人名 地名 组织名  中共中央/nt
                    outp.write(word[0]+"/B_"+tag+" ")  # 中共中央/B_
                    for j in word[1:len(word)-1]:
                        if j != ' ':
                            outp.write(j+"/M_"+tag+" ")
                    outp.write(word[-1]+"/E_"+tag+" ")
                else:
                    for wor in word:
                        outp.write(wor+'/O ')
                i += 1
            outp.write('\n')


def sentence2split():
    with open('./renmin3.txt','r') as inp, open('./renmin4.txt','w') as outp:
        texts = inp.read()
        sentences = re.split(r'[，。！？、‘’“”:]/[O]', texts)
        for sentence in sentences:
            if sentence != " ":
                outp.write(sentence.strip()+'\n')


def data2pkl():
    datas = []
    labels = []

    tags = set()
    tags.add('')
    input_data = open('renmin4.txt', 'r')
    for line in input_data.readlines():
        line = line.split()
        linedata = []
        linelabel = []
        numNotO = 0
        for word in line:
            word = word.split('/')
            linedata.append(word[0])
            linelabel.append(word[1])
            tags.add(word[1])
            if word[1] != 'O':
                numNotO += 1
        if numNotO != 0:
            datas.append(linedata)
            labels.append(linelabel)

    input_data.close()

    # 选出需要的行
    print(len(datas))  # [[], [], ..]
    print(len(labels))

    # from compiler.ast import flatten
    all_words = flatten(datas)  # 合并 [...]
    sr_allwords = pd.Series(all_words)
    sr_allwords = sr_allwords.value_counts()
    """
    >>> import pandas as pd
    >>> data=pd.Series(['python','java','python','php','php','java','python','java'])
    >>> data
    0    python
    1      java
    2    python
    3       php
    4       php
    5      java
    6    python
    7      java
    dtype: object
    >>> 
    >>> data.value_counts()
    python    3
    java      3
    php       2
    dtype: int64
    """
    set_words = sr_allwords.index
    set_ids = range(1, len(set_words) + 1)

    """
    >>> data.index
    RangeIndex(start=0, stop=8, step=1)
    >>> set_words = data.index
    >>> set_idx = range(1, len(set_words) + 1)
    >>> set_idx
    range(1, 9)
    """

    tags = [i for i in tags]
    tag_ids = range(len(tags))
    word2id = pd.Series(set_ids, index=set_words)
    id2word = pd.Series(set_words, index=set_ids)
    tag2id = pd.Series(tag_ids, index=tags)
    id2tag = pd.Series(tags, index=tag_ids)
    word2id["unknow"] = len(word2id) + 1
    id2word[len(word2id)] = "unknow"
    print("tag2id:", tag2id)
    max_len = 60

    def X_padding(words):
        ids = list(word2id[words])
        if len(ids) >= max_len:
            return ids[:max_len]
        ids.extend([0] * (max_len - len(ids)))
        return ids

    def y_padding(tags):
        ids = list(tag2id[tags])
        if len(ids) >= max_len:
            return ids[:max_len]
        ids.extend([0] * (max_len - len(ids)))
        return ids

    df_data = pd.DataFrame({'words': datas, 'tags': labels}, index=range(len(datas)))
    df_data['x'] = df_data['words'].apply(X_padding)
    df_data['y'] = df_data['tags'].apply(y_padding)
    x = np.asarray(list(df_data['x'].values))
    y = np.asarray(list(df_data['y'].values))

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=43)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=43)

    import pickle

    with open('../renmindata.pkl', 'wb') as outp:
        pickle.dump(word2id, outp)
        pickle.dump(id2word, outp)
        pickle.dump(tag2id, outp)
        pickle.dump(id2tag, outp)
        pickle.dump(x_train, outp)
        pickle.dump(y_train, outp)
        pickle.dump(x_test, outp)
        pickle.dump(y_test, outp)
        pickle.dump(x_valid, outp)
        pickle.dump(y_valid, outp)
    print('** Finished saving the data.')


# originHandle()
# originHandle2()
# sentence2split()
data2pkl()

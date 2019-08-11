
import pickle

import torch
import torch.optim as optim
from pytorch.BiLSTM_CRF import BiLSTM_CRF
from pytorch.resultCal import calculate
from tqdm import tqdm


with open('../data/renmindata.pkl', 'rb') as inp:
    word2id = pickle.load(inp)
    id2word = pickle.load(inp)
    tag2id = pickle.load(inp)
    id2tag = pickle.load(inp)
    x_train = pickle.load(inp)
    y_train = pickle.load(inp)
    x_test = pickle.load(inp)
    y_test = pickle.load(inp)
    x_valid = pickle.load(inp)
    y_valid = pickle.load(inp)
print("train len:", len(x_train))
print("test len:", len(x_test))
print("valid len", len(x_valid))


#############
START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 100
HIDDEN_DIM = 200
EPOCHS = 1

# print(tag2id)
tag2id[START_TAG] = len(tag2id)
tag2id[STOP_TAG] = len(tag2id)
# print(tag2id)
"""
            0
M_nt        1
O           2
E_nt        3
M_nr        4
E_nr        5
B_nr        6
E_ns        7
B_ns        8
B_nt        9
M_ns       10
<START>    11
<STOP>     12
dtype: int64
"""

# model = BiLSTM_CRF(len(word2id) + 1, tag2id, EMBEDDING_DIM, HIDDEN_DIM)
#
# optimizer = optim.SGD(model.parameters(), lr=0.005, weight_decay=1e-4)


# index = 0
# bar = tqdm(total=len(x_train))
# for sentence, tags in zip(x_train, y_train):
#     bar.update(1)
#     index += 1
#     model.zero_grad()
#
#     sentence = torch.tensor(sentence, dtype=torch.long)  # (60)
#     tags = torch.tensor([tag2id[t] for t in tags], dtype=torch.long)  # 转化为tensor
#
#     loss = model.neg_log_likelihood(sentence, tags)
#
#     loss.backward()
#     optimizer.step()
#     if index % 5000 == 0:
#         torch.save(model, "./model.pkl")

model = torch.load("./model.pkl")
# entityres = []
# entityall = []
# bar = tqdm(total=len(x_test))
# for sentence, tags in zip(x_test, y_test):
#     bar.update(1)
#     sentence = torch.tensor(sentence, dtype=torch.long)
#     score, predict = model(sentence)
#     entityres = calculate(sentence, predict, id2word, id2tag, entityres)
#     entityall = calculate(sentence, tags, id2word, id2tag, entityall)
# jiaoji = [i for i in entityres if i in entityall]  # 交集
# if len(jiaoji) != 0:
#     zhun = float(len(jiaoji)) / len(entityres)
#     zhao = float(len(jiaoji)) / len(entityall)
#     print("test:")
#     print("zhun:", zhun)
#     print("zhao:", zhao)
#     print("f:", (2 * zhun * zhao) / (zhun + zhao))
# else:
#     print("zhun:", 0)


# 澳/B_ns 门/E_ns 和/O 台/B_ns 湾/E_ns 同/O 胞/O
def predict(sentence):
    sentence = [word2id[i] for i in sentence]
    sentence = torch.tensor(sentence)
    score, pred = model(sentence)
    print(pred)
    print([id2tag[i] for i in pred])


predict("澳门和台湾同胞")
predict("北京在哪")
predict("孙有为")



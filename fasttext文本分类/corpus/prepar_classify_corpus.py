'''
处理准备分类的数据

数据分析 如何 转 数据挖掘 ？	__label__QA
你 知道 谁么	__label__chat
'''
from lib.cut_sentence import cut
from tqdm import tqdm
import json
import random


xiaohuangji_path = "/Users/sun/PycharmProjects/chat_service/corpus/classify/origin_corpus/小黄鸡未分词.conv"
corpus_by_hand_path = "/Users/sun/PycharmProjects/chat_service/corpus/classify/origin_corpus/手动构造的问题.json"
crawled_data_path = "/Users/sun/PycharmProjects/chat_service/corpus/classify/origin_corpus/爬虫抓取的问题.csv"


def save_xiaohuangji(q, f_save, f_test, by_word):
    result = cut(q, by_word)
    if len(result) > 1:
        result = ' '.join(result) + '\t' + '__label__chat' + '\n'

        if random.choice([0, 0, 0, 0, 1]) == 1:
            f_test.write(result)
        else:
            f_save.write(result)


def process_xiaohuangji(f_save, f_test, by_word=False):
    f = open(xiaohuangji_path).readlines()
    temp_list = []
    for line in tqdm(f):
        line = line.strip().lower()
        if line.startswith('e'):
            if len(temp_list) > 0:
                line = temp_list[0]
                line = line[1:].strip()
                save_xiaohuangji(line, f_save, f_test, by_word)  # 提取出问题
            temp_list = []
        else:
            temp_list.append(line)

    if len(temp_list) > 0:
        line = temp_list[0]
        line = line[1:].strip()
        save_xiaohuangji(line, f_save, f_test, by_word)


def process_qa(f_save, f_test, by_word=False):
    qa_dict = json.load(open(corpus_by_hand_path))
    for key, value in tqdm(qa_dict.items()):
        for temp in value:
            for q in temp:
                q = q.strip()
                result = cut(q, by_word=by_word)
                if len(result) > 1:
                    result = " ".join(result) + "\t" + "__label__QA" + "\n"
                    if random.choice([0, 0, 0, 0, 1]) == 1:
                        f_test.write(result)
                    else:
                        f_save.write(result)

    f = open(crawled_data_path).readlines()
    for line in tqdm(f):
        line = line.strip()
        result = cut(line, by_word=by_word)
        if len(result) > 1:
            result = " ".join(result) + "\t" + "__label__QA" + "\n"

            if random.choice([0, 0, 0, 0, 1]) == 1:
                f_test.write(result)
            else:
                f_save.write(result)


def prepar_classify_corpus(by_word=False):
    if by_word:
        f_save = open('/Users/sun/PycharmProjects/chat_service/corpus/classify/qa_by_word.txt', 'a')
        f_save_test = open("/Users/sun/PycharmProjects/chat_service/corpus/classify/test_qa_by_word.txt", "a")
    else:
        f_save = open("/Users/sun/PycharmProjects/chat_service/corpus/classify/qa.txt", "a")
        f_save_test = open("/Users/sun/PycharmProjects/chat_service/corpus/classify/test_qa.txt", "a")

    process_qa(f_save=f_save, f_test=f_save_test, by_word=by_word)
    process_xiaohuangji(f_save, f_save_test, by_word)


if __name__ == '__main__':
    prepar_classify_corpus()



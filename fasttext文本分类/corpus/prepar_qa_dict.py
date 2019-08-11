"""
准备问答对
{
问题：{
        问题：
        答案：
        主体：
        问题单个字分词：
        问题分词：
    }
}
"""
from lib.cut_sentence import cut
import pandas as pd
import json
from tqdm import tqdm


Q_path = '/Users/sun/PycharmProjects/chat_service/corpus/recall/origin_corpus/Q.txt'
A_path = '/Users/sun/PycharmProjects/chat_service/corpus/recall/origin_corpus/A.txt'
excel_path = '/Users/sun/PycharmProjects/chat_service/corpus/recall/origin_corpus/excel.xlsx'


def prepar_qa_dict():
    fq = open(Q_path).readlines()
    fa = open(A_path).readlines()
    cur_dict = {}
    for q, a in zip(fq, fa):
        q = q.strip().lower()
        a = a.strip()
        cur_dict[q] = {}
        cur_dict[q]['ans'] = a
        cur_dict[q]["cuted"] = cut(q, by_word=False)
        cur_dict[q]["cuted_by_word"] = cut(q, by_word=True)

        temp = cut(q, by_word=False, with_sg=True)
        cur_dict[q]['main_enyiyt'] = [i[0] for i in temp if i[-1] == 'kc']

    df = pd.read_excel(excel_path)

    for q, a in zip(df['问题'], df['答案']):
        q = q.strip().lower()
        a = a.strip()
        cur_dict[q] = {}
        cur_dict[q]["ans"] = a
        cur_dict[q]["cuted"] = cut(q, by_word=False)
        cur_dict[q]["cuted_by_word"] = cut(q, by_word=True)
        temp = cut(q, by_word=False, with_sg=True)
        cur_dict[q]["main_entity"] = [i[0] for i in temp if i[-1] == "kc"]

    json.dump(cur_dict, open('/Users/sun/PycharmProjects/chat_service/corpus/recall/QA.json', 'w'),
              ensure_ascii=False, indent=2)


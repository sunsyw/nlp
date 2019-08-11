'''
提供分类预测的的接口
'''
import fastText
import config


class Classify:
    def __init__(self, by_word=False):
        self.by_word = by_word
        model_path = config.classify_model_path if not by_word else config.classify_model_path_by_word
        self.model = fastText.load_model(model_path)

    def predict(self, sentence):
        """
        :param sentence:{"sentence":Str,"cuted":["","",],"cuted_by_word":["",""...],entity:[python]}
        :return:
        """
        sentence = ' '.join(sentence['cuted'] if not self.by_word else sentence['cuted_by_word'])
        label, acc = self.model.predict(sentence)
        label = label[0]
        acc = acc[0]

        if label == '__label__qa':
            if acc > config.classify_V:
                return True
            else:
                return False
        else:
            return False

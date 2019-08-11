"""
进行分类模型的训练
"""
import fastText
import config


def build_model(by_word=False):
    input_path = config.classify_qa_path if not by_word else config.classify_qa_path_by_word
    model = fastText.train_supervised(input_path, wordNgrams=2, minCount=3, epoch=20)
    save_path = config.classify_model_path if not by_word else config.classify_model_path_by_word
    model.save_model(save_path)


def evaluate_model(by_word=False):
    # 加载模型
    model_path = config.classify_model_path if not by_word else config.classify_model_path_by_word
    model = fastText.load_model(model_path)

    # 读取测试集的数据
    test_path = config.classify_qa_test_path if not by_word else config.classify_qa_test_path_by_word
    test_qa = []
    test_label = []
    # 分开特征标签
    for i in open(test_path).readlines():
        qa, label = i.strip().split('\t')
        test_qa.append(qa)
        test_label.append(label)

    pred = model.predict(test_qa)  # [[__label__],[],..]
    # 把pred和test_label进行对比，计算均值，得到准确率
    # print('pred:', pred)
    pred = [i[0] for i in pred[0]]
    print('pred0:', pred)
    print('pred_len:', len(pred))
    acc = sum([1 if test_label[i] == pred[i] else 0 for i in range(len(pred))]) / len(pred)
    print('acc:', acc)
    print('wrong', len(pred) - len(pred)*acc)


def eval(sentence, by_word=False):
    # model_path = config.classify_model_path if not by_word else config.classify_model_path_by_word
    model = fastText.load_model("./model/model.pkl")
    ret = model.predict(sentence)
    return ret


if __name__ == '__main__':
    # build_model()
    # evaluate_model()
    print(eval('python 是 什么'))
    print(eval('python 好学 吗'))
    print(eval('今天 天气 怎么样'))

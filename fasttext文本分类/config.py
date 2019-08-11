"""
配置文件
"""

# ##################### lib 相关的配置 ###################
user_dict_path = "/Users/sun/PycharmProjects/chat_service/corpus/lib/keywords.txt"
stopwords_path = "/Users/sun/PycharmProjects/chat_service/corpus/lib/stopwords.txt"


# ################ classify 相关的配置 ####################
# 分词
classify_qa_path = "/Users/sun/PycharmProjects/chat_service/corpus/classify/qa.txt"  # 训练集
classify_qa_test_path = "/Users/sun/PycharmProjects/chat_service/corpus/classify/test_qa.txt"  # 测试集

# 按照单个字进行分词
classify_qa_path_by_word = "/Users/sun/PycharmProjects/chat_service/corpus/classify/qa_by_word.txt"  # 训练集
classify_qa_test_path_by_word = "/Users/sun/PycharmProjects/chat_service/corpus/classify/test_qa_by_word.txt"  # 测试集

classify_model_path = "/Users/sun/PycharmProjects/chat_service/classify/model/model.pkl"
classify_model_path_by_word = "/Users/sun/PycharmProjects/chat_service/classify/model/model_by_word.pkl"

classify_V = 0.9

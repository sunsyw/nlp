import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator


class TransferModel(object):

    def __init__(self):

        self.model_size = (224, 224)
        self.train_dir = "./data/train/"
        self.test_dir = "./data/test/"
        self.batch_size = 32

        self.train_generator = ImageDataGenerator(rescale=1.0 / 255)
        self.test_generator = ImageDataGenerator(rescale=1.0 / 255)

    def read_img_to_generator(self):
        """
        读取本地固定格式数据
        :return:
        """
        train_gen = self.train_generator.flow_from_directory(directory=self.train_dir,
                                                             target_size=self.model_size,
                                                             batch_size=self.batch_size,
                                                             class_mode='binary',
                                                             shuffle=True)
        test_gen = self.test_generator.flow_from_directory(directory=self.test_dir,
                                                           target_size=self.model_size,
                                                           batch_size=self.batch_size,
                                                           class_mode='binary',
                                                           shuffle=True)
        return train_gen, test_gen
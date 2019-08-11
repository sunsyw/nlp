from tensorflow.python.keras.applications import VGG16
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
from tensorflow.python.keras.applications.vgg16 import preprocess_input, decode_predictions

model = VGG16()

# print(model.summary())

image = load_img('./cat.png', target_size=(224, 224))
image = img_to_array(image)

image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

# 输入数据进行预测,进行图片的归一化处理
image = preprocess_input(image)
y_predict = model.predict(image)

# 进行结果解码
label = decode_predictions(y_predict)
# 进行label获取
res = label[0][0]

print('预测的类别为：%s 概率为：(%.2f%%)' % (res[1], res[2]*100))

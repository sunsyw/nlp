import tensorflow as tf
from tensorflow import keras
import pickle


# 构建模型
vocab_size = 10000
model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))
# model.summary()

# 损失函数和优化器
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='binary_crossentropy',
              metrics=['accuracy'])


x_val = pickle.load(open('data/x_val.pkl', 'rb'))

partial_x_train = pickle.load(open('data/partial_x_train.pkl', 'rb'))

y_val = pickle.load(open('data/y_val.pkl', 'rb'))

partial_y_train = pickle.load(open('data/partial_y_train.pkl', 'rb'))

# 训练模型
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1,)

model.save('model/1.h5')





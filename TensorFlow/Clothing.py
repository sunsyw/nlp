from tensorflow import keras
import tensorflow as tf
import numpy as np


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])


(train, train_label), (test, test_label) = keras.datasets.fashion_mnist.load_data()
train = train / 255.0
test = test / 255.0


model.compile(optimizer=keras.optimizers.SGD(lr=0.01),
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])

model.fit(train, train_label, epochs=5, batch_size=32)

test_loss, test_acc = model.evaluate(test, test_label)
print(test_loss, test_acc)

predictions = model.predict(test)
print(predictions[0])

print(np.argmax(predictions[0]))
print(test_label[0])

from tensorflow import keras
import pickle
import tensorflow as tf


test_data = pickle.load(open('data/test_data.pkl', 'rb'))
test_labels = pickle.load(open('data/test_labels.pkl', 'rb'))

new_model = keras.models.load_model('model/1.h5')
new_model.summary()

new_model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

loss, acc = new_model.evaluate(test_data, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))

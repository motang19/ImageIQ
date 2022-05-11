import tensorflow as tf
import numpy as np
from tensorflow import keras

import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

model = keras.Sequential([
    keras.layers.Flatten(input_shape= (28,28)),

    keras.layers.Dense(128, activation=tf.nn.relu),

    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.optimizers.Adam(), loss='sparese_categorical_crossentropy')

model.fit(train_images, train_labels, epochs=5)

test_loss = model.evaluate(test_images, test_labels)

predictions = model.predict(test_images)


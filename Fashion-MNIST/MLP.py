import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import keras
from get_data import *



class MLP:
     
    def __init__(self, epochs=10, batch_size=32, output_neurons=10):
        self.epochs = epochs
        self.batch_size = batch_size
        self.output_neurons = output_neurons
    


    def build(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(self.output_neurons, activation="softmax"),
        ])
        self.model.compile(optimizer="adam",
                           loss="categorical_crossentropy",
                           metrics=['accuracy'])
    

    def fit(self, x_train, y_train):
        self.model.fit(x_train, y_train, epochs=self.epochs, batch_size=self.batch_size)

    
    
    def predict(self, obs):
        obs = obs.reshape(1, 28, 28, 1)
        prediction = self.model.predict(obs)
        predicted_value = np.argmax(prediction)
        return predicted_value


    def print_atributtes(self):
        print(f"Epochs: {self.epochs}, Batch_size: {self.batch_size}, Output_neurons: {self.output_neurons}")




x_train, y_train, x_test, y_test, classes = prepare_data()
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

mlp = MLP()
mlp.build()
mlp.fit(x_train, y_train)

test_loss, test_acc = mlp.model.evaluate(x_test, y_test)
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)


import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import keras
from get_data import *




class MLP:
     
    def __init__(self, epochs=1, batch_size=32, output_neurons=2):
        self.epochs = epochs
        self.batch_size = batch_size
        self.output_neurons = output_neurons
    

    def prepare_data(self, training_set):    
        X = np.array([i[1] for i in training_set], dtype='float32').reshape(-1, len(training_set[0][1]))
        y = np.array([i[0] for i in training_set])
        return X, y


    def build(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(self.output_neurons, activation="softmax"),
        ])
        self.model.compile(optimizer="adam",
                           loss="categorical_crossentropy",
                           metrics=['accuracy'])
    

    def fit(self, training_set):
        X, y = self.prepare_data(training_set)
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size)

    

    def predict(self, obs):
         
         obs = np.array(obs).reshape(1, -1)

         prediction = self.model.predict(obs)

         predicted_value = np.argmax(prediction)

         return predicted_value
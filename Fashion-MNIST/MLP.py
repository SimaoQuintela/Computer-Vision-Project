import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import keras
from get_data import *



class MLP:
     
    def __init__(self, epochs=10, batch_size=32, output_neurons=10, data_aug=False):
        self.epochs = epochs
        self.batch_size = batch_size
        self.output_neurons = output_neurons
        self.data_aug = data_aug
        if self.data_aug:
            self.datagen = ImageDataGenerator(
                rotation_range = 90, #randomly rotate images in the range (degrees, 0 to 180)
                zoom_range = 0., #set range for random zoom
                horizontal_flip = False, #randomly horizontally flip images
                vertical_flip = True, #randomly vertically flip images
                rescale = None, #rescaling factor (applied before any other transf)
                preprocessing_function = None #function applied on each input
            )
    


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
        if self.data_aug:
            self.model.fit_generator(self.datagen.flow(x_train, y_train, batch_size=self.batch_size), epochs=self.epochs, validation_data=(x_test, y_test))
        else:
            self.model.fit(x_train, y_train, epochs=self.epochs, batch_size=self.batch_size, validation_data=(x_test, y_test))

    
    
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


import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import keras
from get_data import *
from sklearn.model_selection import cross_val_score, KFold, ParameterGrid
import time
import csv



class MLP:
     
    def __init__(self, epochs=10, batch_size=32, output_neurons=10, data_aug=False, f_act="relu", lr=0.001):
        self.epochs = epochs
        self.batch_size = batch_size
        self.output_neurons = output_neurons
        self.data_aug = data_aug
        self.f_act = f_act
        self.lr = lr
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
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)

        self.model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
            tf.keras.layers.Dense(64, activation=self.f_act),
            tf.keras.layers.Dense(128, activation=self.f_act),
            tf.keras.layers.Dense(self.output_neurons, activation="softmax"),
        ])
        self.model.compile(optimizer=optimizer,
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



def cross_validation(mlp, x_train, y_train, x_test, y_test, n_splits=5):
    x = np.concatenate([x_train, x_test])
    y = np.concatenate([y_train, y_test])

    
    kf = KFold(n_splits=n_splits, shuffle=True)

    
    scores = []

    
    for train_index, test_index in kf.split(x):
        x_train_fold, x_val_fold = x[train_index], x[test_index]
        y_train_fold, y_val_fold = y[train_index], y[test_index]


        mlp.fit(x_train_fold, y_train_fold)

        _, accuracy = mlp.model.evaluate(x_val_fold, y_val_fold)

        scores.append(accuracy)

    mean_score = np.mean(scores)
    print("Cross Validation Scores:", scores)
    print("Média Scores:", mean_score)
    return mean_score


def double_digit_sec(secs):
    if secs < 10:
        return f"0{secs}"
    return str(secs)


def tuning_and_csv_save(params, x_train, y_train, x_test, y_test):
    num_classes = 10

    results = []

    for param in ParameterGrid(params):

        start_time = time.time()
        mlp = MLP(epochs=param['epochs'], batch_size=param['batch_size'], data_aug=param['data_aug'], f_act=param['activation_function'], lr=param['learning_rate'])
        mlp.build()
        print("Training with parameters:", param)
        mean_score = cross_validation(mlp, x_train, y_train, x_test, y_test, n_splits=2)
        time_dif = time.time() - start_time
        param["mean_score"] = round(mean_score,3)
        param["time"] = f"{int(time_dif//60)}:" + double_digit_sec(int(time_dif - ((time_dif//60)*60)))
        results.append(param)
    
    with open("Tuning_datasets/results_tuning_MLP.csv", mode='w', newline='') as p:
        writer = csv.DictWriter(p, fieldnames=list(params.keys())+["mean_score", "time"])
        writer.writeheader()
        results = sorted(results, key = lambda dic: -dic['mean_score'])
        for row in results:
            writer.writerow(row)


params = {
    'epochs': [1],
    'batch_size': [128],
    'data_aug': [False, True],
    'activation_function': ['relu'],
    'learning_rate': [0.001, 0.0001]
}

x_train, y_train, x_test, y_test, classes = prepare_data()
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)


tuning_and_csv_save(params, x_train, y_train, x_test, y_test)
#mlp = MLP()
#mlp.build()
#mlp.fit(x_train, y_train)


#test_loss, test_acc = mlp.model.evaluate(x_test, y_test)
#print('Test loss:', test_loss)
#print('Test accuracy:', test_acc)

#cross_validation(mlp, x_train, y_train, x_test, y_test, n_splits=2)





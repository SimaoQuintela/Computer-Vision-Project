import numpy as np
import tensorflow as tf
from get_data import *
from sklearn.model_selection import cross_val_score, KFold
import matplotlib.pyplot as plt
import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import cross_val_score, KFold, ParameterGrid
import time
import csv



def create_rnn(num_classes, f_act):

    
    model = keras.models.Sequential()
    model.add(keras.Input(shape=(28,28,1))) # seq_length, input_size
    model.add(tf.keras.layers.Reshape((28, 28))) 
    #model.add(layers.SimpleRNN(128, return_sequences=True, activation='relu')) # N, 28, 128
    model.add(tf.keras.layers.LSTM(128, return_sequences=False, activation=f_act)) # N, 128
    model.add(tf.keras.layers.Dense(num_classes))
    print(model.summary())
    return model




def compile_fit_cross_validation(model, x_train, y_train, x_test, y_test, batch_size, epochs, data_aug,lr, n_splits):

    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optim = tf.keras.optimizers.Adam(lr)
    metrics = ["accuracy"]

    model.compile(loss=loss, optimizer=optim, metrics=metrics)  

    x = np.concatenate([x_train, x_test])
    y = np.concatenate([y_train, y_test])

    kf = KFold(n_splits=n_splits, shuffle=True)

    scores = []

    for train_index, test_index in kf.split(x):
        x_train_fold, x_val_fold = x[train_index], x[test_index]
        y_train_fold, y_val_fold = y[train_index], y[test_index]

        if not data_aug:
            model.fit(x_train_fold, y_train_fold, batch_size=batch_size, epochs=epochs, verbose=2)
        else:
            datagen = ImageDataGenerator(
                rotation_range = 90, #randomly rotate images in the range (degrees, 0 to 180)
                zoom_range = 0., #set range for random zoom
                horizontal_flip = False, #randomly horizontally flip images
                vertical_flip = True, #randomly vertically flip images
                rescale = None, #rescaling factor (applied before any other transf)
                preprocessing_function = None #function applied on each input
            )

            datagen.fit(x_train_fold)
        
            model.fit(datagen.flow(x_train_fold, y_train_fold, batch_size=batch_size),
                    epochs=epochs, verbose=2)

        _, accuracy = model.evaluate(x_val_fold, y_val_fold, verbose=0)
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
        rnn_model = create_rnn(num_classes, param['activation_function'])
        print("Training with parameters:", param)
        mean_score=compile_fit_cross_validation(rnn_model, x_train, y_train, x_test, y_test, param['batch_size'], param['epochs'], param['data_aug'], param['learning_rate'], n_splits=5)
        time_dif = time.time() - start_time
        param["mean_score"] = round(mean_score,3)
        param["time"] = f"{int(time_dif//60)}:" + double_digit_sec(int(time_dif - ((time_dif//60)*60)))
        results.append(param)
    
    with open("Tuning_datasets/results_tuning_RNN.csv", mode='w', newline='') as p:
        writer = csv.DictWriter(p, fieldnames=list(params.keys())+["mean_score", "time"])
        writer.writeheader()
        results = sorted(results, key = lambda dic: -dic['mean_score'])
        for row in results:
            writer.writerow(row)





params = {
    'epochs': [5, 10],
    'batch_size': [64, 128],
    'data_aug': [False, True],
    'activation_function': ['relu', 'sigmoid'],
    'learning_rate': [0.001, 0.01]
}




#load dados
x_train, y_train, x_test, y_test, classes = prepare_data()

tuning_and_csv_save(params, x_train, y_train, x_test, y_test)



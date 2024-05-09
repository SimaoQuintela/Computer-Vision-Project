import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import keras
from get_data import *
from sklearn.model_selection import cross_val_score, KFold, ParameterGrid
import time


#CNN using the sequential API
def create_cnn(num_classes, f_act):
    model = tf.keras.Sequential()
    #microarchitecture
    model.add(tf.keras.layers.Conv2D(32, (3, 3), (1, 1), padding='same',
    activation=f_act, input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation=f_act))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    model.add(tf.keras.layers.Dropout(0.25))
    #microarchitecture
    model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation=f_act))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation=f_act))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    model.add(tf.keras.layers.Dropout(0.25))
    #bottleneck
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation=f_act))
    model.add(tf.keras.layers.Dropout(0.5))
    #output layer
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    #printing a summary of the model structure
    model.summary()
    return model



def compile_and_fit(model, x_train, y_train, x_test, y_test, batch_size, epochs, apply_data_augmentation, lr):
    #sparse_categorical_crossentropy so that we do not need to one hot encode labels
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
    loss=tf.keras.losses.sparse_categorical_crossentropy,
    metrics=['accuracy'])

    #fit with/without data augmentation
    if not apply_data_augmentation:
        print('No data augmentation')
        history = model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_test, y_test),
            shuffle=True)
    else:
        print('Using data augmentation')
        #preprocessing and realtime data augmentation with ImageDataGenerator
        datagen = ImageDataGenerator(
            rotation_range = 90, #randomly rotate images in the range (degrees, 0 to 180)
            zoom_range = 0., #set range for random zoom
            horizontal_flip = False, #randomly horizontally flip images
            vertical_flip = True, #randomly vertically flip images
            rescale = None, #rescaling factor (applied before any other transf)
            preprocessing_function = None #function applied on each input
            )
        
        #compute quantities required for feature-wise normalization
        datagen.fit(x_train)
        #fit the model on the batches generated by datagen.flow()
        history = model.fit_generator(datagen.flow(x_train, y_train, batch_size = batch_size),
        epochs = epochs,
        validation_data = (x_test, y_test),
        workers = 1)
    return model, history 



#Vizualizing Learning Curves
def plot_learning_curves(history, epochs):
    #accuracies and losses
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(epochs)
    #creating figure
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training/Validation Accuracy')
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training/Validation Loss')
    plt.show()



def cross_validation(model, x_train, y_train, x_test, y_test, batch_size, epochs, apply_data_augmentation, lr, n_splits=5):
    x = np.concatenate([x_train, x_test])
    y = np.concatenate([y_train, y_test])

    
    kf = KFold(n_splits=n_splits, shuffle=True)

    
    scores = []

    
    for train_index, test_index in kf.split(x):
        x_train_fold, x_val_fold = x[train_index], x[test_index]
        y_train_fold, y_val_fold = y[train_index], y[test_index]


        compile_and_fit(model, x_train_fold, y_train_fold, x_val_fold, y_val_fold, batch_size, epochs, apply_data_augmentation, lr)

        _, accuracy = model.evaluate(x_val_fold, y_val_fold)

        scores.append(accuracy)

    print("Cross Validation Scores:", scores)
    print("Média Scores:", np.mean(scores))



# Function for grid search
def grid_search_cnn(params, x_train, y_train, x_test, y_test):
    num_classes = 10

    results = []

    for param in ParameterGrid(params):
        cnn_model = create_cnn(num_classes, param['activation_function'])
        print("Training with parameters:", param)
        cross_validation(cnn_model, x_train, y_train, x_test, y_test, param['batch_size'], param['epochs'], param['data_aug'], param['learning_rate'], n_splits=5)
        results.append(param)
    return(results)


# Definindo os hiperparâmetros que queremos ajustar
params = {
    'epochs': [1, 2],
    'batch_size': [32, 64],
    'data_aug': [True, False],
    'activation_function': ['relu', 'sigmoid', 'linear', 'tanh', 'softplus'],
    'learning_rate': [0.1, 0.01, 0.001, 0.0001]
}


'''
num_classes = 10
batch_size = 128
epochs = 1
apply_data_augmentation = False
lr = 0.001
num_predictions = 20
'''
#load data
x_train, y_train, x_test, y_test, classes = prepare_data()
#create the model
#cnn_model = create_cnn(num_classes, 'relu')
#compile and fit model

grid_search_cnn(params, x_train, y_train, x_test, y_test)



'''
cnn_model, history = compile_and_fit(cnn_model, x_train, y_train, x_test, y_test,
 batch_size, epochs, apply_data_augmentation)
#Evaluate trained model
score = cnn_model.evaluate(x_test, y_test)
print('Evaluation Loss:', score[0])
print('Evaluation Accuracy:', score[1])

plot_learning_curves(history, epochs)
'''


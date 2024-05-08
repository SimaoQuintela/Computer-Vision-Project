import numpy as np
import tensorflow as tf
from get_data import *
from sklearn.model_selection import cross_val_score, KFold
from tensorflow.keras.preprocessing.image import ImageDataGenerator


#usar simpleRNN, GRU, LTSM ou Bidirectional
def create_rnn(num_classes):

    
    model = keras.models.Sequential()
    model.add(keras.Input(shape=(28,28,1))) # seq_length, input_size
    model.add(tf.keras.layers.Reshape((28, 28))) 
    #model.add(layers.SimpleRNN(128, return_sequences=True, activation='relu')) # N, 28, 128
    model.add(tf.keras.layers.LSTM(128, return_sequences=False, activation='relu')) # N, 128
    model.add(tf.keras.layers.Dense(num_classes))
    print(model.summary())
    return model


def compile(model):
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optim = tf.keras.optimizers.Adam(lr=0.001)
    metrics = ["accuracy"]

    model.compile(loss=loss, optimizer=optim, metrics=metrics)  
    return model


def cross_validation(model, x_train, y_train, x_test, y_test, batch_size, epochs, data_aug, n_splits=5):
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

    print("Cross Validation Scores:", scores)
    print("Média Scores:", np.mean(scores))



num_classes = 10
batch_size = 64
epochs = 2
data_aug = False

#ir buscar dados
x_train, y_train, x_test, y_test, classes = prepare_data()
#criar o modelo
rnnmodel=create_rnn(num_classes)
#compilar o modelo
compile(rnnmodel)

'''
#ajuste do modelo
rnnmodel.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=2)

# avaliação dos resultados
scores=rnnmodel.evaluate(x_test, y_test, batch_size=batch_size, verbose=2)
print("Accuracy: %.2f%%" % (scores[1]*100))
print("Loss: %.2f%%" %(scores[0]*100))
'''

cross_validation(rnnmodel, x_train, y_train, x_test, y_test, batch_size, epochs, data_aug, n_splits=5)

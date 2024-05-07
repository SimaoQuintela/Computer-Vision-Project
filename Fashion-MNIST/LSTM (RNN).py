import numpy as np
import tensorflow as tf
from get_data import *
from sklearn.model_selection import cross_val_score, KFold


#usar simpleRNN, GRU, LTSM ou Bidirectional
def create_rnn(num_classes):

    
    model = keras.models.Sequential()
    model.add(keras.Input(shape=(28,28))) # seq_length, input_size
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


def cross_validation(model, x_train, y_train, x_test, y_test, batch_size, epochs, n_splits=5):
    x = np.concatenate([x_train, x_test])
    y = np.concatenate([y_train, y_test])

    
    kf = KFold(n_splits=n_splits, shuffle=True)

    
    scores = []

    
    for train_index, test_index in kf.split(x):
        x_train_fold, x_val_fold = x[train_index], x[test_index]
        y_train_fold, y_val_fold = y[train_index], y[test_index]


        model.fit(x_train_fold, y_train_fold, batch_size=batch_size, epochs=epochs, verbose=2)

        _, accuracy = model.evaluate(x_val_fold, y_val_fold)

        scores.append(accuracy)

    print("Cross Validation Scores:", scores)
    print("Média Scores:", np.mean(scores))


num_classes = 10
batch_size = 64
epochs = 2

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

cross_validation(rnnmodel, x_train, y_train, x_test, y_test, batch_size, epochs, n_splits=5)

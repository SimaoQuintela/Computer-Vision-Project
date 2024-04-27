import tensorflow as tf
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense
#from tensorflow.keras.layers import LSTM
#from tensorflow.keras.layers import Embedding
#from tensorflow.keras.preprocessing import sequence
from get_data import *



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


num_classes = 10
batch_size = 64
epochs = 5

#ir buscar dados
x_train, y_train, x_test, y_test, classes = prepare_data()
#criar o modelo
rnnmodel=create_rnn(num_classes)
#compilar o modelo
compile(rnnmodel)
#ajuste do modelo
rnnmodel.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=2)

# avaliação dos resultados
scores=rnnmodel.evaluate(x_test, y_test, batch_size=batch_size, verbose=2)
print("Accuracy: %.2f%%" % (scores[1]*100))
print("Loss: %.2f%%" %(scores[0]*100))






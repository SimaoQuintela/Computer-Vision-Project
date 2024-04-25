import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os 
from pprint import pprint
import cv2

one_hot_encoding = {
    'Am' : [1, 0, 0, 0, 0, 0],
    'C'  : [0, 1, 0, 0, 0, 0],
    'D'  : [0, 0, 1, 0, 0, 0],
    'Em' : [0, 0, 0, 1, 0, 0],
    'F'  : [0, 0, 0, 0, 1, 0],
    'G'  : [0, 0, 0, 0, 0, 1],
}

def convert_image_to_vector(images):
    X = []
    for image in images:
        X.append(image.getdata())

    return np.array(X)

def load_data():        
    train = []
    train_labels = [] 
    test = []
    test_labels = []
    # Usar 4 imagens do dataset
    for chord in os.listdir('./dataset/'):
        images = os.listdir(f'./dataset/{chord}')
        
        train.append( cv2.imread(f'./dataset/{chord}/{images[0]}')  )
        train.append( cv2.imread(f'./dataset/{chord}/{images[5]}')  )
        train.append( cv2.imread(f'./dataset/{chord}/{images[-1]}') )
        test.append( cv2.imread(f'./dataset/{chord}/{images[-5]}') )
        
        train_labels.append(one_hot_encoding[chord])
        train_labels.append(one_hot_encoding[chord])
        train_labels.append(one_hot_encoding[chord])
        test_labels.append(one_hot_encoding[chord])

    
       
    
    #x_train = convert_image_to_vector(train)
    x_train = np.array(train)
    y_train = np.array(train_labels)
    
    #x_test = convert_image_to_vector(test)
    x_test = np.array(test)
    y_test = np.array(test_labels)
    
    return (x_train, y_train), (x_test, y_test)
    

def create_cnn(num_classes, input_shape):
    model = tf.keras.Sequential()
    
    # micro-architecture
    model.add(tf.keras.layers.Conv2D(32, (3,3), (1,1), padding='same', activation='relu', input_shape=input_shape ))
    model.add(tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D( pool_size=(2,2), strides=(2,2)) )
    model.add(tf.keras.layers.Dropout(0.25))
    
    # micro-architecture
    model.add(tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu' ))
    model.add(tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu' ))
    model.add(tf.keras.layers.MaxPooling2D( pool_size=(2,2), strides=(2,2)) )
    model.add(tf.keras.layers.Dropout(0.25))
    
    # bottleneck
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    
    # output layer
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    
    # printing a summary of the model structure
    #model.summary()
    
    return model

def compile_and_fit(model, x_train, y_train, x_test, y_test, batch_size, epochs):
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.categorical_crossentropy,
        metrics=['accuracy']
    )

    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test),
        shuffle=True
    )

    return model, history

# setup 
num_classes = 6
batch_size = 32
epochs = 10

(x_train, y_train), (x_test, y_test) = load_data()
print(x_train.shape)
print(x_train[0])
model = create_cnn(6, (720, 1280, 3))
cnn_model, history = compile_and_fit(model, x_train, y_train, x_test, y_test, batch_size, epochs)
score = cnn_model.evaluate(x_test, y_test)
print('Evaluation Loss: ', score[0])
print('Evaluation Accuracy: ', score[1])


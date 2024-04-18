import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()


print(f"dimensao dados treino:", x_train.shape)
print(f"dimensao dados teste:", x_test.shape)


print(np.unique(y_train, return_counts=True)) #há 6000 de cada peça de roupa nos dados de treino
print(np.unique(y_test, return_counts=True)) #1000 nos de teste

print(f"número de missing values nos dados de treino:", np.count_nonzero(np.isnan(x_train)))  #detetar missing values
print(f"número de missing values nos dados de teste:", np.count_nonzero(np.isnan(x_test)))



import tensorflow as tf
import matplotlib.pyplot as plt
import keras

print("TF Version:", tf.__version__)
print("Keras version:", keras.__version__)


#Loading training and the testing sets (numpy arrays)
def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    #we have 10 labels (0:T-shirt, 1:Trouser, …, 9:Ankle boot)
    #each image is mapped to one single label (class names not included)
    classes = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    return (x_train, y_train), (x_test, y_test), classes

# Analyzing the datasets (the prints on the right)
def analyze_data(x_train, y_train, x_test, y_test, classes):
    print(50*'*')
    print("Training set shape:", x_train.shape, "and testing set shape:", x_test.shape)
    print("Training labels shape:", y_train. shape, "and testing labels shape:", y_test.shape)



def visualize_data(x_train, y_train, classes):
    plt.figure(figsize=(10,10))
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(tf.squeeze(x_train[i]))
        plt.xlabel(classes[y_train[i]])
    plt.show()




def prepare_data():
    (x_train, y_train), (x_test, y_test), classes = load_data()
    analyze_data(x_train, y_train, x_test, y_test, classes)
    visualize_data(x_train, y_train, classes)
    #normalizing/scaling pixel values to [0, 1]
    x_train = x_train.astype('float32')/255.
    x_test = x_test.astype('float32')/255.
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    return x_train, y_train, x_test, y_test, classes
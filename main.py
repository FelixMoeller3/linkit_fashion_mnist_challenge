from tensorflow.keras.datasets import fashion_mnist
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#import matplotlib.pyplot as plt
#import tensorflow as tf


#print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = x_train.reshape(-1,28,28, 1)
x_test = x_test.reshape(-1,28,28, 1)

x_train = x_train/255
x_test = x_test/255

def build_model():
    model = keras.models.Sequential()
    model.add(Conv2D(160, kernel_size=3, input_shape=x_train.shape[1:], activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(96, kernel_size=3, activation='relu'))
    model.add(Conv2D(64, kernel_size=3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(10,activation='softmax'))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


model = build_model()
#model.summary()
model.fit(x_train, y_train, batch_size=64, epochs=5, validation_data=(x_test, y_test), verbose=1)


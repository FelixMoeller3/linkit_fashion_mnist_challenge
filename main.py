from tensorflow.keras.datasets import fashion_mnist
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#import matplotlib.pyplot as plt
#import tensorflow as tf

NUM_EPOCHS = 35
INIT_LR = 1e-2 * 0.5
BATCH_SIZE = 32

#gen = ImageDataGenerator(vertical_flip=True, horizontal_flip=True)

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = x_train.reshape(-1,28,28, 1)
x_test = x_test.reshape(-1,28,28, 1)

x_train = x_train/255
x_test = x_test/255

def build_model():
    model = keras.models.Sequential()

    model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=x_train.shape[1:], padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size=3, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, kernel_size=3, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=3, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))


    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))

    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer=SGD(lr=INIT_LR, momentum=0.9, decay=INIT_LR/NUM_EPOCHS),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


model = build_model()
#model.summary()
model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, validation_data=(x_test, y_test), verbose=1)


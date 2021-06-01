from tensorflow.keras.datasets import fashion_mnist
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
from train_data import load_dataset
#from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from random_eraser import get_random_eraser
#import matplotlib.pyplot as plt
#import tensorflow as tf

NUM_EPOCHS = 200
INIT_LR = 1e-3
BATCH_SIZE = 32
L2_PENALTY = 0.003

gen = ImageDataGenerator(horizontal_flip=True, zoom_range=0.1)

(x_train, y_train), (x_test, y_test) = load_dataset(3)

def build_model():
    model = keras.models.Sequential()

    model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=x_train.shape[1:], padding='same', kernel_regularizer=l2(L2_PENALTY)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(L2_PENALTY)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(L2_PENALTY)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(L2_PENALTY)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(L2_PENALTY)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(L2_PENALTY)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))


    model.add(Flatten())
    model.add(Dense(256, activation='relu', kernel_regularizer=l2(L2_PENALTY)))
    model.add(BatchNormalization())
    model.add(Dense(256, activation='relu', kernel_regularizer=l2(L2_PENALTY)))
    model.add(Dropout(0.25))

    model.add(Dense(256, activation='relu', kernel_regularizer=l2(L2_PENALTY)))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer=SGD(lr=INIT_LR, momentum=0.9, decay=INIT_LR/NUM_EPOCHS),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


model = build_model()
model.summary()
model.fit(gen.flow(x_train, y_train), batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, validation_data=(x_test, y_test), verbose=1)


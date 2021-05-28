from tensorflow.keras.datasets import fashion_mnist
import pandas as pd

def flip_image_vertical(image):
    flipped_image = []
    for row in image:
        flipped_image.append(row[::-1])
    return flipped_image

def flip_image_horizontal(image):
    flipped_image = np.transpose(image)
    return np.transpose(flip_image_vertical(flipped_image))

def rotate_270(image):
    return np.transpose(flip_image_vertical(image))

def rotate_90(image):
    return flip_image_vertical(np.transpose(image))

def load_dataset():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_dataset()

    for i in range(0,len(x_train)):
        x_train.append(flip_image_horizontal(x_train[i]))
        y_train.append(y_train[i])
        x_train.append(rotate_90(x_train[i]))
        y_train.append(y_train[i])
        x_train.append(rotate_270(x_train[i]))
        y_train.append(y_train[i])

    x_train = x_train / 255
    x_test = x_test / 255
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    return (x_train, y_train), (x_test, y_test)


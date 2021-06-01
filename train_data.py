from tensorflow.keras.datasets import fashion_mnist
import numpy as np
from random_eraser import get_random_eraser

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

def load_dataset(num_erases):
    eraser = get_random_eraser(p=1, v_l=0, v_h=1, pixel_level=True)
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    training_set_size= len(x_train)
    augmented_images = []
    labels = []
    for i in range(0,training_set_size):
        for j in range(0,num_erases):
            duplicated_image = np.array(x_train[i])
            augmented_images.append(eraser(duplicated_image))
            labels.append(y_train[i])
        augmented_images.append(flip_image_horizontal(x_train[i]))
        labels.append(y_train[i])
        augmented_images.append(rotate_90(x_train[i]))
        labels.append(y_train[i])
        augmented_images.append(rotate_270(x_train[i]))
        labels.append(y_train[i])

    np.append(x_train, augmented_images)
    np.append(y_train, labels)
    x_train = x_train / 255
    x_test = x_test / 255
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    return (x_train, y_train), (x_test, y_test)


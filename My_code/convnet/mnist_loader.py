"""This reads in the MNIST Dataset"""

import struct
from array import array
import numpy as np

paths = {
    "train_img": "input/train-images-idx3-ubyte",
    "train_lab": "input/train-labels-idx1-ubyte",
    "test_img": "input/t10k-images-idx3-ubyte",
    "test_lab": "input/t10k-labels-idx1-ubyte",
}


def load_mnist_data(images_filepath, labels_filepath):
    """Load MNIST images and labels from files"""
    # Read labels
    with open(labels_filepath, "rb") as file:
        magic, size = struct.unpack(">II", file.read(8))
        if magic != 2049:
            raise ValueError(f"Magic number mismatch, expected 2049, got {magic}")
        labels = array("B", file.read())

    # Read images
    with open(images_filepath, "rb") as file:
        magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
        if magic != 2051:
            raise ValueError(f"Magic number mismatch, expected 2051, got {magic}")
        image_data = array("B", file.read())

    # Convert to appropriate format
    images = np.array(image_data, dtype=np.float32).reshape(size, rows * cols) / 255.0
    labels = np.array(labels, dtype=np.int32)

    return images, labels


def load_data_wrapper():
    """Load and prepare MNIST data for neural network"""
    # Load raw data
    x_train, y_train = load_mnist_data(paths["train_img"], paths["train_lab"])
    x_test, y_test = load_mnist_data(paths["test_img"], paths["test_lab"])

    # Reshape inputs to column vectors (784,1)
    x_train = x_train.reshape(-1, 784, 1)
    x_test = x_test.reshape(-1, 784, 1)

    # One-hot encode training labels
    y_train_encoded = np.zeros((y_train.size, 10, 1))
    for i, label in enumerate(y_train):
        y_train_encoded[i, label, 0] = 1.0

    # Split test data into validation and test sets
    n_validation = 5000
    x_validation, x_test = x_test[:n_validation], x_test[n_validation:]
    y_validation, y_test = y_test[:n_validation], y_test[n_validation:]

    # Create final datasets
    training_data = list(zip(x_train, y_train_encoded))
    validation_data = list(zip(x_validation, y_validation))
    test_data = list(zip(x_test, y_test))

    return training_data, validation_data, test_data

# mnist_loader.py

"""This reads in the MNIST Dataset"""

import struct
from array import array

import numpy as np

# dict of file paths
paths = {
    "train_img": "input/train-images-idx3-ubyte",
    "train_lab": "input/train-labels-idx1-ubyte",
    "test_img": "input/t10k-images-idx3-ubyte",
    "test_lab": "input/t10k-labels-idx1-ubyte",
}


# MNIST Data Loader Class


class MnistDataloader(object):
    def __init__(
        self,
        training_images_filepath,
        training_labels_filepath,
        test_images_filepath,
        test_labels_filepath,
    ):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath

    def read_images_labels(self, images_filepath, labels_filepath):
        labels = []
        with open(labels_filepath, "rb") as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError(
                    "Magic number mismatch, expected 2049, got {}".format(magic)
                )
            labels = array("B", file.read())

        with open(images_filepath, "rb") as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError(
                    "Magic number mismatch, expected 2051, got {}".format(magic)
                )
            image_data = array("B", file.read())
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols : (i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img

        return images, labels

    def load_data(self):
        x_train, y_train = self.read_images_labels(
            self.training_images_filepath, self.training_labels_filepath
        )
        x_test, y_test = self.read_images_labels(
            self.test_images_filepath, self.test_labels_filepath
        )
        return (x_train, y_train), (x_test, y_test)


def to_categorical(y, num_classes=10):
    """Convert class vector to binary class matrix (one-hot encoding)"""
    y = np.array(y, dtype="int")
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    return categorical


def finish_up_data():
    mnist_dataloader = MnistDataloader(
        paths["train_img"],
        paths["train_lab"],
        paths["test_img"],
        paths["test_lab"],
    )

    # Load MNIST data
    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

    # Convert to numpy arrays first
    x_train = np.array(x_train, dtype=np.float32)
    x_test = np.array(x_test, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int32)
    y_test = np.array(y_test, dtype=np.int32)

    # Preprocess the training data
    # Reshape to (num_samples, 1, 28*28) and normalize to range [0, 1]
    x_train = x_train.reshape(x_train.shape[0], 1, 28 * 28).astype("float32") / 255
    # Convert ONLY training labels to one-hot encoding
    y_train = to_categorical(y_train)

    # Preprocess the test data
    x_test = x_test.reshape(x_test.shape[0], 1, 28 * 28).astype("float32") / 255
    # Keep test labels as scalars (no one-hot encoding)

    return x_train, y_train, x_test, y_test


def load_data_wrapper():
    # Load preprocessed data
    x_train, y_train, x_test, y_test = finish_up_data()

    # Check if the data is loaded correctly
    if x_train.size == 0 or x_test.size == 0:
        raise ValueError("Failed to load MNIST data")

    # Reshape x_train from (samples, 1, 784) to (samples, 784, 1)
    x_train = x_train.reshape(x_train.shape[0], 784, 1)
    x_test = x_test.reshape(x_test.shape[0], 784, 1)

    # y_train is one-hot encoded but in wrong shape (samples, 10)
    # Convert to list of (10, 1) arrays
    y_train_reshaped = [y.reshape(10, 1) for y in y_train]

    # Split test data into validation and test sets
    n_validation = 5000
    x_validation, x_test = x_test[:n_validation], x_test[n_validation:]
    y_validation, y_test = y_test[:n_validation], y_test[n_validation:]

    # Test data is already scalar - no reshaping needed

    # Create the data tuples
    training_data = list(zip(x_train, y_train_reshaped))
    validation_data = list(zip(x_validation, y_validation))
    test_data = list(zip(x_test, y_test))

    return (training_data, validation_data, test_data)

# mnist_loader.py

from tensorflow import keras
import numpy as np

def load_data_wrapper():
    try:
        # Load the MNIST dataset
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

        # Check if the data is loaded correctly
        if x_train.size == 0 or x_test.size == 0:
            raise ValueError("Failed to load MNIST data")

        # Normalize and reshape the data
        x_train = x_train.reshape(x_train.shape[0], 784, 1).astype('float32') / 255
        x_test = x_test.reshape(x_test.shape[0], 784, 1).astype('float32') / 255

        # Prepare the labels
        def vectorized_result(j):
            e = np.zeros((10, 1))
            e[j] = 1.0
            return e

        y_train = [vectorized_result(y) for y in y_train]
        
        # Split test data into validation and test sets
        n_validation = 5000
        x_validation, x_test = x_test[:n_validation], x_test[n_validation:]
        y_validation, y_test = y_test[:n_validation], y_test[n_validation:]

        # Create the data tuples
        training_data = list(zip(x_train, y_train))
        validation_data = list(zip(x_validation, y_validation))
        test_data = list(zip(x_test, y_test))

        return (training_data, validation_data, test_data)

    except Exception as e:
        print(f"Error in load_data_wrapper: {e}")
        print(f"x_train shape: {x_train.shape if 'x_train' in locals() else 'Not loaded'}")
        print(f"x_test shape: {x_test.shape if 'x_test' in locals() else 'Not loaded'}")
        raise

# Add this at the end of the file to test the loader
if __name__ == "__main__":
    try:
        training_data, validation_data, test_data = load_data_wrapper()
        print("Data loaded successfully")
        print(f"Training data size: {len(training_data)}")
        print(f"Validation data size: {len(validation_data)}")
        print(f"Test data size: {len(test_data)}")
    except Exception as e:
        print(f"Failed to load data: {e}")



print('end of loader')
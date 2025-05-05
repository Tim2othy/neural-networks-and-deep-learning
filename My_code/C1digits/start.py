# starting_ch1.py
from mnist_loader import load_data_wrapper
import network

# Get data directly from the wrapper
training_data, validation_data, test_data = load_data_wrapper()

net = network.Network([784, 32, 10])

net.SGD(training_data, 5, 8, 3.0, test_data=test_data)

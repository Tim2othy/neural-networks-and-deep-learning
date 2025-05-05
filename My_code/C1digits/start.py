"""Start the MNIST digit recognition network from chapter 1"""

from mnist_loader import load_data_wrapper
from network import Network

training_data, validation_data, test_data = load_data_wrapper()

net = Network([784, 32, 10])

net.SGD(training_data, 5, 8, 3.0, test_data=test_data)

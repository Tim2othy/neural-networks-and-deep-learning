# starting_ch1.py

import mnist_loader
import network

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

net = network.Network([784, 32, 10])

net.SGD(
    training_data,
    1,
    8,
    3.0,
    test_data=test_data
)

print("end of start")

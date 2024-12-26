# starting_ch1.py

import mnist_loader
import network


training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

net = network.Network([784, 46, 24, 10])

net.SGD(training_data, 5, 8, 3.0, test_data=test_data)


print("end of start")

# starting_ch1.py

import mnist_loader2
import network2
import network

training_data, validation_data, test_data = mnist_loader2.load_data_wrapper()

net = network.Network([784, 12, 10])
# net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)

# net.large_weight_initializer()

net.SGD(training_data, 1, 8, 3.0, test_data=test_data)


print("end of start")

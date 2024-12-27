# starting_ch1.py

import mnist_loader2
import network2

training_data, validation_data, test_data = mnist_loader2.load_data_wrapper()

net = network2.Network([784, 32, 10], cost=network2.CrossEntropyCost)

net.large_weight_initializer()

net.SGD(
    training_data,
    1,
    8,
    0.5,
    evaluation_data=test_data,
    monitor_evaluation_accuracy=True,
)

print("end of start2")

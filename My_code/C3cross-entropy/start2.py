# starting_ch2.py

from mnist_loader import load_data_wrapper
from network2 import CrossEntropyCost, Network

training_data, validation_data, test_data = load_data_wrapper()

net = Network([784, 32, 10], cost=CrossEntropyCost)

net.large_weight_initializer()

net.SGD(
    training_data,
    3,
    8,
    0.5,
    evaluation_data=test_data,
    monitor_evaluation_accuracy=True,
)

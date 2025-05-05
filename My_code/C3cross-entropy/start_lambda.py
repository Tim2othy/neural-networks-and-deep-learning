# starting_ch3 using regularization.py

from mnist_loader import load_data_wrapper
from network2 import CrossEntropyCost, Network

training_data, validation_data, test_data = load_data_wrapper()

net = Network([784, 32, 10], cost=CrossEntropyCost)

net.large_weight_initializer()

net.SGD(
    training_data,
    7,
    8,
    0.5,
    evaluation_data=test_data,
    lmbda=0.1,
    monitor_evaluation_cost=True,
    monitor_evaluation_accuracy=True,
    monitor_training_cost=True,
    monitor_training_accuracy=True,
)

from network3 import ConvPoolLayer, FullyConnectedLayer, Network, SoftmaxLayer

from mnist_loader import load_data_wrapper

training_data, validation_data, test_data = load_data_wrapper()

mini_batch_size = 10


net = Network(
    [
        ConvPoolLayer(
            image_shape=(mini_batch_size, 1, 28, 28),
            filter_shape=(20, 1, 5, 5),
            poolsize=(2, 2),
        ),
        FullyConnectedLayer(n_in=20 * 12 * 12, n_out=100),
        SoftmaxLayer(n_in=100, n_out=10),
    ],
    mini_batch_size,
)
net.SGD(training_data, 60, mini_batch_size, 0.1, validation_data, test_data)

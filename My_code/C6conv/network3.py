"""
Got the code from https://github.com/MichalDanielDobrzanski/DeepLearningPython/pull/14/
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

"""network3.py
~~~~~~~~~~~~~~
A PyTorch-based program for training and running simple neural
networks.
Supports several layer types (fully connected, convolutional, max
pooling, softmax), and activation functions (sigmoid, tanh, and
rectified linear units, with more easily added).
When run on a CPU, this program is much faster than network.py and
network2.py.  However, unlike network.py and network2.py it can also
be run on a GPU, which makes it faster still.
Because the code is based on torch, the code is different in many
ways from network.py and network2.py.  However, where possible I have
tried to maintain consistency with the earlier programs.  In
particular, the API is similar to network2.py.  Note that I have
focused on making the code simple, easily readable, and easily
modifiable.  It is not optimized, and omits many desirable features.
This program incorporates ideas from the torch documentation on
convolutional neural nets (notably,
http://deeplearning.net/tutorial/lenet.html ), from Misha Denil's
implementation of dropout (https://github.com/mdenil/dropout ), and
from Chris Olah (http://colah.github.io ).
"""

#### Constants


def linear(z):
    return z


def ReLU(z):
    return max(0.0, z)


#### Main class used to construct and train networks
class Network(object):

    def __init__(self, layers, mini_batch_size):
        """Takes a list of `layers`, describing the network architecture, and
        a value for the `mini_batch_size` to be used during training
        by stochastic gradient descent.
        """
        self.layers = layers
        self.mini_batch_size = mini_batch_size
        self.params = [param for layer in self.layers for param in layer.params]
        self.x = torch.empty((mini_batch_size, layers[0].n_in), dtype=torch.float32)
        self.y = torch.empty(mini_batch_size, dtype=torch.long)
        init_layer = self.layers[0]
        init_layer.set_inpt(self.x, self.x, self.mini_batch_size)
        for j in range(1, len(self.layers)):
            prev_layer, layer = self.layers[j - 1], self.layers[j]
            layer.set_inpt(
                prev_layer.output, prev_layer.output_dropout, self.mini_batch_size
            )
        self.output = self.layers[-1].output
        self.output_dropout = self.layers[-1].output_dropout

    def forward(self, x):
        """Forward pass through the network"""
        for layer in self.layers:
            x = layer(x)
        return x

    def SGD(
        self,
        training_data,
        epochs,
        mini_batch_size,
        eta,
        validation_data,
        test_data,
        lmbda=0.0,
    ):
        """Train the network using mini-batch stochastic gradient descent."""
        training_x, training_y = training_data
        validation_x, validation_y = validation_data
        test_x, test_y = test_data

        # compute number of minibatches for training, validation and testing
        num_training_batches = int(size(training_data) / mini_batch_size)
        num_validation_batches = int(size(validation_data) / mini_batch_size)
        num_test_batches = int(size(test_data) / mini_batch_size)

        # define the (regularized) cost function, symbolic gradients, and updates
        l2_norm_squared = sum([(layer.w**2).sum() for layer in self.layers])
        cost = (
            self.layers[-1].cost(self)
            + 0.5 * lmbda * l2_norm_squared / num_training_batches
        )
        grads = T.grad(cost, self.params)
        updates = [
            (param, param - eta * grad) for param, grad in zip(self.params, grads)
        ]

        # define functions to train a mini-batch, and to compute the
        # accuracy in validation and test mini-batches.
        i = T.lscalar()  # mini-batch index
        train_mb = torch.function(
            [i],
            cost,
            updates=updates,
            givens={
                self.x: training_x[
                    i * self.mini_batch_size : (i + 1) * self.mini_batch_size
                ],
                self.y: training_y[
                    i * self.mini_batch_size : (i + 1) * self.mini_batch_size
                ],
            },
        )
        validate_mb_accuracy = torch.function(
            [i],
            self.layers[-1].accuracy(self.y),
            givens={
                self.x: validation_x[
                    i * self.mini_batch_size : (i + 1) * self.mini_batch_size
                ],
                self.y: validation_y[
                    i * self.mini_batch_size : (i + 1) * self.mini_batch_size
                ],
            },
        )
        test_mb_accuracy = torch.function(
            [i],
            self.layers[-1].accuracy(self.y),
            givens={
                self.x: test_x[
                    i * self.mini_batch_size : (i + 1) * self.mini_batch_size
                ],
                self.y: test_y[
                    i * self.mini_batch_size : (i + 1) * self.mini_batch_size
                ],
            },
        )
        self.test_mb_predictions = torch.function(
            [i],
            self.layers[-1].y_out,
            givens={
                self.x: test_x[
                    i * self.mini_batch_size : (i + 1) * self.mini_batch_size
                ]
            },
        )
        # Do the actual training
        best_validation_accuracy = 0.0
        for epoch in range(epochs):
            for minibatch_index in range(num_training_batches):
                iteration = num_training_batches * epoch + minibatch_index
                if iteration % 1000 == 0:
                    print("Training mini-batch number {0}".format(iteration))
                cost_ij = train_mb(minibatch_index)
                if (iteration + 1) % num_training_batches == 0:
                    validation_accuracy = torch.mean(
                        [validate_mb_accuracy(j) for j in range(num_validation_batches)]
                    )
                    print(
                        f"Epoch {epoch}: validation accuracy {validation_accuracy:.2%}"
                    )

                    if validation_accuracy > best_validation_accuracy:
                        print("This is the best validation accuracy to date.")
                        best_validation_accuracy = validation_accuracy
                        best_iteration = iteration

                        # Test accuracy
                        correct = 0
                        total = 0

                        with torch.no_grad():
                            for test_data, test_target in test_loader:
                                test_output = self(test_data)
                                _, predicted = torch.max(test_output.data, 1)
                                total += test_target.size(0)
                                correct += (predicted == test_target).sum().item()

                        test_accuracy = correct / total
                        print(f"The corresponding test accuracy is {test_accuracy:.2%}")

        print("Finished training network.")
        print(
            f"Best validation accuracy of {best_validation_accuracy:.2%} obtained at iteration {best_iteration}"
        )
        print("Corresponding test accuracy of {0:.2%}".format(test_accuracy))


#### Define layer types


class ConvPoolLayer(object):
    """Used to create a combination of a convolutional and a max-pooling
    layer.  A more sophisticated implementation would separate the
    two, but for our purposes we'll always use them together, and it
    simplifies the code, so it makes sense to combine them.
    """

    def __init__(
        self, filter_shape, image_shape, poolsize=(2, 2), activation_fn=F.sigmoid
    ):
        """`filter_shape` is a tuple of length 4, whose entries are the number
        of filters, the number of input feature maps, the filter height, and the
        filter width.
        `image_shape` is a tuple of length 4, whose entries are the
        mini-batch size, the number of input feature maps, the image
        height, and the image width.
        `poolsize` is a tuple of length 2, whose entries are the y and
        x pooling sizes.
        """
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.poolsize = poolsize
        self.activation_fn = activation_fn
        # initialize weights and biases
        n_out = filter_shape[0] * np.prod(filter_shape[2:]) / np.prod(poolsize)
        self.w = torch.randn(filter_shape)
        self.b = (
            np.asarray(
                np.random.normal(loc=0, scale=1.0, size=(filter_shape[0],)),
                dtype=torch.config.floatX,
            ),
            borrow=True,
        )
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape(self.image_shape)
        conv_out = conv.conv2d(
            input=self.inpt,
            filters=self.w,
            filter_shape=self.filter_shape,
            image_shape=self.image_shape,
        )
        pooled_out = pool_2d(input=conv_out, ws=self.poolsize, ignore_border=True)
        self.output = self.activation_fn(
            pooled_out + self.b.dimshuffle("x", 0, "x", "x")
        )
        self.output_dropout = self.output  # no dropout in the convolutional layers


class FullyConnectedLayer(nn.Module):
    """Standard fully connected layer with optional dropout"""

    def __init__(self, n_in, n_out, activation_fn=ReLU, p_dropout=0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.activation_fn = activation_fn
        self.p_dropout = p_dropout
        # Initialize weights and biases
        self.w = (
            np.asarray(
                np.random.normal(
                    loc=0.0, scale=np.sqrt(1.0 / n_out), size=(n_in, n_out)
                ),
                dtype=torch.config.floatX,
            ),
            name="w",
            borrow=True,
        )
        self.b = (
            np.asarray(
                np.random.normal(loc=0.0, scale=1.0, size=(n_out,)),
                dtype=torch.config.floatX,
            ),
            name="b",
            borrow=True,
        )
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = self.activation_fn(
            (1 - self.p_dropout) * T.dot(self.inpt, self.w) + self.b
        )
        self.y_out = T.argmax(self.output, axis=1)
        self.inpt_dropout = dropout_layer(
            inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout
        )
        self.output_dropout = self.activation_fn(
            T.dot(self.inpt_dropout, self.w) + self.b
        )

    def accuracy(self, y):
        "Return the accuracy for the mini-batch."
        return T.mean(T.eq(y, self.y_out))


class SoftmaxLayer(object):

    def __init__(self, n_in, n_out, p_dropout=0.0):
        super(SoftmaxLayer, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.p_dropout = p_dropout
        # Initialize weights and biases
        self.w = (
            np.zeros((n_in, n_out), dtype=torch.config.floatX), name="w", borrow=True
        )
        self.b = (
            np.zeros((n_out,), dtype=torch.config.floatX), name="b", borrow=True
        )
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = softmax((1 - self.p_dropout) * T.dot(self.inpt, self.w) + self.b)
        self.y_out = T.argmax(self.output, axis=1)
        self.inpt_dropout = dropout_layer(
            inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout
        )
        self.output_dropout = softmax(T.dot(self.inpt_dropout, self.w) + self.b)

    def cost(self, net):
        "Return the log-likelihood cost."
        return -T.mean(T.log(self.output_dropout)[T.arange(net.y.shape[0]), net.y])

    def accuracy(self, y):
        "Return the accuracy for the mini-batch."
        return T.mean(T.eq(y, self.y_out))


#### Miscellanea
def size(data):
    "Return the size of the dataset `data`."
    return data[0].get_value(borrow=True).shape[0]


def dropout_layer(layer, p_dropout):
    srng = shared_randomstreams.RandomStreams(np.random.RandomState(0).randint(999999))
    mask = srng.binomial(n=1, p=1 - p_dropout, size=layer.shape)
    return layer * T.cast(mask, torch.config.floatX)

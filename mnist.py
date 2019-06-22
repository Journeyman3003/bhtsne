#!/usr/bin/env python

# using the following files:
#
# MNIST/mnist2500_X.txt - space separated X values
# MNIST/mnist2500_labels.txt - space separated labels
#
# MNIST/mnist_train.csv - comma separated, all 60.000 observations
# MNIST/mnist_test.csv - comma separated, all 10.000 test observations, not used for now
#

import numpy as np
import os
from keras.datasets import mnist

DEFAULT_DIR = 'MNIST'


def load_mnist_data(all_data=False, dir_name=DEFAULT_DIR):
    if all_data:
        mnist = np.loadtxt(fname=os.path.join(dir_name, 'mnist_train.csv'), delimiter=',', skiprows=1)
        return mnist[:, 1:], mnist[:, 0]
    else:
        mnist = np.loadtxt(fname=os.path.join(dir_name, 'mnist2500_X.txt'))
        label = np.loadtxt(fname=os.path.join(dir_name, 'mnist2500_labels.txt'))
        return mnist, label


def load_mnist_keras():
    """
    loads mnist data from keras
    :return: 
    """
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # for now, we don't care for train test split

    mnist_X = np.vstack((x_train, x_test))
    mnist_labels = np.hstack((y_train, y_test))

    # convert to 1d representation
    return mnist_2d_to_1d(mnist_X, mnist_X.shape[0], mnist_X.shape[1], mnist_X.shape[2]), mnist_labels


def mnist_1d_to_2d(data, num_observations=70000, img_rows=28, img_cols=28):
    return data.reshape((num_observations, img_rows, img_cols))


def mnist_2d_to_1d(data, num_observations=70000, img_rows=28, img_cols=28):
    return data.reshape((num_observations, img_rows * img_cols))

if __name__ == '__main__':
    # x, label = load_mnist_data(True)

    load_mnist_keras()





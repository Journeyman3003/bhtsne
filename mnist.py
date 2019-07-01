#!/usr/bin/env python

# using the following files:
#
# MNIST/mnist2500_X.txt - space separated X values
# MNIST/mnist2500_labels.txt - space separated labels
#
# MNIST repository: http://deeplearning.net/data/mnist/mnist.pkl.gz
#

import numpy as np
import pickle
import os
import gzip
import urllib.request


DEFAULT_DIR = 'MNIST'


def load_mnist_data(all_data=False, dir_name=DEFAULT_DIR):
    if all_data:
        pickle_mnist = os.path.join(dir_name, 'mnist.pkl.gz')
        if not os.path.exists(pickle_mnist):
            print('downloading MNIST')
            urllib.request.urlretrieve('http://deeplearning.net/data/mnist/mnist.pkl.gz', pickle_mnist)
            print('downloaded')

        with gzip.open(pickle_mnist, "rb") as mnist_unzip:
            train, val, test = pickle.load(mnist_unzip, encoding='latin1')

        # Get all data in one array
        _train = np.asarray(train[0], dtype=np.float64)
        _val = np.asarray(val[0], dtype=np.float64)
        _test = np.asarray(test[0], dtype=np.float64)
        mnist = np.vstack((_train, _val, _test))

        # Also the classes, for labels in the plot later
        classes = np.hstack((train[1], val[1], test[1]))

        return mnist, classes

    else:
        mnist = np.loadtxt(fname=os.path.join(dir_name, 'mnist2500_X.txt'))
        label = np.loadtxt(fname=os.path.join(dir_name, 'mnist2500_labels.txt'))
        return mnist, label


def mnist_1d_to_2d(data, num_observations=70000, img_rows=28, img_cols=28):
    return data.reshape((num_observations, img_rows, img_cols))


def mnist_2d_to_1d(data, num_observations=70000, img_rows=28, img_cols=28):
    return data.reshape((num_observations, img_rows * img_cols))


if __name__ == '__main__':
    x, label = load_mnist_data(True)





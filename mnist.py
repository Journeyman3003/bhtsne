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
import pickle
import os
import gzip
import urllib.request


DEFAULT_DIR = 'MNIST'


def load_mnist_data(all_data=False, dir_name=DEFAULT_DIR):
    os.chdir(os.path.join(os.path.dirname(os.path.realpath(__file__)), DEFAULT_DIR))
    if all_data:
        if not os.path.exists(os.path.join(DEFAULT_DIR, 'mnist.pkl.gz')):
            print('downloading MNIST')
            urllib.request.urlretrieve('http://deeplearning.net/data/mnist/mnist.pkl.gz', 'mnist.pkl.gz')
            print('downloaded')

        with gzip.open("mnist.pkl.gz", "rb") as zip:
            train, val, test = pickle.load(zip, encoding='latin1')

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





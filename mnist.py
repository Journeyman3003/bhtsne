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

DEFAULT_DIR = 'MNIST'


def load_mnist_data(all_data=False, dir_name=DEFAULT_DIR):
    if all_data:
        mnist = np.loadtxt(fname=os.path.join(dir_name, 'mnist_train.csv'), delimiter=',', skiprows=1)
        return mnist[:, 1:], mnist[:, 0]
    else:
        mnist = np.loadtxt(fname=os.path.join(dir_name, 'mnist2500_X.txt'))
        label = np.loadtxt(fname=os.path.join(dir_name, 'mnist2500_labels.txt'))
        return mnist, label


if __name__ == '__main__':
    x, label = load_mnist_data()





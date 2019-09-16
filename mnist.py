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


def sample_data(data, labels, num_samples=2500):

    # get unique labels
    unique_labels = np.unique(labels)
    num_samples_per_class = int(num_samples/len(unique_labels))
    filter_index = []

    for l in unique_labels:
        indices = np.where(labels == l)
        filter_index.extend(indices[0][:num_samples_per_class])

    # filter for found indices
    _sample_data = data[filter_index]
    _sample_labels = labels[filter_index]

    return _sample_data, _sample_labels


def load_mnist_data(all_data=True, dir_name=DEFAULT_DIR):
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


def load_fashion_mnist_data(all_data=True, train_test_split=0, len_sample=2500, dir_name=DEFAULT_DIR):
    # init empty numpy arrays

    fashion_mnist_images = np.empty((0, 784))
    fashion_mnist_labels = np.empty((0, ))

    for data in ['train', 't10k']:

        image_file = '{}-images-idx3-ubyte.gz'.format(data)
        label_file = '{}-labels-idx1-ubyte.gz'.format(data)
        img_ubyte_path = os.path.join(dir_name, image_file)
        lb_ubyte_path = os.path.join(dir_name, label_file)
        if not os.path.exists(img_ubyte_path):
            print('downloading {}'.format(image_file))
            urllib.request.urlretrieve('http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/' + image_file,
                                       img_ubyte_path)
            print('downloaded {}'.format(image_file))

        if not os.path.exists(lb_ubyte_path):
            print('downloading {}'.format(label_file))
            urllib.request.urlretrieve('http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/' + label_file,
                                       lb_ubyte_path)
            print('downloaded {}'.format(label_file))

        with gzip.open(lb_ubyte_path, 'rb') as lbpath:
            labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                                   offset=8)

        with gzip.open(img_ubyte_path, 'rb') as imgpath:
            images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                                   offset=16).reshape(len(labels), 784)

        fashion_mnist_images = np.vstack((fashion_mnist_images, images))
        fashion_mnist_labels = np.hstack((fashion_mnist_labels, labels))

    if all_data:
        if train_test_split > 0:
            return (fashion_mnist_images[:60000], fashion_mnist_images[60000:]),\
                   (fashion_mnist_labels[:60000], fashion_mnist_labels[60000:])
        else:
            return fashion_mnist_images, fashion_mnist_labels
    else:
        if train_test_split > 0:
            sample_images, sample_labels = sample_data(fashion_mnist_images, fashion_mnist_labels, len_sample)
            train_index = np.array(range(int(train_test_split/10)))
            test_index = np.array(range(int(train_test_split/10), int(len_sample/10)))
            train_index_all = []
            test_index_all = []
            for i in range(10):
                train_index_all.extend(train_index + i * int(len_sample/10))
                test_index_all.extend(test_index + i * int(len_sample/10))
            return (sample_images[train_index_all], sample_images[test_index_all]),\
                   (sample_labels[train_index_all], sample_labels[test_index_all])
        else:
            return sample_data(fashion_mnist_images, fashion_mnist_labels, len_sample)


def mnist_1d_to_2d(data, num_observations=70000, img_rows=28, img_cols=28):
    return data.reshape((num_observations, img_rows, img_cols))


def mnist_2d_to_1d(data, num_observations=70000, img_rows=28, img_cols=28):
    return data.reshape((num_observations, img_rows * img_cols))


if __name__ == '__main__':
    #x, label = load_mnist_data(True)
    fashion_x, fashion_label = load_fashion_mnist_data(all_data=False, len_sample=2500, train_test_split=2000)
    arr, counts = np.unique(fashion_label[1], return_counts=True)
    print(counts)




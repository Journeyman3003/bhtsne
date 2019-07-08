#!/usr/bin/env python

import numpy as np
import pickle
import os
from sklearn import decomposition, manifold, metrics
import mnist


# directory structure
CWD = os.path.dirname(os.path.realpath(__file__))
INIT = os.path.join(CWD, "initial_solutions")


SUPPORTED_METHODS = ['random', 'gaussian', 'pca', 'mds', 'spectral']
DEFAULT_SEED = 42


def _embedding(_data, method='gaussian', dist='euclidean', scaling_factor=.0001, **kwargs):
    """
    input: N x F dataframe of observations
    output: N x 2 image of input data under lens function
    """

    if method == 'random':
        latent = np.random.rand(_data.shape[0], 2)
    elif method == 'gaussian':
        latent = np.random.normal(0, 1, (_data.shape[0], 2))
    elif method == 'pca':
        latent = decomposition.PCA(n_components=2, **kwargs).fit_transform(_data)
    elif method == 'mds':
        _dist = metrics.pairwise.pairwise_distances(_data, metric=dist)
        latent = manifold.MDS(n_components=2, dissimilarity='precomputed', **kwargs).fit_transform(_dist)
    elif method == 'spectral':
        _dist = metrics.pairwise.pairwise_distances(_data, metric=dist)
        latent = manifold.SpectralEmbedding(n_components=2, **kwargs).fit_transform(_dist)
    else:
        print('no valid method specified, initializing at random (uniformly)')
        latent = np.random.rand(_data.shape[0], 2)
    return latent * scaling_factor


def get_initial_embedding(data_name, method_name, i=1):

    filename = "initial_solution_" + data_name + "_" + method_name  \
               + "_" + str(i) + ".pickle" if method_name in ['random', 'gaussian'] else ".pickle"

    # format to abspath
    file_abspath = os.path.join(INIT, filename)

    with open(file_abspath, 'rb') as pickle_file:
        return pickle.load(pickle_file)


def create_initial_solutions(data_name, data,  scaling_factor=.0001):
    """

    :param data:
    :param data_name:
    :param scaling_factor:
    :return:
    """

    # set seed first to obtain reproducible results
    np.random.seed(DEFAULT_SEED)

    # random and gaussian 5 times, other ones only once
    for method in ['mds']:
        if method in ['random', 'gaussian']:
            for i in range(1, 6):
                latent_embedding = _embedding(_data=data, method=method, scaling_factor=scaling_factor)
                filename = "initial_solution_" + data_name + "_" + method + "_" + str(i) + '.pickle'
                # format to abspath
                file_abspath = os.path.join(INIT, filename)

                with open(file_abspath, 'wb') as pickle_file:
                    pickle.dump(latent_embedding, pickle_file)
        else:
            latent_embedding = _embedding(_data=data, method=method, scaling_factor=scaling_factor)
            filename = "initial_solution_" + data_name + "_" + method + '.pickle'

            file_abspath = os.path.join(INIT, filename)

            with open(file_abspath, 'wb') as pickle_file:
                pickle.dump(latent_embedding, pickle_file)


if __name__ == '__main__':

    try:
        os.makedirs(INIT)
    except FileExistsError:
        # directory already exists
        pass

    # MNIST
    mnist_data, _ = mnist.load_mnist_data(all_data=True)
    create_initial_solutions("mnist", mnist_data)

    # MNIST2500
    #mnist2500_data, _ = mnist.load_mnist_data(all_data=False)
    #create_initial_solutions("mnist2500", mnist2500_data)



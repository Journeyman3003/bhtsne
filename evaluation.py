from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score
from sklearn.manifold.t_sne import trustworthiness
import numpy as np

from itertools import compress


def get_unsupervised_nearest_neighbors_excl_self(data, n=1):
    """
    custom version to avoid having the point itself to be its closest neighbor
    :param data:
    :param n:
    :return:
    """
    nN = NearestNeighbors(n_neighbors=n+1, algorithm='ball_tree').fit(data)
    distances, indices = nN.kneighbors(data)
    distances = np.array([x[1:] for x in distances])
    indices = np.array([x[1:] for x in indices])
    return distances, indices


def _most_frequent_element_in_list(_list):
    return max(set(list(_list)), key=list(_list).count)


def predict_labels(data, labels, n_neighbors=1):
    _ , _indices = get_unsupervised_nearest_neighbors_excl_self(data, n=n_neighbors)
    _predicted_labels = [labels[_most_frequent_element_in_list(_i)] for _i in _indices]
    return _predicted_labels


def compute_generalization_error(_labels, _predicted_labels):
    return accuracy_score(_labels, _predicted_labels)


def _list_to_rank(_list):
    ranks = [0] * len(_list)
    for _i, _x in enumerate(sorted(range(len(_list)), key=lambda y: _list[y])):
        ranks[_x] = _i + 1

    return ranks


if __name__ == '__main__':

    #X_high = np.array([[1,10,3,4], [1,1,30,5], [10,1,7,4], [1,1,9,40], [1,1, 110, 12], [1,1, 12, 14]])
    #X_low = np.array([[80, 4], [3, 5], [70, 4], [9, 4], [110, 12], [12, 140]])

    import mnist

    X_high, _ = mnist.load_mnist_data(True)
    X_low = np.random.rand(70000, 2)

    print(trustworthiness(X_high, X_low, n_neighbors=12))


    # X = np.array([[1,1], [2,1], [3,3], [4,4]])
    # label = [1, 2, 3, 4]
    #
    # d, i = get_unsupervised_nearest_neighbors_excl_self(X, 2)
    # print(d)
    # print(i)


    # predicted_labels = predict_labels(X, label)
    #
    # print(label)
    # print(predicted_labels)
    # print(compute_generalization_error(label, predicted_labels))

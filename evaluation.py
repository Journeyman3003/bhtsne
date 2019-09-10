from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score
from sklearn.manifold.t_sne import trustworthiness
from bhtsne import read_bh_tsne_result
import numpy as np
import json
import os
import glob
import mnist


# directory structure
CWD = os.path.dirname(os.path.realpath(__file__))
RESULT_DIR = os.path.join(CWD, 'results')


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


def compute_metrics(original_data, embedding_dict, labels):

    metric_dict = {'1NNgeneralization_error':{},
                   'trustworthiness':{},
                   'cost_function_value':{}}
    for key in embedding_dict.keys():

        # 1NN generalization error
        labels_predict = predict_labels(embedding_dict[key], labels)
        generalization_error = compute_generalization_error(labels, labels_predict)
        metric_dict['1NNgeneralization_error'][key[0]] = generalization_error

        if len(labels) < 20000:
            # trustworthiness (12)
            trustw = trustworthiness(original_data, embedding_dict[key][:, 0:2], n_neighbors=12)
            metric_dict['trustworthiness'][key[0]] = trustw

        # cost function value
        cost = sum(embedding_dict[key][:, 2])
        metric_dict['cost_function_value'][key[0]] = cost

    return metric_dict


def get_bh_tsne_filtered_results(root_dir=RESULT_DIR, data_identifier='fashion_mnist7000', algorithm='tSNE',
                                 task='parametertuning'):

    files = [f for f in glob.glob(os.path.join(root_dir, "**/*.pickle"), recursive=True)]

    # filter for paths that actually include the desired data
    print("filtering for data: {}".format(data_identifier))

    files = list(filter(lambda x: data_identifier in str(x).split(os.path.sep)
                        and algorithm in str(x).split(os.path.sep)
                        and task in str(x).split(os.path.sep), files))

    # sort list
    # essential for grouping
    files.sort()

    return files


def evaluate_bh_tsne_results(data, labels, root_dir=RESULT_DIR, data_identifier='fashion_mnist7000', algorithm='tSNE',
                             task='parametertuning'):

    result_list = get_bh_tsne_filtered_results(root_dir=root_dir, data_identifier=data_identifier, algorithm=algorithm,
                                               task=task)

    # temporary additional filter
    result_list = list(filter(lambda x: "exaggeration" in str(x).split(os.path.sep)
                                  and str(1) == str(x).split(os.path.sep)[-4]
                                  #or str(500) in str(x).split(os.path.sep)
                                  #or str(1000) in str(x).split(os.path.sep)
                                  #or str(2330) in str(x).split(os.path.sep)
                                       , result_list))

    for result in result_list:
        print("Computing metrics for result file at {}".format(result))
        filename = os.path.splitext(result)[0] + '-metrics.json'

        embedding_dict = read_bh_tsne_result(result)

        metric_dict = compute_metrics(data, embedding_dict, labels)

        with open(filename, 'w') as file:
            file.write(json.dumps(metric_dict, sort_keys=True))


if __name__ == '__main__':

    #data, labels = mnist.load_fashion_mnist_data(False, len_sample=7000)
    data, labels = mnist.load_fashion_mnist_data(True)

    evaluate_bh_tsne_results(data, labels, data_identifier='fashion_mnist', algorithm='BHtSNE', task='initial_embeddings')






    #X_high = np.array([[1,10,3,4], [1,1,30,5], [10,1,7,4], [1,1,9,40], [1,1, 110, 12], [1,1, 12, 14]])
    #X_low = np.array([[80, 4], [3, 5], [70, 4], [9, 4], [110, 12], [12, 140]])

    #import mnist

    #X_high, _ = mnist.load_mnist_data(True)
    #X_low = np.random.rand(70000, 2)

    #print(trustworthiness(X_high, X_low, n_neighbors=12))


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

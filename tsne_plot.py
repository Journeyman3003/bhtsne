#!/usr/bin/env python

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import os
import bhtsne
import mnist
import numpy as np
import glob
import itertools
import operator

# directory structure
CWD = os.path.dirname(os.path.realpath(__file__))
RESULT_DIR = os.path.join("I:", "MasterThesis", "Experimental Results")
TARGET = os.path.join("I:", "MasterThesis")
PLOT_DIR = os.path.join(TARGET, "Plots")


def get_bh_tsne_grouped_result_generator(root_dir=RESULT_DIR, data_identifier='mnist'):

    files = [f for f in glob.glob(os.path.join(root_dir, "**/*.pickle"), recursive=True)]

    # filter for paths that actually include the desired data
    print("filtering for data: {}".format(data_identifier))

    files = list(filter(lambda x: data_identifier in str(x).split(os.path.sep), files))

    # sort list
    # essential for grouping
    files.sort()

    files_tuples = [(str(x).split(os.path.sep)[-5] + str(x).split(os.path.sep)[-4], x) for x in files]
    files_tuples.sort()

    for _key, _grouper in itertools.groupby(files_tuples, operator.itemgetter(0)):
        yield _key, list(_grouper)


def plot_bh_tsne_result(_data, _labels, _legend="full", _palette="bright", _ax=None):
    #plt.box(False)
    sns.despine()
    sns.set_style("white")
    _g = sns.scatterplot(x=_data[:, 0],
                         y=_data[:, 1],
                         hue=_labels,
                         legend=_legend,
                         palette=sns.color_palette(_palette),
                         ax=_ax)

    return _g


def load_result_and_plot_comparison(_labels, root_dir=RESULT_DIR, data_identifier="mnist",
                                    plot_title_from_filepath_index=0):

    for _paramvalue, _file_list in get_bh_tsne_grouped_result_generator(root_dir=root_dir,
                                                                        data_identifier=data_identifier):
        print("Creating plot for data {} with parameter {}".format(data_identifier, _paramvalue))
        _result_list = [bhtsne.read_bh_tsne_result(_file) for _k, _file in _file_list]
        # make titles based on file path
        _title_list = [str(_file).split(os.path.sep)[plot_title_from_filepath_index]
                       if plot_title_from_filepath_index < 0 else ""
                       for _k, _file in _file_list]
        _paramvalue = _paramvalue.replace(".", "-")
        _dir = os.path.join(PLOT_DIR, _paramvalue, data_identifier)
        try:
            os.makedirs(_dir)
        except FileExistsError:
            # directory already exists
            pass

        # assuming that the keys of the first dict in list represent all dicts' keys
        for _key in _result_list[0].keys():
            print("Creating plot for iteration {}".format(_key[0]))
            _data_list = [result[_key] for result in _result_list]
            _fig = compare_n_results(_labels=_labels, _data_list=_data_list, _title_list=_title_list)

            save_figure(_fig, _dir, "-", _paramvalue, str(_key[0]))
            plt.close(_fig)




def compare_n_results(_labels, _data_list, _title_list, _size=8):
    """

    :param _size: default width and height of single plot
    :param _data_list: list of tuples of type (data, label)
    :return: the figure created
    """
    mpl.rcParams['xtick.labelsize'] = _size * 2
    mpl.rcParams['ytick.labelsize'] = _size * 2
    mpl.rcParams['axes.titlesize'] = _size * 3

    if len(_data_list) <= 5:
        _nrows = int((len(_data_list) + 1) / 2)
        _ncols = 2
    else:
        _nrows = int((len(_data_list) + 1) / 3)
        _ncols = 3

    fig_size = (_size * _ncols, _size * _nrows)

    _fig, _axs = plt.subplots(ncols=_ncols, nrows=_nrows, figsize=fig_size)
    #_fig, _axs = plt.subplots(len(_data_list), figsize=fig_size)

    for idx, _data in enumerate(_data_list):

        i = int(idx / _ncols)
        j = idx % _ncols

        plot_bh_tsne_result(_data, _labels, _legend="full", _ax=_axs[i, j])
        _axs[i, j].set_title("{} Cost: {}".format(_title_list[idx], np.sum(_data[:, 2])))
    # retrieve legend
    _handles, _labels = _axs[0, 0].get_legend_handles_labels()

    # remove legends of idividual subplots

    for idx, _data in enumerate(_data_list):
        i = int(idx / _ncols)
        j = idx % _ncols
        _axs[i, j].get_legend().remove()

    # plot legend if uneven number of plots into last axes
    if len(_data_list) % 2 == 1:
        _axs[_nrows - 1, _ncols - 1].legend(_handles, _labels, loc="center", prop={'size': _size * 4}, markerscale=3)
        _axs[_nrows - 1, _ncols - 1].axis("off")
    else:
        _fig.legend(_handles, _labels, loc="lower center", prop={'size': _size * 4}, markerscale=3)#, bbox_to_anchor=(1.14, 1))
    _fig.tight_layout()

    return _fig


def compare_two_results(_data1, _label1, _data2, _label2, figsize=(10, 5)):
    _fig, _axs = plt.subplots(ncols=2, gridspec_kw={'width_ratios': [1, 1]}, figsize=figsize)
    plot_bh_tsne_result(_data1, _label1, _legend="full", _ax=_axs[0])
    _axs[0].set_title("Cost: {}".format(np.sum(_data1[:, 2])))
    plot_bh_tsne_result(_data2, _label2, _legend="full", _ax=_axs[1])
    _axs[1].set_title("Cost: {}".format(np.sum(_data2[:, 2])))
    # retrieve legend
    _handles, _labels = _axs[0].get_legend_handles_labels()

    _axs[0].get_legend().remove()
    _axs[1].get_legend().remove()

    _fig.legend(_handles, _labels, loc="upper right", bbox_to_anchor=(1.14, 1))
    _fig.tight_layout()

    return _fig


def save_figure(figure, directory=PLOT_DIR, sep='-', *filename_extensions):
    figure_name = "tSNE-" + sep.join(filename_extensions)
    figure.savefig(os.path.join(directory, figure_name), bbox_inches="tight")


if __name__ == "__main__":
    #if argv < 2:
    #    print("Please specify the .pickle file to be read: tsne_plot.py <.pickle-file>!")

    _, labels = mnist.load_mnist_data(True)

    load_result_and_plot_comparison(_labels=labels, root_dir=os.path.join(RESULT_DIR, "buildingblocks"),
                                    plot_title_from_filepath_index=-6)

    # basepath1 = "C:\\Users\\Tobi\\Documents\\SS_19\\Master Thesis\\04 - Experiment Results\\MNIST\\base\\unoptimized sptree\\1"
    # basepath2 = "C:\\Users\\Tobi\\Documents\\SS_19\\Master Thesis\\04 - Experiment Results\\MNIST\\base\\optimized sptree\\1"
    # benchmark1 = bhtsne.read_bh_tsne_result(os.path.join(basepath1, "bh_tsne_result-08-07-2019_22-27-37.pickle"))
    # benchmark2 = bhtsne.read_bh_tsne_result(os.path.join(basepath2, "bh_tsne_result-09-07-2019_17-58-55.pickle"))
    #
    # for key in benchmark1.keys():
    #     print(str(key[0]))
    #     mnist_benchmark1 = np.hstack((labels[:, None], benchmark1[key]))
    #     mnist_benchmark2 = np.hstack((labels[:, None], benchmark2[key]))
    #
    #     fig = compare_two_results(mnist_benchmark1[:, 1:4], mnist_benchmark1[:, 0],
    #                               mnist_benchmark2[:, 1:4], mnist_benchmark2[:, 0])
    #     save_figure(fig, PLOT_DIR, "-", "testcompare3", str(key[0]))
    #     plt.close(fig)

# # plots
    # fig, ax = plt.subplots()
    # scatter = ax.scatter(x=mnist_latent[:, 0], y=mnist_latent[:, 1], c=mnist_latent[:, 2])
    # produce a legend with the unique colors from the scatter
    # legend1 = ax.legend(*scatter.legend_elements(),
    #                    loc="lower left", title="MNIST")
    # ax.add_artist(legend1)
    # plt.show()

    # seaborn
    # fig, axs = plt.subplots(ncols=2)
    # g = sns.scatterplot(x=mnist_benchmark1_0[:, 1],
    #                     y=mnist_benchmark1_0[:, 2],
    #                     hue=mnist_benchmark1_0[:, 0],
    #                     legend="full",
    #                     palette=sns.color_palette("bright"),
    #                     ax=axs[0])
    #
    # j = sns.scatterplot(x=mnist_benchmark2_0[:, 1],
    #                     y=mnist_benchmark2_0[:, 2],
    #                     hue=mnist_benchmark2_0[:, 0],
    #                     legend="full",
    #                     palette=sns.color_palette("bright"),
    #                     ax=axs[1])
    #
    # fig = g.get_figure()
    # figure_name = "tSNE-MNIST-" + str(datetime.now()).replace(":", "_").replace(".", "_")
    # fig.savefig(os.path.join(PLOT_DIR, figure_name))
    #
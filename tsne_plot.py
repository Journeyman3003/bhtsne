#!/usr/bin/env python

import matplotlib.pyplot as plt
import seaborn as sns
import os
import bhtsne
import mnist
from sys import argv
import numpy as np

# directory structure
CWD = os.path.dirname(os.path.realpath(__file__))
PLOT_DIR = os.path.join(CWD, "plots")


def plot_bh_tsne_result(_data, _labels, _legend="full", _palette="bright", _ax=None):
    _g = sns.scatterplot(x=_data[:, 0],
                         y=_data[:, 1],
                         hue=_labels,
                         legend=_legend,
                         palette=sns.color_palette(_palette),
                         ax=_ax)

    return _g


def compare_two_results(_data1, _label1, _data2, _label2, figsize=(10, 5)):
    _fig, _axs = plt.subplots(ncols=2, gridspec_kw={'width_ratios': [1, 1]}, figsize=figsize)
    plot_bh_tsne_result(_data1, _label1, _legend="full", _ax=_axs[0])
    plot_bh_tsne_result(_data2, _label2, _legend="full", _ax=_axs[1])

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

    _, labels = mnist.load_mnist_data(False)

    basepath = "C:\\Users\\Tobi\\git\\bhtsne\\results\\parametertuning\\iterations\\1000\\mnist2500\\2"
    benchmark1 = bhtsne.read_bh_tsne_result(os.path.join(basepath, "bh_tsne_result-08-07-2019_14-07-03.pickle"))
    benchmark2 = bhtsne.read_bh_tsne_result(os.path.join(basepath, "bh_tsne_result-08-07-2019_14-09-11.pickle"))

    for key in benchmark1.keys():
        mnist_benchmark1 = np.hstack((labels[:, None], benchmark1[key]))
        mnist_benchmark2 = np.hstack((labels[:, None], benchmark2[key]))

        fig = compare_two_results(mnist_benchmark1[:, 1:3], mnist_benchmark1[:, 0],
                                  mnist_benchmark2[:, 1:3], mnist_benchmark2[:, 0])
        save_figure(fig, PLOT_DIR, "-", "testcompare2", str(key[0]))

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
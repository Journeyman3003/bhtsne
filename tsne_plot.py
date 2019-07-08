#!/usr/bin/env python

import matplotlib.pyplot as plt
import seaborn as sns
import os
import bhtsne
from sys import argv
import numpy as np

# directory structure
CWD = os.path.dirname(os.path.realpath(__file__))
PLOT_DIR = os.path.join(CWD, "plots")


if __name__ == "__main__":
    #if argv < 2:
    #    print("Please specify the .pickle file to be read: tsne_plot.py <.pickle-file>!")
    dictionary_bh = bhtsne.debug_bh_tsne_post()
    #bhtsne.read_bh_tsne_result("C:\\Users\\Tobi\\git\\bhtsne\\windows\\result-0.dat")
    init = dictionary_bh[(0,)]
    np.sum(init[:, 2])
# # plots
    # fig, ax = plt.subplots()
    # scatter = ax.scatter(x=mnist_latent[:, 0], y=mnist_latent[:, 1], c=mnist_latent[:, 2])
    # produce a legend with the unique colors from the scatter
    # legend1 = ax.legend(*scatter.legend_elements(),
    #                    loc="lower left", title="MNIST")
    # ax.add_artist(legend1)
    # plt.show()

    # seaborn
    # g = sns.scatterplot(x=mnist_latent[:, 0],
    #                    y=mnist_latent[:, 1],
    #                    hue=mnist_latent[:, 2],
    #                    legend="full",
    #                    palette=sns.color_palette("bright"))
    #
    # fig = g.get_figure()
    # figure_name = "tSNE-MNIST-" + str(datetime.now()).replace(":", "_").replace(".", "_")
    # fig.savefig(os.path.join(PLOT_DIR, figure_name))
    #
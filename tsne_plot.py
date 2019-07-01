#!/usr/bin/env python

import matplotlib.pyplot as plt
import seaborn as sns
import os
import bhtsne

# directory structure
CWD = os.path.dirname(os.path.realpath(__file__))
PLOT_DIR = os.path.join(CWD, "plots")


if __name__ == "__main__":



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
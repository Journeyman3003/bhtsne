#!/usr/bin/env python

import numpy as np
import bhtsne
# import matplotlib.pyplot as plt
import seaborn as sns
# from datetime import datetime
# import os
# import pickle

# data load/preprocessing
# MNIST
import mnist
import notification
import logging
from streamlogger import StreamToLogger
import sys

PLOT_DIR = "plots"
DATA_DIR = "results"


def init_logger(logfile="bhtsne.log"):
    # logging stuff

    stdout_logger = logging.getLogger('STDOUT')
    sl = StreamToLogger(stdout_logger, logging.INFO, logfile)
    sys.stdout = sl

    stderr_logger = logging.getLogger('STDERR')
    sl = StreamToLogger(stderr_logger, logging.ERROR, logfile)
    sys.stderr = sl


if __name__ == "__main__":

    # initialize logging to file
    # init_logger()

    # mnist_data, mnist_labels = mnist.load_mnist_data()

    ###########################################################
    #                           DEBUG                         #
    ###########################################################

    #bhtsne.debug_bh_tsne_pre(mnist_data, initial_dims=mnist_data.shape[1], verbose=True)
    embedding_dict = bhtsne.debug_bh_tsne_post()

    # sanity check of error
    #np.sum(embedding_array[:, 2])

    ###########################################################
    #                           RUN                           #
    ###########################################################

    # embedding_dict = bhtsne.run_bh_tsne(mnist_data, initial_dims=mnist_data.shape[1], verbose=True)
    #
    # mnist_latent = np.hstack((embedding_array[:, 0:2], np.reshape(mnist_labels, (mnist_labels.shape[0], 1))))

    # # save embedding
    # embedding_name = "tSNE-MNIST-" + str(datetime.now()).replace(":", "_").replace(".", "_")





    # # plots
    # fig, ax = plt.subplots()
    # scatter = ax.scatter(x=mnist_latent[:, 0], y=mnist_latent[:, 1], c=mnist_latent[:, 2])
    # produce a legend with the unique colors from the scatter
    # legend1 = ax.legend(*scatter.legend_elements(),
    #                    loc="lower left", title="MNIST")
    # ax.add_artist(legend1)
    # plt.show()

    # seaborn
    g = sns.scatterplot(x=mnist_latent[:, 0],
                        y=mnist_latent[:, 1],
                        hue=mnist_latent[:, 2],
                        legend="full",
                        palette=sns.color_palette("bright"))
    #
    # fig = g.get_figure()
    # figure_name = "tSNE-MNIST-" + str(datetime.now()).replace(":", "_").replace(".", "_")
    # fig.savefig(os.path.join(PLOT_DIR, figure_name))
    #
    # # send final notification
    # notification.send_mail()




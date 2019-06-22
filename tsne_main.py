#!/usr/bin/env python

import bhtsne
import seaborn as sns
from datetime import datetime
import os

# data load/preprocessing
# MNIST
import mnist
import notification
import logging
from streamlogger import StreamToLogger
import sys

# directory structure
CWD = os.path.dirname(os.path.realpath(__file__))
PLOT_DIR = os.path.join(CWD, "plots")
RESULT_DIR = os.path.join(CWD, "results")

# Parameter tuning
PARAMTUNING_DIR = os.path.join(RESULT_DIR, "parametertuning")

# Building block experiments
BUILDINGBLOCK_DIR = os.path.join(RESULT_DIR, "buildingblocks")


LOGGING_DIR = os.path.join(CWD, "logging")
LOGGING_FILE = os.path.join(LOGGING_DIR, "bhtsne.log")

# PARAMETER TESTING LIST
PERPLEXITY = [2, 5, 10, 20, 30, 40, 50, 100]
T_MAX = 1000


def init_directories():
    try:
        os.makedirs(PLOT_DIR)
    except FileExistsError:
        # directory already exists
        print(PLOT_DIR)
        pass

    try:
        os.makedirs(PLOT_DIR)
    except FileExistsError:
        # directory already exists
        pass

    try:
        os.makedirs(PLOT_DIR)
    except FileExistsError:
        # directory already exists
        pass

    try:
        os.makedirs(PLOT_DIR)
    except FileExistsError:
        # directory already exists
        pass


def init_logger(logfile=LOGGING_FILE):
    # logging stuff

    stdout_logger = logging.getLogger('STDOUT')
    sl = StreamToLogger(stdout_logger, logging.INFO, logfile)
    sys.stdout = sl

    stderr_logger = logging.getLogger('STDERR')
    sl = StreamToLogger(stderr_logger, logging.ERROR, logfile)
    sys.stderr = sl


if __name__ == "__main__":

    # initialize directories
    init_directories()

    # initialize logging to file
    init_logger()

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

    # send final notification
    notification.send_mail()




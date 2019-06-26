#!/usr/bin/env python

import bhtsne
# import seaborn as sns
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

# DATA SUBDIRECTORIES
MNIST = "mnist"
# ...

# Parameter tuning
PARAMTUNING_DIR = os.path.join(RESULT_DIR, "parametertuning")
TMAX_TUNING_DIR = os.path.join(PARAMTUNING_DIR, "iterations")
PERPLEXITY_TUNING_DIR = os.path.join(PARAMTUNING_DIR, "perplexity")

# Building block experiments
BUILDINGBLOCK_DIR = os.path.join(RESULT_DIR, "buildingblocks")


LOGGING_DIR = os.path.join(CWD, "logging")
LOGGING_FILE = os.path.join(LOGGING_DIR, "bhtsne.log")

# PARAMETER TESTING LIST
PERPLEXITY = [2, 5, 10, 20, 30, 40, 50, 100]
T_MAX = [1000]


def init_directories():
    try:
        os.makedirs(PLOT_DIR)
    except FileExistsError:
        # directory already exists
        pass

    try:
        os.makedirs(PERPLEXITY_TUNING_DIR)
    except FileExistsError:
        # directory already exists
        pass

    try:
        os.makedirs(TMAX_TUNING_DIR)
    except FileExistsError:
        # directory already exists
        pass

    try:
        os.makedirs(BUILDINGBLOCK_DIR)
    except FileExistsError:
        # directory already exists
        pass

    try:
        os.makedirs(LOGGING_DIR)
    except FileExistsError:
        # directory already exists
        pass


def init_logger(logfile=LOGGING_FILE):
    # logging stuff

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s:%(levelname)s:%(name)s:%(message)s',
        filename=logfile,
        filemode='a'
    )

    stdout_logger = logging.getLogger('STDOUT')
    out_l = StreamToLogger(stdout_logger, logging.INFO, logfile)
    sys.stdout = out_l

    stderr_logger = logging.getLogger('STDERR')
    err_l = StreamToLogger(stderr_logger, logging.ERROR, logfile)
    sys.stderr = err_l

    return out_l, err_l


if __name__ == "__main__":

    # initialize directories
    init_directories()

    # for very paranoid beings...
    os.chdir(CWD)

    # initialize logging to file
    out, err = init_logger()

    ###########################################################
    #                       LOAD DATA                         #
    ###########################################################

    # MNIST DATA SMALL
    mnist_X, mnist_labels = mnist.load_mnist_data(all_data=False)
    # MNIST DATA ALL
    # mnist_X, mnist_labels = mnist.load_mnist_keras()

    ###########################################################
    #                           DEBUG                         #
    ###########################################################

    # bhtsne.debug_bh_tsne_pre(mnist_data, initial_dims=mnist_data.shape[1], verbose=True)
    # embedding_dict = bhtsne.debug_bh_tsne_post()

    # sanity check of error
    #np.sum(embedding_array[:, 2])

    ###########################################################
    #                           RUN                           #
    ###########################################################

    # For each Data set and parameter, perform tsne 5 times to have some reliable data

    ###########################################################
    #               PARAMETER TUNING - ITERATIONS             #
    ###########################################################

    for max_iter in T_MAX:
        print("Using T_MAX: " + str(max_iter))
        # 5 times to validate
        for i in range(5):
            print("###", "### Round:" + str(i+1), "###")
            # create directory if non-existent
            try:
                os.makedirs(os.path.join(TMAX_TUNING_DIR, str(max_iter), str(i+1)))
            except FileExistsError:
                # directory already exists
                pass

            # run t-SNE
            bh_tsne_dict = bhtsne.run_bh_tsne(mnist_X, initial_dims=mnist_X.shape[1], max_iter=max_iter, verbose=True)

            # save results
            # timestamp
            timestamp = str(datetime.now()).replace(":", "_").replace(".", "_")
            bhtsne.write_bh_tsne_result(bh_tsne_dict, os.path.join(TMAX_TUNING_DIR, str(max_iter), str(i+1)),
                                        "-", timestamp)




    ###########################################################
    #               PARAMETER TUNING - PERPLEXITY             #
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

    # send final notification
    notification.send_mail()




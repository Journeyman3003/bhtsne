#!/usr/bin/env python

import bhtsne

from datetime import datetime
import os

# data load/preprocessing
# MNIST
import mnist
import notification
import logging
from streamlogger import StreamToLogger
import sys
import shutil

# directory structure
CWD = os.path.dirname(os.path.realpath(__file__))
RESULT_DIR = os.path.join(CWD, "results")
RESULT_ZIP = os.path.join(CWD, "results.zip")

# DATA SUBDIRECTORIES
MNIST_TEST = "mnist2500"
MNIST = "mnist"
# ...

# Parameter tuning
PARAMTUNING_DIR = os.path.join(RESULT_DIR, "parametertuning")

TMAX_TUNING_DIR = os.path.join(PARAMTUNING_DIR, "iterations")
PERPLEXITY_TUNING_DIR = os.path.join(PARAMTUNING_DIR, "perplexity")
EXAGGERATION_TUNING_DIR = os.path.join(PARAMTUNING_DIR, "exaggeration")
THETA_TUNING_DIR = os.path.join(PARAMTUNING_DIR, "theta")
LEARNING_RATE_TUNING_DIR = os.path.join(PARAMTUNING_DIR, "learningrate")
MOMENTUM_TUNING_DIR = os.path.join(PARAMTUNING_DIR, "momentum")
FINAL_MOMENTUM_TUNING_DIR = os.path.join(PARAMTUNING_DIR, "finalmomentum")
STOP_LYING_TUNING_DIR = os.path.join(PARAMTUNING_DIR, "stoplying")
RESTART_LYING_TUNING_DIR = os.path.join(PARAMTUNING_DIR, "restartlying")
MOMENTUM_SWITCH_TUNING_DIR = os.path.join(PARAMTUNING_DIR, "momentumswitch")


# Building block experiments
BUILDINGBLOCK_DIR = os.path.join(RESULT_DIR, "buildingblocks")


LOGGING_DIR = os.path.join(CWD, "logging")
DAY = datetime.now().strftime("%d-%m-%Y")
LOGGING_FILE_NAME = "bhtsne-" + DAY + ".log"
LOGGING_FILE_ABSPATH = os.path.join(LOGGING_DIR, LOGGING_FILE_NAME)

# PARAMETER TESTING LIST
# TMAX is also default case, removed default variable setting from parameter list
T_MAX = [1000]
PERPLEXITY = [2, 5, 10, 20, 30, 40, 100]
EXAGGERATION = [1, 4, 8, 20]
THETA = [0, 0.2, 0.4, 0.6, 0.8, 1]
LEARNING_RATE = [50, 100, 500, 1000]
MOMENTUM = [0.0, 0.2, 0.4, 0.6, 0.8]
FINAL_MOMENTUM = [0.0, 0.2, 0.4, 0.5, 0.6]
STOP_LYING_ITER = [500, 750, 1000]
RESTART_LYING_ITER = [750]
MOMENTUM_SWITCH_ITER = [500, 750, 1000]


def init_directories():
    # try:
    #     os.makedirs(PLOT_DIR)
    # except FileExistsError:
    #     # directory already exists
    #     pass

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


def init_logger(logfile=LOGGING_FILE_ABSPATH):
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


def load_data(data_identifier):
    if data_identifier == MNIST:
        return mnist.load_mnist_data(all_data=True)
    elif data_identifier == MNIST_TEST:
        return mnist.load_mnist_data(all_data=False)


def tsne_parametertuning_workflow(parameter_name, value_list, data, result_base_dir, data_result_subdirectory):
    """

    :param parameter_name:
    :param value_list:
    :param data:
    :param result_base_dir:
    :param data_result_subdirectory:
    :return:

    """

    for value in value_list:
        print("###########################################")
        print("##              Start t-SNE              ##")
        print("###########################################")

        print("Using Dataset: {}".format(data_result_subdirectory))

        print("Tuning parameter: " + parameter_name + ", value: " + str(value))
        # 5 times to validate
        for i in range(5):
            print("###", "### Round:" + str(i+1), "###")
            # create directory if non-existent
            result_dir = os.path.join(result_base_dir, str(value), data_result_subdirectory, str(i + 1))
            try:
                os.makedirs(result_dir)
            except FileExistsError:
                # directory already exists
                pass

            # run t-SNE
            bh_tsne_dict = bhtsne.run_bh_tsne(data, initial_dims=data.shape[1], verbose=True,
                                              **{parameter_name: value})

            # save results
            # timestamp
            timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
            bhtsne.write_bh_tsne_result(bh_tsne_dict, result_dir, "-", timestamp)


if __name__ == "__main__":
    from sys import argv
    from distutils.util import strtobool

    # for default debug operation
    data_name = MNIST_TEST

    if len(argv) != 2 or argv[1] not in [MNIST, MNIST_TEST]:
        print("Error: did not call script passing correct data identifier!\n"
              "Please pass either of the following data identifiers:\n"
              "{}".format(str([MNIST, MNIST_TEST])))
        while True:
            try:
                debug = strtobool(input("Do you want to use the mnist2500 for debugging purposes? [y/n] "))
                if debug:
                    print("Running in debug mode with mnist2500")
                    break
                else:
                    print("Shutting down...")
                    quit()
            except ValueError:
                print("Please answer 'yes' ('y') or 'no' ('n').")
                continue
    else:
        data_name = argv[1]
    # initialize directories
    init_directories()

    # for very paranoid beings...
    os.chdir(CWD)

    # initialize logging to file
    init_logger()

    ###########################################################
    #                       LOAD DATA                         #
    ###########################################################

    data, _ = load_data(data_name)

    ###########################################################
    #                           DEBUG                         #
    ###########################################################

    # bhtsne.debug_bh_tsne_pre(mnist_data, initial_dims=mnist_data.shape[1], verbose=True)
    # embedding_dict = bhtsne.debug_bh_tsne_post()

    # sanity check of error
    # np.sum(embedding_array[:, 2])

    ###########################################################
    #                           RUN                           #
    ###########################################################

    # For each Data set and parameter, perform tsne 5 times to have some reliable data

    # MNIST_TEST

    ###########################################################
    #               PARAMETER TUNING - ITERATIONS             #
    ###########################################################

    tsne_parametertuning_workflow(parameter_name="max_iter", value_list=T_MAX, data=data,
                                  data_result_subdirectory=data_name, result_base_dir=TMAX_TUNING_DIR)

    # ###########################################################
    # #               PARAMETER TUNING - PERPLEXITY             #
    # ###########################################################
    #
    # tsne_parametertuning_workflow(parameter_name="perplexity", value_list=PERPLEXITY, data=data,
    #                               data_result_subdirectory=data_name, result_base_dir=PERPLEXITY_TUNING_DIR)
    #
    # ###########################################################
    # #               PARAMETER TUNING - EXAGGERATION           #
    # ###########################################################
    #
    # tsne_parametertuning_workflow(parameter_name="lying_factor", value_list=EXAGGERATION, data=data,
    #                               data_result_subdirectory=data_name, result_base_dir=EXAGGERATION_TUNING_DIR)
    #
    # ###########################################################
    # #               PARAMETER TUNING - THETA                  #
    # ###########################################################
    #
    # tsne_parametertuning_workflow(parameter_name="theta", value_list=THETA, data=data,
    #                               data_result_subdirectory=data_name, result_base_dir=THETA_TUNING_DIR)
    #
    # ###########################################################
    # #               PARAMETER TUNING - LEARNING RATE          #
    # ###########################################################
    #
    # tsne_parametertuning_workflow(parameter_name="learning_rate", value_list=LEARNING_RATE, data=data,
    #                               data_result_subdirectory=data_name, result_base_dir=LEARNING_RATE_TUNING_DIR)
    #
    # ###########################################################
    # #               PARAMETER TUNING - MOMENTUM               #
    # ###########################################################
    #
    # tsne_parametertuning_workflow(parameter_name="momentum", value_list=MOMENTUM, data=data,
    #                               data_result_subdirectory=data_name, result_base_dir=MOMENTUM_TUNING_DIR)
    #
    # ###########################################################
    # #               PARAMETER TUNING - FINAL MOMENTUM         #
    # ###########################################################
    #
    # tsne_parametertuning_workflow(parameter_name="final_momentum", value_list=FINAL_MOMENTUM, data=data,
    #                               data_result_subdirectory=data_name, result_base_dir=FINAL_MOMENTUM_TUNING_DIR)
    #
    # ###########################################################
    # #               PARAMETER TUNING - STOP LYING ITER        #
    # ###########################################################
    #
    # tsne_parametertuning_workflow(parameter_name="stop_lying_iter", value_list=STOP_LYING_ITER, data=data,
    #                               data_result_subdirectory=data_name, result_base_dir=STOP_LYING_TUNING_DIR)
    #
    # ###########################################################
    # #               PARAMETER TUNING - RESTART LYING ITER     #
    # ###########################################################
    #
    # tsne_parametertuning_workflow(parameter_name="restart_lying_iter", value_list=RESTART_LYING_ITER, data=data,
    #                               data_result_subdirectory=data_name, result_base_dir=RESTART_LYING_TUNING_DIR)
    #
    # ###########################################################
    # #               PARAMETER TUNING - MOMENTUM SWITCH ITER   #
    # ###########################################################
    #
    # tsne_parametertuning_workflow(parameter_name="momentum_switch_iter", value_list=MOMENTUM_SWITCH_ITER, data=data,
    #                               data_result_subdirectory=data_name, result_base_dir=MOMENTUM_SWITCH_TUNING_DIR)

    # create zip archive of results
    shutil.make_archive(RESULT_DIR, 'zip', RESULT_DIR)

    # send final notification
    notification.send_mail(LOGGING_FILE_NAME, LOGGING_FILE_ABSPATH, "results.zip", RESULT_ZIP)




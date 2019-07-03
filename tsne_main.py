#!/usr/bin/env python

import bhtsne

from datetime import datetime
import os

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

DATA_SETS = [MNIST_TEST, MNIST]
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

# main
T_MAX = [1000]
PERPLEXITY = [2, 5, 10, 20, 30, 40, 100]
EXAGGERATION = [1, 4, 8, 20]
THETA = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1]
LEARNING_RATE = [50, 100, 500, 1000]

# secondary
MOMENTUM = [0.0, 0.2, 0.4, 0.6, 0.8]
FINAL_MOMENTUM = [0.0, 0.2, 0.4, 0.5, 0.6]
STOP_LYING_ITER = [500, 750, 1000]
RESTART_LYING_ITER = [750]
MOMENTUM_SWITCH_ITER = [500, 750, 1000]


#############################################
#       PARAMETER TUNING - DICTIONARY       #
#############################################

# Python dict to retrieve all params by key
# parameter-name: [value-list, directory to store results]
PARAM_DICT = {
    "max_iter": [T_MAX, TMAX_TUNING_DIR],
    "perplexity": [PERPLEXITY, PERPLEXITY_TUNING_DIR],
    "lying_factor": [EXAGGERATION, EXAGGERATION_TUNING_DIR],
    "learning_rate": [LEARNING_RATE, LEARNING_RATE_TUNING_DIR],
    "momentum": [MOMENTUM, MOMENTUM_TUNING_DIR],
    "final_momentum": [FINAL_MOMENTUM, FINAL_MOMENTUM_TUNING_DIR],
    "stop_lying_iter": [STOP_LYING_ITER, STOP_LYING_TUNING_DIR],
    "restart_lying_iter": [RESTART_LYING_ITER, RESTART_LYING_TUNING_DIR],
    "momentum_switch_iter": [MOMENTUM_SWITCH_ITER, MOMENTUM_SWITCH_TUNING_DIR],
    "theta": [THETA, THETA_TUNING_DIR]
}


def init_directories():
    try:
        os.makedirs(PARAMTUNING_DIR)
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
        for i in range(1):
            print("###", "### Round:" + str(i+1), "###")
            # create directory if non-existent
            result_dir = os.path.join(result_base_dir, str(value), data_result_subdirectory, str(i + 1))
            try:
                os.makedirs(result_dir)
            except FileExistsError:
                # directory already exists
                pass

            # run t-SNE
            # perform PCA to 50 dims beforehand
            bh_tsne_dict = bhtsne.run_bh_tsne(data, verbose=True, **{parameter_name: value})

            # save results
            # timestamp
            timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
            bhtsne.write_bh_tsne_result(bh_tsne_dict, result_dir, "-", timestamp)


if __name__ == "__main__":
    # put everything into try except clause and send error notification on failure
    from sys import argv
    from distutils.util import strtobool
    import traceback
    try:
        # for default debug operation
        data_name = MNIST_TEST
        param_list = ["max_iter"]

        # for parallelism
        # num_processes = 1

        if len(argv) < 2:
            print("Error: did not call script passing correct data and parameter identifier!\n"
                  "Run script as follows: python3 tsne_main.py <data_identifier> [optional] <parameters>\n"
                  "available data identifiers: {}".format(str(DATA_SETS)),
                  "available [optional] parameter identifiers: {}".format(str(PARAM_DICT.keys())))
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
        elif argv[1] not in [MNIST, MNIST_TEST]:
            # validate passed dataset identifier
            print("data identifier has to be either of the following: {}".format(str(DATA_SETS)))
            print("Shutting down...")
            quit()
        elif len(argv) < 3:
            # all parameters
            while True:
                try:
                    all_param = strtobool(input("No parameter identifier specified."
                                                "Do you want to use all parameters [y] or just maximum iterations [n]?"))
                    if all_param:
                        print("Using all parameters:\n {}".format(str(PARAM_DICT.keys())))
                        break
                    else:
                        print("Using maximum iterations")
                        break
                except ValueError:
                    print("Please answer 'yes' ('y') or 'no' ('n').")
                    continue
        elif argv[2] == "all":
            # all parameters from dict
            print("Using all parameters")
            data_name = argv[1]
            param_list = PARAM_DICT.keys()
        else:
            # selected parameters
            for param in argv[2:]:
                if param not in PARAM_DICT.keys():
                    print("unrecognized parameter identifier: {}".format(param))
                    print("Shutting down...")
                    quit()
            # all checks passed, set values
            data_name = argv[1]
            param_list = argv[2:]

            # num_processes = 4 if len(param_list) >= 4 else len(param_list)

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

        # bhtsne.debug_bh_tsne_pre(data)
        # embedding_dict = bhtsne.debug_bh_tsne_post()

        # sanity check of error
        # np.sum(embedding_array[:, 2])

        ###########################################################
        #                   RUN PARAMETER TUNING                  #
        ###########################################################

        # For each Data set and parameter, perform tsne 5 times to have some reliable data

        for param in param_list:
            tsne_parametertuning_workflow(parameter_name=param, value_list=PARAM_DICT[param][0], data=data,
                                          data_result_subdirectory=data_name, result_base_dir=PARAM_DICT[param][1])

        # create zip archive of results
        shutil.make_archive(RESULT_DIR, 'zip', RESULT_DIR)

        # send final notification
        notification.send_mail(LOGGING_FILE_NAME, LOGGING_FILE_ABSPATH, "results.zip", RESULT_ZIP, argv)
    except Exception:
        traceback.print_exc()
        notification.send_error(LOGGING_FILE_NAME, LOGGING_FILE_ABSPATH, argv)



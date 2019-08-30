#!/usr/bin/env python

import bhtsne

from datetime import datetime
import os

# MNIST
import mnist

import notification
import logging
from streamlogger import StreamToLogger
from data_initializer import get_initial_embedding, get_supported_non_random_methods
import sys
from argparse import ArgumentParser
import shutil

# directory structure
CWD = os.path.dirname(os.path.realpath(__file__))
RESULT_DIR = os.path.join(CWD, "results")
RESULT_ZIP = os.path.join(CWD, "results.zip")

# DATA SUBDIRECTORIES
MNIST_TEST = "mnist2500"
MNIST = "mnist"
FASHION_MNIST = "fashion_mnist"
FASHION_MNIST10 = "fashion_mnist10"
FASHION_MNIST100 = "fashion_mnist100"
FASHION_MNIST1000 = "fashion_mnist1000"
FASHION_MNIST2500 = "fashion_mnist2500"
FASHION_MNIST5000 = "fashion_mnist5000"
FASHION_MNIST7000 = "fashion_mnist7000"
FASHION_MNIST10000 = "fashion_mnist10000"
FASHION_MNIST20000 = "fashion_mnist20000"


DATA_SETS = [MNIST_TEST, MNIST, FASHION_MNIST, FASHION_MNIST10, FASHION_MNIST100, FASHION_MNIST1000, FASHION_MNIST2500,
             FASHION_MNIST5000, FASHION_MNIST7000, FASHION_MNIST10000, FASHION_MNIST20000]

# Runtime testing
RUNTIME_DIR = "run_time"

# Parameter tuning
PARAMTUNING_DIR = "parametertuning"

TMAX_TUNING_DIR = "iterations"
PERPLEXITY_TUNING_DIR = "perplexity"
EXAGGERATION_TUNING_DIR = "exaggeration"
THETA_TUNING_DIR = "theta"
LEARNING_RATE_TUNING_DIR = "learningrate"
MOMENTUM_TUNING_DIR = "momentum"
FINAL_MOMENTUM_TUNING_DIR = "finalmomentum"
STOP_LYING_TUNING_DIR = "stoplying"
RESTART_LYING_TUNING_DIR = "restartlying"
MOMENTUM_SWITCH_TUNING_DIR = "momentumswitch"

# Initial solutions
INIT = os.path.join(CWD, "initial_solutions")

# Building block experiments
BUILDINGBLOCK_DIR = "buildingblocks"

INITIAL_EMBEDDING_DIR = os.path.join(BUILDINGBLOCK_DIR, "initial_embeddings")

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
    "theta": [THETA, THETA_TUNING_DIR],
    "lying_factor": [EXAGGERATION, EXAGGERATION_TUNING_DIR],
    "learning_rate": [LEARNING_RATE, LEARNING_RATE_TUNING_DIR],
    "momentum": [MOMENTUM, MOMENTUM_TUNING_DIR],
    "final_momentum": [FINAL_MOMENTUM, FINAL_MOMENTUM_TUNING_DIR],
    "stop_lying_iter": [STOP_LYING_ITER, STOP_LYING_TUNING_DIR],
    "restart_lying_iter": [RESTART_LYING_ITER, RESTART_LYING_TUNING_DIR],
    "momentum_switch_iter": [MOMENTUM_SWITCH_ITER, MOMENTUM_SWITCH_TUNING_DIR]
}


def init_directories():
    try:
        os.makedirs(os.path.join(RESULT_DIR, "tSNE", PARAMTUNING_DIR))
    except FileExistsError:
        # directory already exists
        pass

    try:
        os.makedirs(os.path.join(RESULT_DIR, "BHtSNE", PARAMTUNING_DIR))
    except FileExistsError:
        # directory already exists
        pass

    try:
        os.makedirs(os.path.join(RESULT_DIR, "tSNE", BUILDINGBLOCK_DIR))
    except FileExistsError:
        # directory already exists
        pass

    try:
        os.makedirs(os.path.join(RESULT_DIR, "BHtSNE", BUILDINGBLOCK_DIR))
    except FileExistsError:
        # directory already exists
        pass

    try:
        os.makedirs(LOGGING_DIR)
    except FileExistsError:
        # directory already exists
        pass

    try:
        os.makedirs(INIT)
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
    elif data_identifier == FASHION_MNIST:
        return mnist.load_fashion_mnist_data(all_data=True)
    elif data_identifier == FASHION_MNIST10:
        return mnist.load_fashion_mnist_data(all_data=False, len_sample=10)
    elif data_identifier == FASHION_MNIST100:
        return mnist.load_fashion_mnist_data(all_data=False, len_sample=100)
    elif data_identifier == FASHION_MNIST1000:
        return mnist.load_fashion_mnist_data(all_data=False, len_sample=1000)
    elif data_identifier == FASHION_MNIST2500:
        return mnist.load_fashion_mnist_data(all_data=False, len_sample=2500)
    elif data_identifier == FASHION_MNIST5000:
        return mnist.load_fashion_mnist_data(all_data=False, len_sample=5000)
    elif data_identifier == FASHION_MNIST7000:
        return mnist.load_fashion_mnist_data(all_data=False, len_sample=7000)
    elif data_identifier == FASHION_MNIST10000:
        return mnist.load_fashion_mnist_data(all_data=False, len_sample=10000)
    elif data_identifier == FASHION_MNIST20000:
        return mnist.load_fashion_mnist_data(all_data=False, len_sample=10000)
    else:
        print("unsupported data identifier: " + data_identifier)
        print("Shutting down...")
        quit()


def _argparse():
    argparse = ArgumentParser('Script to run parametertuning and building block analysis of bh_tsne')
    argparse.add_argument('-e', '--exact', action='store_true', default=False)
    argparse.add_argument('-d', '--data_set', choices=DATA_SETS, default=MNIST_TEST,
                          help="use one of the following available data identifiers: {}".format(str(DATA_SETS)))
    available_parameters = ["all"]
    available_parameters.extend(PARAM_DICT.keys())
    argparse.add_argument('-p', '--parameter_list', choices=available_parameters, nargs='+', default=["max_iter"],
                          help="use all or selected parameter identifiers from the following list: {}"
                          .format(str(PARAM_DICT.keys())))
    argparse.add_argument('-i', '--initial_embedding', choices=["gaussian", "pca", "lle"], default="gaussian")
    argparse.add_argument('-pt', '--parametertuning', action='store_true', default=False)
    argparse.add_argument('-y', '--y_init', action='store_true', default=False)
    argparse.add_argument('-r', '--run_time', action='store_true', default=False)
    argparse.add_argument('-insim', '--input_similarities', default="gaussian",
                          choices=bhtsne.BUILDING_BLOCK_DICT["input_similarities"].keys())
    argparse.add_argument('-outsim', '--output_similarities', default="student",
                          choices=bhtsne.BUILDING_BLOCK_DICT["output_similarities"].keys())
    argparse.add_argument('-cf', '--cost_function', default="KL",
                          choices=bhtsne.BUILDING_BLOCK_DICT["cost_function"].keys())
    argparse.add_argument('-opt', '--optimization', default="gradient_descent",
                          choices=bhtsne.BUILDING_BLOCK_DICT["optimization"].keys())

    return argparse


def tsne_workflow(parameter_name, value_list, data, result_base_dir, data_result_subdirectory,
                  initial_embedding_method=None, **kwargs):
    """

    :param parameter_name:
    :param value_list:
    :param data:
    :param result_base_dir:
    :param data_result_subdirectory:
    :param initial_embedding_method:
    :return:

    """

    for value in value_list:
        print("###########################################")
        print("##              Start t-SNE              ##")
        print("###########################################")

        print("Using Dataset: {}".format(data_result_subdirectory))

        print("Tuning parameter: " + parameter_name + ", value: " + str(value))
        # 5 times to validate for random methods, once for specified inputs

        max_round = 6 if initial_embedding_method in ['gaussian', 'random'] else 2

        for i in range(1, max_round):
            print("###", "### Round:" + str(i), "###")
            # create directory if non-existent
            result_dir = os.path.join(result_base_dir, str(value), data_result_subdirectory, str(i))
            try:
                os.makedirs(result_dir)
            except FileExistsError:
                # directory already exists
                pass

            # load the initial embedding if specified
            _initial_embedding = None
            if initial_embedding_method is not None:
                _initial_embedding = get_initial_embedding(data_name=data_result_subdirectory,
                                                           method_name=initial_embedding_method, i=i)
                filename = "initial_solution_" + data_result_subdirectory + "_" + initial_embedding_method  \
                           + "{}" + ".pickle"
                filename = filename.format("_" + str(i) if initial_embedding_method in ['random', 'gaussian'] else "")

                print("Using initial embedding file: {}".format(filename))

            # run t-SNE
            # perform PCA to 50 dims beforehand
            # use initial embedding
            bh_tsne_dict = bhtsne.run_bh_tsne(data, verbose=True, initial_solution=_initial_embedding,
                                              **{parameter_name: value}, **kwargs)

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
        parser = _argparse()

        if len(argv) <= 1:
            print(parser.print_help())
            while True:
                try:
                    debug = strtobool(input("Do you want to run in debug mode? [y/n] "))
                    if debug:
                        print("Running in debug mode: tsne_main.py mnist2500 max_iter")
                        break
                    else:
                        print("Shutting down...")
                        quit()
                except ValueError:
                    print("Please answer 'yes' ('y') or 'no' ('n').")
                    continue

        argp = parser.parse_args(argv[1:])

        print("Using exact tSNE" if argp.exact else "Using BH-tSNE")
        exact = argp.exact
        algorithm = "tSNE" if exact else "BHtSNE"
        algorithm_dir = os.path.join(RESULT_DIR, algorithm)
        print("Using data set: {}".format(argp.data_set))
        data_name = argp.data_set
        print("Using parameters: {}".format(argp.parameter_list))
        param_list = PARAM_DICT.keys() if "all" in argp.parameter_list else argp.parameter_list
        # Remove theta if algorithm is exact!
        if exact:
            param_list = [x for x in param_list if x != 'theta']
            if not param_list:
                # if param_list is now empty, quit
                print("Exact tsne cannot be run with theta parameter tuning!")
                quit()
        if not argp.parametertuning and argp.y_init:
            print("Running y_init buildingblock test with initial embeddings: {}"
                  .format(get_supported_non_random_methods()))
        else:
            print("Using initial embeddings: {}".format(argp.initial_embedding))
        initial_embedding = argp.initial_embedding
        operation_dir = os.path.join(algorithm_dir, PARAMTUNING_DIR if argp.parametertuning else BUILDINGBLOCK_DIR)
        print("Using base directory: {}".format(str(operation_dir)))

        # initialize directories
        init_directories()

        # for very paranoid beings...
        os.chdir(CWD)

        # initialize logging to file
        init_logger()

        ###########################################################
        #                       LOAD DATA                         #
        ###########################################################

        data, labels = load_data(data_name)

        ###########################################################
        #                           DEBUG                         #
        ###########################################################

        #bhtsne.debug_bh_tsne_pre(data, data_name)
        #quit()
        # bhtsne.debug_data_file("windows", 2500, 784)
        #embedding_dict = bhtsne.debug_bh_tsne_post()
        #bhtsne.plot_bh_tsne_post(embedding_dict, labels)
        #quit()
        # sanity check of error
        # np.sum(embedding_array[:, 2])
        # quit()

        ###########################################################
        #                   RUNTIME Evaluation                    #
        ###########################################################

        if argp.run_time:
            if exact:
                for param in param_list:
                    for num_samples in [200, 500, 1000, 2500, 5000, 7500, 10000, 20000, 30000]:

                        data, _ = mnist.load_fashion_mnist_data(False, len_sample=num_samples)

                        tsne_workflow(parameter_name=param, value_list=PARAM_DICT[param][0], data=data,
                                      data_result_subdirectory="fashion_mnist" + str(num_samples),
                                      result_base_dir=os.path.join(algorithm_dir, RUNTIME_DIR),
                                      theta=0.0)
            else:
                for param in param_list:
                    for num_samples in [200, 500, 1000, 2500, 5000, 7500, 10000, 20000, 30000, 40000, 50000, 60000, 70000]:
                        data, _ = mnist.load_fashion_mnist_data(False, len_sample=num_samples)

                        tsne_workflow(parameter_name=param, value_list=PARAM_DICT[param][0], data=data,
                                      data_result_subdirectory="fashion_mnist" + str(num_samples),
                                      result_base_dir=os.path.join(algorithm_dir, RUNTIME_DIR))

        ###########################################################
        #                   RUN PARAMETER TUNING                  #
        ###########################################################

        # For each Data set and parameter, perform tsne 5 times to have some reliable data
        elif argp.parametertuning:
            for param in param_list:
                if exact:
                    """
                    pass an additional theta=0.0 if running exact tSNE
                    """
                    tsne_workflow(parameter_name=param, value_list=PARAM_DICT[param][0], data=data,
                                  data_result_subdirectory=data_name,
                                  result_base_dir=os.path.join(operation_dir, PARAM_DICT[param][1]),
                                  initial_embedding_method=initial_embedding, theta=0.0)
                else:
                    tsne_workflow(parameter_name=param, value_list=PARAM_DICT[param][0], data=data,
                                  data_result_subdirectory=data_name,
                                  result_base_dir=os.path.join(operation_dir, PARAM_DICT[param][1]),
                                  initial_embedding_method=initial_embedding)

        ###########################################################
        #                    INITIAL EMBEDDINGS                   #
        ###########################################################
        elif argp.y_init:
            for param in param_list:
                for method in get_supported_non_random_methods():
                    if exact:
                        """
                        pass an additional theta=0.0 if running exact tSNE
                        """
                        tsne_workflow(parameter_name=param, value_list=PARAM_DICT[param][0], data=data,
                                      data_result_subdirectory=data_name,
                                      result_base_dir=os.path.join(algorithm_dir, INITIAL_EMBEDDING_DIR,
                                                                   method, PARAM_DICT[param][1]),
                                      initial_embedding_method=method, theta=0.0)
                    else:
                        tsne_workflow(parameter_name=param, value_list=PARAM_DICT[param][0], data=data,
                                      data_result_subdirectory=data_name,
                                      result_base_dir=os.path.join(algorithm_dir, INITIAL_EMBEDDING_DIR,
                                                                   method, PARAM_DICT[param][1]),
                                      initial_embedding_method=method,)

        ###########################################################
        #                RUN BUILDINGBLOCK ANALYSIS               #
        ###########################################################
        else:
            building_blocks = [("input_similarities", argp.input_similarities)]
            building_blocks.append(("output_similarities", argp.output_similarities))
            building_blocks.append(("cost_function", argp.cost_function))
            building_blocks.append(("optimization", argp.optimization))

            modified_buildingblocks = list(filter(lambda x: bhtsne.BUILDING_BLOCK_DICT[x[0]][x[1]] != 0,
                                                  building_blocks))

            if not modified_buildingblocks:
                modified_buildingblocks = [("default", "default")]

            directory = os.path.join("-".join([x[0] for x in modified_buildingblocks]),
                                     "-".join([x[1] for x in modified_buildingblocks]))
            kwargs = {x[0]: bhtsne.BUILDING_BLOCK_DICT[x[0]][x[1]] for x in building_blocks}
            if exact:
                kwargs['theta'] = 0.0
            for param in param_list:
                tsne_workflow(parameter_name=param, value_list=PARAM_DICT[param][0], data=data,
                              data_result_subdirectory=data_name,
                              result_base_dir=os.path.join(operation_dir, directory, PARAM_DICT[param][1]),
                              initial_embedding_method=initial_embedding, **kwargs)

        # skip zip attachment as it simply grows too big
        # create zip archive of results
        # shutil.make_archive(RESULT_DIR, 'zip', RESULT_DIR)

        # send final notification
        # notification.send_mail(LOGGING_FILE_NAME, LOGGING_FILE_ABSPATH, "results.zip", RESULT_ZIP, argv)
        notification.send_mail(LOGGING_FILE_NAME, LOGGING_FILE_ABSPATH, argv)
    except Exception:
        traceback.print_exc()
        notification.send_error(LOGGING_FILE_NAME, LOGGING_FILE_ABSPATH, argv)



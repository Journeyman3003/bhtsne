#!/usr/bin/env python

"""
A simple Python wrapper for the bh_tsne binary that makes it easier to use it
for TSV files in a pipeline without any shell script trickery.

Note: The script does some minimal sanity checking of the input, but don't
    expect it to cover all cases. After all, it is a just a wrapper.

Example:

    > echo -e '1.0\t0.0\n0.0\t1.0' | ./bhtsne.py -d 2 -p 0.1
    -2458.83181442  -6525.87718385
    2458.83181442   6525.87718385

The output will not be normalised, maybe the below one-liner is of interest?:

    python -c 'import numpy;  from sys import stdin, stdout;
        d = numpy.loadtxt(stdin); d -= d.min(axis=0); d /= d.max(axis=0);
        numpy.savetxt(stdout, d, fmt="%.8f", delimiter="\t")'

Authors:     Pontus Stenetorp    <pontus stenetorp se>
             Philippe Remy       <github: philipperemy>
Version:    2016-03-08
"""

# Copyright (c) 2013, Pontus Stenetorp <pontus stenetorp se>
#
# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted, provided that the above
# copyright notice and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
# WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
# ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
# WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
# ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
# OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

from argparse import ArgumentParser, FileType
from os.path import abspath, dirname, isfile, join as path_join
from shutil import rmtree
from struct import calcsize, pack, unpack
from subprocess import Popen, PIPE
from sys import stderr, stdin, stdout
from tempfile import mkdtemp
from platform import system
from os import devnull
import numpy as np
import os
import io
import glob
import pickle
from data_initializer import get_initial_embedding

### Constants
IS_WINDOWS = True if system() == 'Windows' else False
BH_TSNE_BIN_PATH = path_join(dirname(__file__), 'windows', 'bh_tsne.exe') if IS_WINDOWS \
                                                                          else path_join(dirname(__file__), 'bh_tsne')
assert isfile(BH_TSNE_BIN_PATH), ('Unable to find the bh_tsne binary in the '
                                  'same directory as this script, have you forgotten to compile it?: {}'
                                  ).format(BH_TSNE_BIN_PATH)
# Default hyper-parameter values from van der Maaten (2014)
# https://lvdmaaten.github.io/publications/papers/JMLR_2014.pdf (Experimental Setup, page 13)
DEFAULT_NO_DIMS = 2
INITIAL_DIMENSIONS = 50
DEFAULT_PERPLEXITY = 50
DEFAULT_THETA = 0.5
DEFAULT_LEARNING_RATE = 200.0
DEFAULT_MOMENTUM = 0.5
DEFAULT_FINAL_MOMENTUM = 0.8
EMPTY_SEED = -1
DEFAULT_USE_PCA = True
DEFAULT_MAX_ITERATIONS = 1000
DEFAULT_STOP_LYING_ITERATION = 250
DEFAULT_RESTART_LYING_ITERATION = 1001
DEFAULT_MOMENTUM_SWITCH_ITERATION = 250
DEFAULT_EXAGGERATION_FACTOR = 12

# Additional building block hyperparameters
DEFAULT_BUILDING_BLOCK_INDEX = 0
BUILDING_BLOCK_DICT = {
    "input_similarities": {
        "gaussian": 0,
        "laplacian": 1,
        "student": 2
    },
    "output_similarities":  {
        "student": 0,
        "chi": 1,
        "studenthalf": 2,
        "studentalpha": 3
    },
    "cost_function": {
        "KL": 0,
        "RKL": 1,
        "JS": 2
    },
    "optimization": {
        "gradient_descent": 0,
        "genetic": 1
    }
}

# Index for which points to keep fixed
DEFAULT_FREEZE_INDEX = 0


def _argparse():
    argparse = ArgumentParser('bh_tsne Python wrapper')
    argparse.add_argument('-d', '--no_dims', type=int, default=DEFAULT_NO_DIMS)
    argparse.add_argument('-p', '--perplexity', type=float, default=DEFAULT_PERPLEXITY)
    argparse.add_argument('-t', '--theta', type=float, default=DEFAULT_THETA)
    argparse.add_argument('-r', '--randseed', type=int, default=EMPTY_SEED)
    argparse.add_argument('-n', '--initial_dims', type=int, default=INITIAL_DIMENSIONS)
    argparse.add_argument('-v', '--verbose', action='store_true')
    argparse.add_argument('-i', '--input', type=FileType('r'), default=stdin)
    argparse.add_argument('-o', '--output', type=FileType('w'), default=stdout)
    argparse.add_argument('--use_pca', action='store_true')
    argparse.add_argument('--no_pca', dest='use_pca', action='store_false')
    argparse.set_defaults(use_pca=DEFAULT_USE_PCA)
    argparse.add_argument('-m', '--max_iter', type=int, default=DEFAULT_MAX_ITERATIONS)
    return argparse


def _read_unpack(fmt, fh):
    return unpack(fmt, fh.read(calcsize(fmt)))


def _is_filelike_object(f):
    try:
        return isinstance(f, (file, io.IOBase))
    except NameError:
        # 'file' is not a class in python3
        return isinstance(f, io.IOBase)


def _pca(samples, target_dimensions):
    samples = samples - np.mean(samples, axis=0)
    cov_x = np.dot(np.transpose(samples), samples)
    [eig_val, eig_vec] = np.linalg.eig(cov_x)

    # sorting the eigen-values in the descending order
    eig_vec = eig_vec[:, eig_val.argsort()[::-1]]

    # check validity of target_dimensions
    if target_dimensions > len(eig_vec):
        target_dimensions = len(eig_vec)

    # truncating the eigen-vectors matrix to keep the most important vectors
    eig_vec = np.real(eig_vec[:, :target_dimensions])
    return np.dot(samples, eig_vec)


def init_bh_tsne(samples, workdir, no_dims=DEFAULT_NO_DIMS, initial_dims=INITIAL_DIMENSIONS, initial_solution=None,
                 perplexity=DEFAULT_PERPLEXITY, learning_rate=DEFAULT_LEARNING_RATE, momentum=DEFAULT_MOMENTUM,
                 final_momentum=DEFAULT_FINAL_MOMENTUM, theta=DEFAULT_THETA, randseed=EMPTY_SEED,
                 use_pca=DEFAULT_USE_PCA, max_iter=DEFAULT_MAX_ITERATIONS, stop_lying_iter=DEFAULT_STOP_LYING_ITERATION,
                 restart_lying_iter=DEFAULT_RESTART_LYING_ITERATION,
                 momentum_switch_iter=DEFAULT_MOMENTUM_SWITCH_ITERATION, lying_factor=DEFAULT_EXAGGERATION_FACTOR,
                 input_similarities=DEFAULT_BUILDING_BLOCK_INDEX, output_similarities=DEFAULT_BUILDING_BLOCK_INDEX,
                 cost_function=DEFAULT_BUILDING_BLOCK_INDEX, optimization=DEFAULT_BUILDING_BLOCK_INDEX,
                 freeze_index=DEFAULT_FREEZE_INDEX):

    # apply PCA if desired
    if use_pca:
        samples = _pca(samples, initial_dims)

    # Assume that the dimensionality of the first sample is representative for
    #   the whole batch
    sample_dim = len(samples[0])
    sample_count = len(samples)

    # Note: The binary format used by bh_tsne is roughly the same as for
    #   vanilla tsne
    with open(path_join(workdir, 'data.dat'), 'wb') as data_file:
        # Write the bh_tsne header
        data_file.write(pack('iidddddiiiiii',
                             # 2 ints
                             sample_count, sample_dim,
                             # 5 double
                             theta, perplexity, learning_rate, momentum, final_momentum,
                             # 6 ints
                             no_dims, max_iter, stop_lying_iter, restart_lying_iter, momentum_switch_iter,
                             lying_factor))
        # Write the building block instructions
        data_file.write(pack('iiii', input_similarities, output_similarities, cost_function, optimization))

        # Write the freeze index
        data_file.write(pack('i', freeze_index))
        # Then write the data
        for sample in samples:
            data_file.write(pack('{}d'.format(len(sample)), *sample))
        # Write random seed always (see changes to TSNE::load_data(...))
        data_file.write(pack('i', randseed))
        # Write initial solution if passed
        if initial_solution is not None:
            for sample in initial_solution:
                data_file.write(pack('{}d'.format(len(sample)), *sample))


def load_data(input_file):
    # Read the data, using numpy's good judgement
    return np.loadtxt(input_file)


def bh_tsne(workdir, verbose=False):

    # Call bh_tsne and let it do its thing
    with open(devnull, 'w') as dev_null:
        bh_tsne_p = Popen((abspath(BH_TSNE_BIN_PATH), ), cwd=workdir, universal_newlines=True, shell=False,
                          stdout=PIPE if verbose else dev_null)

        # process stdout and print: redirect to print scan streamlogger
        for line in iter(bh_tsne_p.stdout.readline, ""):
            print(line)
        bh_tsne_p.stdout.close()
        bh_tsne_p.wait()
        assert not bh_tsne_p.returncode, ('ERROR: Call to bh_tsne exited '
                                          'with a non-zero return code exit status, please ' +
                                          ('enable verbose mode and ' if not verbose else '') +
                                          'refer to the bh_tsne output for further details')


def result_reader(result_file):

    # Read and pass on the results
    with open(result_file, 'rb') as output_file:

        # First integer is the number of iterations
        num_iteration = _read_unpack('i', output_file)

        # Integers two and three are just the number of samples and the
        #   dimensionality
        result_samples, result_dims = _read_unpack('ii', output_file)

        # Collect the results, but they may be out of order
        results = [_read_unpack('{}d'.format(result_dims), output_file) for _ in range(result_samples)]

        # Now collect the landmark data so that we can return the data in
        #   the order it arrived
        results = [(_read_unpack('i', output_file), e) for e in results]

        # Lastly, add the costs to each observation
        results = [(l, (e + _read_unpack('d', output_file))) for l, e in results]

        # Put the results in order
        results.sort()

        res = []
        for _, result in results:
            sample_res = []
            # needed to be flexible to number of dimensions
            for r in result:
                sample_res.append(r)
            res.append(sample_res)

        # returns a dict containing the number of the iteration and the actual data with the costs as last column
        return {num_iteration: np.asarray(res, dtype='float64')}


def run_bh_tsne(data, no_dims=DEFAULT_NO_DIMS, perplexity=DEFAULT_PERPLEXITY, theta=DEFAULT_THETA,
                learning_rate=DEFAULT_LEARNING_RATE, momentum=DEFAULT_MOMENTUM, final_momentum=DEFAULT_FINAL_MOMENTUM,
                initial_dims=INITIAL_DIMENSIONS, use_pca=DEFAULT_USE_PCA, max_iter=DEFAULT_MAX_ITERATIONS,
                stop_lying_iter=DEFAULT_STOP_LYING_ITERATION, restart_lying_iter=DEFAULT_RESTART_LYING_ITERATION,
                momentum_switch_iter=DEFAULT_MOMENTUM_SWITCH_ITERATION, lying_factor=DEFAULT_EXAGGERATION_FACTOR,
                randseed=-1, initial_solution=None, verbose=False, input_similarities=DEFAULT_BUILDING_BLOCK_INDEX,
                output_similarities=DEFAULT_BUILDING_BLOCK_INDEX, cost_function=DEFAULT_BUILDING_BLOCK_INDEX,
                optimization=DEFAULT_BUILDING_BLOCK_INDEX, freeze_index=DEFAULT_FREEZE_INDEX):
    """
    Run TSNE based on the Barnes-HT algorithm

    Parameters:
    ----------
    data: file or numpy.array
        The data used to run TSNE, one sample per row
    no_dims: int
    perplexity: int
    randseed: int
    theta: float
    initial_dims: int
    verbose: boolean
    use_pca: boolean
    max_iter: int
    """

    # bh_tsne works with fixed input and output paths, give it a temporary
    #   directory to work in so we don't clutter the filesystem
    tmp_dir_path = mkdtemp()

    # distinguish between windows that does not support os.fork() and any other OS
    if IS_WINDOWS:

        # for windows: run initialization immediately and do not load data into forked process
        init_bh_tsne(data, tmp_dir_path, no_dims=no_dims, initial_dims=initial_dims, initial_solution=initial_solution,
                     perplexity=perplexity, learning_rate=learning_rate, momentum=momentum,
                     final_momentum=final_momentum, theta=theta, use_pca=use_pca, max_iter=max_iter,
                     stop_lying_iter=stop_lying_iter, restart_lying_iter=restart_lying_iter,
                     momentum_switch_iter=momentum_switch_iter, lying_factor=lying_factor, randseed=randseed,
                     input_similarities=input_similarities, output_similarities=output_similarities,
                     cost_function=cost_function, optimization=optimization, freeze_index=freeze_index)

    else:
        # for linux: do all the linux stuff in child process
        # Load data in forked process to free memory for actual bh_tsne calculation
        child_pid = os.fork()
        if child_pid == 0:
            if _is_filelike_object(data):
                data = load_data(data)

            init_bh_tsne(data, tmp_dir_path, no_dims=no_dims, initial_dims=initial_dims,
                         initial_solution=initial_solution, perplexity=perplexity, learning_rate=learning_rate,
                         momentum=momentum, final_momentum=final_momentum, theta=theta, use_pca=use_pca,
                         max_iter=max_iter, stop_lying_iter=stop_lying_iter, restart_lying_iter=restart_lying_iter,
                         momentum_switch_iter=momentum_switch_iter, lying_factor=lying_factor, randseed=randseed,
                         input_similarities=input_similarities, output_similarities=output_similarities,
                         cost_function=cost_function, optimization=optimization, freeze_index=freeze_index)
            os._exit(0)
        else:
            try:
                os.waitpid(child_pid, 0)
            except KeyboardInterrupt:
                print("Please run this program directly from python and not from ipython or jupyter.")
                print("This is an issue due to asynchronous error handling.")

    # executes the actual bhtsne algorithm
    bh_tsne(tmp_dir_path, verbose)

    # load result files into single python dict object
    files = [f for f in glob.glob(path_join(tmp_dir_path, "result*"))]

    # build final result dict:
    #{ 1: nparray[...]
    #  50: nparray[...]
    # ...
    #  1000: nparray[...]

    bh_tsne_result = {}
    for result_file in files:
        bh_tsne_result.update(result_reader(result_file))

    # cleanup temp directory
    rmtree(tmp_dir_path)

    # return final dict
    return bh_tsne_result


def write_bh_tsne_result(bh_tsne_result_dict, directory, sep='-', *filename_extensions):
    filename = "bh_tsne_result-" + sep.join(filename_extensions) + '.pickle'
    # format to abspath
    file_abspath = path_join(directory, filename)

    try:
        os.makedirs(directory)
    except FileExistsError:
        # directory already exists
        pass

    with open(file_abspath, 'wb') as pickle_file:
        pickle.dump(bh_tsne_result_dict, pickle_file)


def read_bh_tsne_result(file_abspath):
    with open(file_abspath, 'rb') as pickle_file:
        return pickle.load(pickle_file)


#######################################################################################################################
#                                               DEBUG CODE                                                            #
#######################################################################################################################

def debug_bh_tsne_pre(data, data_name):
    """
    debug TSNE pre: just write the data matrix into directory windows for windows execution
    """

    tmp_dir_path = os.path.abspath(path_join(os.path.dirname(__file__), "windows",))

    _initial_embedding = get_initial_embedding(data_name=data_name,
                                               method_name="gaussian", i=1)

    init_bh_tsne(data, tmp_dir_path, no_dims=DEFAULT_NO_DIMS, initial_dims=INITIAL_DIMENSIONS,
                 initial_solution=_initial_embedding,
                 perplexity=20, learning_rate=DEFAULT_LEARNING_RATE, momentum=DEFAULT_MOMENTUM,
                 final_momentum=DEFAULT_FINAL_MOMENTUM, theta=0.5, randseed=EMPTY_SEED,
                 use_pca=DEFAULT_USE_PCA, max_iter=DEFAULT_MAX_ITERATIONS, stop_lying_iter=DEFAULT_STOP_LYING_ITERATION,
                 restart_lying_iter=DEFAULT_RESTART_LYING_ITERATION,
                 momentum_switch_iter=DEFAULT_MOMENTUM_SWITCH_ITERATION, lying_factor=1,
                 input_similarities=DEFAULT_BUILDING_BLOCK_INDEX, output_similarities=DEFAULT_BUILDING_BLOCK_INDEX,
                 cost_function=DEFAULT_BUILDING_BLOCK_INDEX, optimization=1)


def debug_data_file(workdir, sample_count, len_sample):

    with open(path_join(workdir, 'data.dat'), 'rb') as data_file:
        # Write the bh_tsne header

        sample_count, sample_dim = _read_unpack('ii', data_file)
        theta, perplexity, learning_rate, momentum, final_momentum = _read_unpack('ddddd', data_file)
        no_dims, max_iter, stop_lying_iter, restart_lying_iter, momentum_switch_iter, \
        lying_factor = _read_unpack('iiiiii', data_file)

        results = [_read_unpack('{}d'.format(sample_dim), data_file) for _ in range(sample_count)]

        randseed = _read_unpack('i', data_file)

        while True:
            read = _read_unpack('i', data_file)


def debug_bh_tsne_post():
    """
    Do not execute bh_tsne, just read result.dat
    :return:
    """
    debug_dir_path = os.path.abspath(path_join(os.path.dirname(__file__), "windows", ))
    #debug_dir_path = "C:\\Users\\Tobi\\git\\bhtsne\\results\\BHtSNE\\buildingblocks\\cost_function\\RKL\\iterations\\1000\\fashion_mnist\\max_iter\\1000"

    # load result files into single python dict object
    files = [f for f in glob.glob(path_join(debug_dir_path, "result*"))]

    # build final result dict:
    # { 1: nparray[...]
    #  50: nparray[...]
    # ...
    #  1000: nparray[...]

    bh_tsne_result = {}
    for result_file in files:
        bh_tsne_result.update(result_reader(result_file))

    # return final dict
    return bh_tsne_result


def get_axlims(series, marginfactor):
    """
    Fix for a scaling issue with matplotlibs scatterplot and small values.
    Takes in a pandas series, and a marginfactor (float).
    A marginfactor of 0.2 would for example set a 20% border distance on both sides.
    Output:[bottom,top]
    To be used with .set_ylim(bottom,top)
    """
    minv = series.min()
    maxv = series.max()
    datarange = maxv-minv
    border = abs(datarange*marginfactor)
    maxlim = maxv+border
    minlim = minv-border

    return minlim, maxlim


def plot_bh_tsne_post(embedding_dict, labels):

    import seaborn as sns
    import matplotlib.pyplot as plt
    for key in embedding_dict.keys():


        x_min, x_max = get_axlims(embedding_dict[key][:, 0], .1)
        y_min, y_max = get_axlims(embedding_dict[key][:, 1], .1)

        sns.despine()
        sns.set_style("white")
        g = sns.scatterplot(x=embedding_dict[key][:, 0],
                            y=embedding_dict[key][:, 1],
                            hue=labels,
                            legend="full",
                            palette=sns.color_palette("bright"))

        plt.gca().set_xlim(x_min, x_max)
        plt.gca().set_ylim(y_min, y_max)

        figure = g.get_figure()
        figure.savefig(os.path.join("windows", "tsne-debug" + str(key[0])) + ".png", bbox_inches="tight")
        plt.close(figure)


def main(args):
    parser = _argparse()

    if len(args) <= 1:
        print(parser.print_help())
        return 

    argp = parser.parse_args(args[1:])

    for result in run_bh_tsne(argp.input, no_dims=argp.no_dims, perplexity=argp.perplexity,
                              theta=argp.theta, randseed=argp.randseed, verbose=argp.verbose,
                              initial_dims=argp.initial_dims, use_pca=argp.use_pca, max_iter=argp.max_iter):
        fmt = ''
        for i in range(1, len(result)):
            fmt = fmt + '{}\t'
        fmt = fmt + '{}\n'
        argp.output.write(fmt.format(*result))


if __name__ == '__main__':
    from sys import argv
    exit(main(argv))

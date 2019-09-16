#!/usr/bin/env python

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import seaborn as sns
import os
import bhtsne
import mnist
import numpy as np
import glob
import itertools
import operator

FASHION_LABELDICT = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}

# directory structure
CWD = os.path.dirname(os.path.realpath(__file__))
RESULT_DIR = os.path.join("I:", "MasterThesis", "Experimental Results") if os.path.exists("I:") \
    else os.path.join(CWD, "results")
PLOT_DIR = os.path.join("I:", "MasterThesis", "Plots") if os.path.exists("I:") \
    else os.path.join(CWD, "plots")


def get_bh_tsne_grouped_result_generator(root_dir=RESULT_DIR, data_identifier='mnist'):

    files = [f for f in glob.glob(os.path.join(root_dir, "**/*.pickle"), recursive=True)]

    # filter for paths that actually include the desired data
    print("filtering for data: {}".format(data_identifier))

    files = list(filter(lambda x: data_identifier in str(x).split(os.path.sep), files))

    # sort list
    # essential for grouping
    files.sort()

    files_tuples = [(str(x).split(os.path.sep)[-5] + str(x).split(os.path.sep)[-4], x) for x in files]
    files_tuples.sort()

    for _key, _grouper in itertools.groupby(files_tuples, operator.itemgetter(0)):
        yield _key, list(_grouper)


def plot_bh_tsne_result(_data, _labels, _legend="full", rearrange_legend=False, _palette="bright", _ax=None,
                        axes_off=True, alpha=1.0, markers="o", zorder=1, linewidth=0.5):
    #plt.box(False)
    sns.despine()
    sns.set_style("white")

    palette = sns.color_palette(_palette, n_colors=10)
    color_dict = {FASHION_LABELDICT[i]: palette[i] for i in FASHION_LABELDICT.keys()}

    _g = sns.scatterplot(x=_data[:, 0],
                         y=_data[:, 1],
                         hue=_labels,
                         style=np.zeros(len(_data[:, 0])),
                         alpha=alpha,
                         legend=_legend,
                         palette=color_dict,
                         markers=[markers],
                         zorder=zorder,
                         linewidth=linewidth,
                         ax=_ax)

    if _legend is not None:

        _handles, _labels = _ax.get_legend_handles_labels()

        if rearrange_legend:
            new_order = [1, 6, 3, 2, 8, 5, 7, 4, 9, 0]

            _handles = [_handles[i] for i in new_order]
            _labels = [_labels[i] for i in new_order]

        _ax.legend(_handles, _labels, loc=3, prop=dict(size=15))
        #_ax.legend()

    #_g.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1)
    #_g.legend.set_title("Classes")

    x_min, x_max = get_axlims(_data[:, 0], .1)
    y_min, y_max = get_axlims(_data[:, 1], .1)

    if _ax is None:
        plt.gca().set_xlim(x_min, x_max)
        plt.gca().set_ylim(y_min, y_max)
    else:
        _ax.set_xlim(x_min, x_max)
        _ax.set_ylim(y_min, y_max)

    if axes_off:
        if _ax is None:
            plt.gca().axis('off')
        else:
            _ax.axis('off')

    return _g


def load_result_and_plot_comparison(_labels, root_dir=RESULT_DIR, data_identifier="mnist",
                                    plot_title_from_filepath_index=0):

    for _paramvalue, _file_list in get_bh_tsne_grouped_result_generator(root_dir=root_dir,
                                                                        data_identifier=data_identifier):
        print("Creating plot for data {} with parameter {}".format(data_identifier, _paramvalue))
        _result_list = [bhtsne.read_bh_tsne_result(_file) for _k, _file in _file_list]
        # make titles based on file path
        _title_list = [str(_file).split(os.path.sep)[plot_title_from_filepath_index]
                       if plot_title_from_filepath_index < 0 else ""
                       for _k, _file in _file_list]
        _paramvalue = _paramvalue.replace(".", "-")
        _dir = os.path.join(PLOT_DIR, _paramvalue, data_identifier)
        try:
            os.makedirs(_dir)
        except FileExistsError:
            # directory already exists
            pass

        # assuming that the keys of the first dict in list represent all dicts' keys
        for _key in _result_list[0].keys():
            print("Creating plot for iteration {}".format(_key[0]))
            _data_list = [result[_key] for result in _result_list]
            _fig = compare_n_results(_labels=_labels, _data_list=_data_list, _title_list=_title_list)

            save_figure(_fig, _dir, "-", _paramvalue, str(_key[0]))
            plt.close(_fig)




def compare_n_results(_labels, _data_list, _title_list, _size=8):
    """

    :param _size: default width and height of single plot
    :param _data_list: list of tuples of type (data, label)
    :return: the figure created
    """
    mpl.rcParams['xtick.labelsize'] = _size * 2
    mpl.rcParams['ytick.labelsize'] = _size * 2
    mpl.rcParams['axes.titlesize'] = _size * 3

    if len(_data_list) <= 5:
        _nrows = int((len(_data_list) + 1) / 2)
        _ncols = 2
    else:
        _nrows = int((len(_data_list) + 1) / 3)
        _ncols = 3

    fig_size = (_size * _ncols, _size * _nrows)

    _fig, _axs = plt.subplots(ncols=_ncols, nrows=_nrows, figsize=fig_size)
    #_fig, _axs = plt.subplots(len(_data_list), figsize=fig_size)

    for idx, _data in enumerate(_data_list):

        i = int(idx / _ncols)
        j = idx % _ncols

        plot_bh_tsne_result(_data, _labels, _legend="full", _ax=_axs[i, j])
        _axs[i, j].set_title("{} Cost: {}".format(_title_list[idx], np.sum(_data[:, 2])))
    # retrieve legend
    _handles, _labels = _axs[0, 0].get_legend_handles_labels()

    # remove legends of idividual subplots

    for idx, _data in enumerate(_data_list):
        i = int(idx / _ncols)
        j = idx % _ncols
        _axs[i, j].get_legend().remove()

    # plot legend if uneven number of plots into last axes
    if len(_data_list) % 2 == 1:
        _axs[_nrows - 1, _ncols - 1].legend(_handles, _labels, loc="center", prop={'size': _size * 4}, markerscale=3)
        _axs[_nrows - 1, _ncols - 1].axis("off")
    else:
        _fig.legend(_handles, _labels, loc="lower center", prop={'size': _size * 4}, markerscale=3)#, bbox_to_anchor=(1.14, 1))
    _fig.tight_layout()

    return _fig


def compare_two_results(_data1, _label1, _data2, _label2, figsize=(10, 5)):
    _fig, _axs = plt.subplots(ncols=2, gridspec_kw={'width_ratios': [1, 1]}, figsize=figsize)
    plot_bh_tsne_result(_data1, _label1, _legend="full", _ax=_axs[0])
    _axs[0].set_title("Cost: {}".format(np.sum(_data1[:, 2])))
    plot_bh_tsne_result(_data2, _label2, _legend="full", _ax=_axs[1])
    _axs[1].set_title("Cost: {}".format(np.sum(_data2[:, 2])))
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


#def plot_images():
#    for image, (x, y) in zip(inputs, coords):
#        im = OffsetImage(image.reshape(28, 28), zoom=1, cmap='gray')
#        ab = AnnotationBbox(im, (x, y), xycoords='data', frameon=False)
#        ax.add_artist(ab)
#    ax.update_datalim(coords)
#    ax.autoscale()

def load_result_and_plot_single(_input_data, _labels, path_to_bh_tsne_result, data_filter="fashion_mnist7000",
                                mode=0, legend=None, rearrange_legend=False, max_iter_only=False):

    # assuming only a single result file
    files = glob.glob(os.path.join(path_to_bh_tsne_result, "**/*.pickle"), recursive=True)

    files = list(filter(lambda x: data_filter in str(x).split(os.path.sep)
                        #and
                        #(#"perplexity" in str(x).split(os.path.sep)
                         #or
                         #"iterations" in str(x).split(os.path.sep)
                         #)
                         , files))


    # for now, focus on run 1
    #files = list(filter(lambda x: str(1) == str(x).split(os.path.sep)[-2], files))

    # for fashion_mnist= adjust labels
    fashion_labels = [FASHION_LABELDICT[l] for l in _labels]

    for file in files:
        embedding_dict = bhtsne.read_bh_tsne_result(file)
        for key in embedding_dict.keys():
            if not max_iter_only or key[0] == 1000 or key[0] == 0:
                data = embedding_dict[key]

                dirname = os.path.dirname(file)
                dirname = dirname.replace("results", "plots\\single")
                try:
                    os.makedirs(dirname)
                except FileExistsError:
                    # directory already exists
                    pass

                if mode == 0 or mode == 1:
                    fig, ax = plt.subplots(figsize=(8, 8))

                    plot_bh_tsne_result(_data=data, _labels=fashion_labels, _legend="full" if key[0] == 10 else legend,
                                        rearrange_legend=rearrange_legend, _palette="bright", _ax=ax, axes_off=True)
                    filename = os.path.splitext(file)[0] + "-iteration{}".format(str(key[0]))
                    filename = filename.replace("results", "plots\\single")

                    plt.savefig(filename, bbox_inches="tight")

                    plt.close()

                if mode == 0 or mode == 2:
                    fig, ax = plt.subplots(figsize=(8, 8))
                    # just a sample of 300 as images
                    _sample_input_data, _sample_labels = mnist.sample_data(_input_data, _labels, num_samples=1000)
                    _sample_embedding_data, _ = mnist.sample_data(data, _labels, num_samples=1000)
                    color_palette = sns.color_palette("bright", n_colors=10)
                    _sample_labels = _sample_labels.astype(int)

                    for image, (x, y), label in zip(_sample_input_data, _sample_embedding_data[:, 0:2], _sample_labels):
                        im = OffsetImage(image.reshape(28, 28), zoom=1, cmap='gray')
                        ab = AnnotationBbox(im, (x, y), xycoords='data', frameon=True, pad=.0,
                                            bboxprops=dict(edgecolor=color_palette[label], facecolor=color_palette[label]))
                        ax.add_artist(ab)
                    ax.update_datalim(data[:, 0:2])
                    ax.autoscale()
                    ax.axis('off')

                    filename = os.path.splitext(file)[0] + "-iteration{}-images".format(str(key[0]))
                    filename = filename.replace("results", "plots\\single")
                    plt.savefig(filename, bbox_inches="tight")

                    plt.close()


def load_result_and_plot_single_freeze(x_train, x_test, y_train, y_test, path_to_bh_tsne_result, data_filter="fashion_mnist7000",
                                       legend=None, rearrange_legend=False, max_iter_only=False):

    # assuming only a single result file
    files = glob.glob(os.path.join(path_to_bh_tsne_result, "**/*.pickle"), recursive=True)

    files = list(filter(lambda x: data_filter in str(x).split(os.path.sep), files))


    # for now, focus on run 1
    #files = list(filter(lambda x: str(1) == str(x).split(os.path.sep)[-2], files))

    # for fashion_mnist= adjust labels
    _y_train = [FASHION_LABELDICT[l] for l in y_train]
    _y_test = [FASHION_LABELDICT[l] for l in y_test]

    for file in files:
        embedding_dict = bhtsne.read_bh_tsne_result(file)
        for key in embedding_dict.keys():
            if not max_iter_only or key[0] == 1000 or key[0] == 0:
                x_embedded_train = embedding_dict[key][:len(x_train),:2]
                x_embedded_test = embedding_dict[key][len(x_train):,:2]

                dirname = os.path.dirname(file)
                dirname = dirname.replace("results", "plots\\single")
                try:
                    os.makedirs(dirname)
                except FileExistsError:
                    # directory already exists
                    pass


                fig, ax = plt.subplots(figsize=(8, 8))

                plot_bh_tsne_result(_data=x_embedded_train, _labels=_y_train, _legend="full" if key[0] == 10 else legend,
                                    rearrange_legend=rearrange_legend, _palette="bright", _ax=ax, axes_off=True, alpha=0.15)

                plot_bh_tsne_result(_data=x_embedded_test, _labels=_y_test,
                                    _legend="full" if key[0] == 10 else legend,
                                    rearrange_legend=rearrange_legend, _palette="bright", _ax=ax, axes_off=True,
                                    alpha=1, markers="P", zorder=20, linewidth=0.5)


                filename = os.path.splitext(file)[0] + "-iteration{}".format(str(key[0]))
                filename = filename.replace("results", "plots\\single")

                plt.savefig(filename, bbox_inches="tight")

                plt.close()






if __name__ == "__main__":

    #data, labels = mnist.load_fashion_mnist_data(False, len_sample=7000)
    #data, labels = mnist.load_fashion_mnist_data(True)

    #load_result_and_plot_comparison(_labels=labels, root_dir=os.path.join(RESULT_DIR, "tSNE", "parametertuning",
    #                                                                      "output_similarities", "studentalpha"),
    #                                plot_title_from_filepath_index=-6, data_identifier="fashion_mnist")

    # load_result_and_plot_single(data, labels,
    #                             "C:\\Users\\Tobi\\git\\bhtsne\\results\\BHtSNE\\buildingblocks\\initial_embeddings",
    #                             data_filter="fashion_mnist",
    #                             mode=1, legend=None, rearrange_legend=True, max_iter_only=True)

    #data, labels = mnist.load_fashion_mnist_data(False, len_sample=7000)

    #load_result_and_plot_single(data, labels,
    #                            "C:\\Users\\Tobi\\git\\bhtsne\\results\\BHtSNE\\buildingblocks\\cost_function\\JS",
    #                            data_filter="fashion_mnist7000",
    #                            mode=1, legend=None, rearrange_legend=False, max_iter_only=False)

    # freeze index

    (x_train, x_test), (y_train, y_test) = mnist.load_fashion_mnist_data(False, len_sample=2500, train_test_split=2000)

    #data = np.vstack((x_train, x_train))
    #labels = np.concatenate((y_train, y_test))

    load_result_and_plot_single_freeze(x_train, x_test, y_train, y_test,
                                       "C:\\Users\\Tobi\\git\\bhtsne\\results\\BHtSNE\\freeze_index",
                                       data_filter="fashion_mnist2500",
                                       legend=None, rearrange_legend=False, max_iter_only=True)

    # basepath1 = "C:\\Users\\Tobi\\Documents\\SS_19\\Master Thesis\\04 - Experiment Results\\MNIST\\base\\unoptimized sptree\\1"
    # basepath2 = "C:\\Users\\Tobi\\Documents\\SS_19\\Master Thesis\\04 - Experiment Results\\MNIST\\base\\optimized sptree\\1"
    # benchmark1 = bhtsne.read_bh_tsne_result(os.path.join(basepath1, "bh_tsne_result-08-07-2019_22-27-37.pickle"))
    # benchmark2 = bhtsne.read_bh_tsne_result(os.path.join(basepath2, "bh_tsne_result-09-07-2019_17-58-55.pickle"))
    #
    # for key in benchmark1.keys():
    #     print(str(key[0]))
    #     mnist_benchmark1 = np.hstack((labels[:, None], benchmark1[key]))
    #     mnist_benchmark2 = np.hstack((labels[:, None], benchmark2[key]))
    #
    #     fig = compare_two_results(mnist_benchmark1[:, 1:4], mnist_benchmark1[:, 0],
    #                               mnist_benchmark2[:, 1:4], mnist_benchmark2[:, 0])
    #     save_figure(fig, PLOT_DIR, "-", "testcompare3", str(key[0]))
    #     plt.close(fig)

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
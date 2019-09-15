#!/usr/bin/env python

import matplotlib.pyplot as plt
from functools import partial
import math
import numpy as np
from mnist import load_fashion_mnist_data
import scipy.stats as stats
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib as mpl
import matplotlib.ticker as ticker
import matplotlib.patches as patches
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import matplotlib.lines as mlines
import json

blue = (57 / 256, 106 / 256, 177 / 256)
orange = (218 / 256, 124 / 256, 48 / 256)
green = (132 / 256, 186 / 256, 91 / 256)
red = (211 / 256, 94 / 256, 96 / 256)
grey = (83 / 256, 81 / 256, 84 / 256)
ercis = (146 / 256, 36 / 256, 40 / 256)


def find_x_given_y(x_lim_low, x_lim_high, y_value, y_function):
    x = np.linspace(x_lim_low, x_lim_high, 10000)
    y = y_function(x)

    rev_x = x[::-1]
    rev_y = y[::-1]

    return np.interp(y_value, rev_y, rev_x)


def draw_vector(v0, v1, arrow=True, color=grey, ax=None):
    ax = ax or plt.gca()
    arrowprops = dict(arrowstyle='->' if arrow else '-',
                      linewidth=2, ls='-' if arrow else '--',
                      shrinkA=0, shrinkB=0, color=color)
    ax.annotate('', v1, v0, arrowprops=arrowprops)
    #ax.arrow(v0[0], v0[1], v1[0], v1[1], **arrowprops)


def plot_lle():
    sns.set()
    sns.set_style("whitegrid")

    rng = np.random.RandomState(1)
    X = np.dot(rng.rand(2, 2), rng.randn(2, 20)).T
    print(X)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(X[:, 0], X[:, 1], alpha=0.4)
    nN = NearestNeighbors(n_neighbors=6, algorithm='ball_tree').fit(X)
    _, indices = nN.kneighbors(X)
    indices = indices[6][1:]
    ax.scatter(X[indices, 0], X[indices, 1], alpha=0.4)
    ax.scatter(X[6, 0], X[6, 1], alpha=0.4)
    ax.axis('equal')
    plt.show()

def plot_lda():
    sns.set()
    sns.set_style("whitegrid")

    rng = np.random.RandomState(1)
    X = np.dot(rng.rand(2, 2), rng.randn(2, 200)).T
    labels = [0 if x[0] < 0 else 1 for x in X]
    X_pre = np.hstack((X, np.reshape(labels, (-1, 1))))

    g = sns.JointGrid(X[:, 0], X[:, 1], xlim=(-2.9,2.9), ylim=(-2.9,2.9), height=8)
    for label in [0, 1]:
        col = grey if label == 0 else ercis
        sns.distplot(X_pre[X_pre[:, 2] == label, 0], kde=False, ax=g.ax_marg_x, color=col)
        sns.distplot(X_pre[X_pre[:, 2] == label, 1], kde=False, ax=g.ax_marg_y, vertical=True, color=col)
        #sns.kdeplot(X_pre[X_pre[:, 2] == label, 0], ax=g.ax_marg_x, legend=False, shade=True)
        #sns.kdeplot(X_pre[X_pre[:, 2] == label, 1], ax=g.ax_marg_y, vertical=True, legend=False, shade=True)
        g.ax_joint.plot(X_pre[X_pre[:, 2] == label, 0], X_pre[X_pre[:, 2] == label, 1], "o", ms=5, color=col,
                        label="Class {}".format(str(label)), alpha=0.4)

    g.ax_joint.set(xlabel='x', ylabel='y')
    g.ax_joint.xaxis.label.set_size(15)
    g.ax_joint.yaxis.label.set_size(15)

    g.ax_marg_x.grid(color="w")
    g.ax_marg_x.spines['bottom'].set_color('w')
    g.ax_marg_x.spines['top'].set_color('w')
    g.ax_marg_x.spines['right'].set_color('w')
    g.ax_marg_x.spines['left'].set_color('w')

    g.ax_marg_y.grid(color="w")
    g.ax_marg_y.spines['bottom'].set_color('w')
    g.ax_marg_y.spines['top'].set_color('w')
    g.ax_marg_y.spines['right'].set_color('w')
    g.ax_marg_y.spines['left'].set_color('w')

    g.ax_joint.legend(fontsize=15)

    sns.despine(left=True, right=True, bottom=True, top=True)
    plt.savefig("lda_split", bbox_inches="tight")
    plt.show()


def plot_pca_new_directions():
    sns.set()
    sns.set_style("whitegrid")
    rng = np.random.RandomState(1)
    X = np.dot(rng.rand(2, 2), rng.randn(2, 200)).T
    pca = PCA(n_components=2, whiten=True)
    pca.fit(X)

    fig, ax = plt.subplots(figsize=(8, 8))

    # plot data
    ax.scatter(X[:, 0], X[:, 1], alpha=0.4)
    for i, (length, vector) in enumerate(zip(pca.explained_variance_, pca.components_)):
        v = vector * 3 * np.sqrt(length)
        draw_vector(pca.mean_, pca.mean_ + v, color=ercis if i == 1 else grey, ax=ax)
        draw_vector(pca.mean_, pca.mean_ - v, color=ercis if i == 1 else grey, arrow=False, ax=ax)
    ax.axis('equal')
    ax.set(xlabel='x', ylabel='y')
    ax.xaxis.label.set_size(15)
    ax.yaxis.label.set_size(15)

    grey_arrow_head = mlines.Line2D([], [], color=grey, linestyle='', marker='<', markersize=5)
    grey_arrow_tail = mlines.Line2D([], [], color=grey, marker='', linestyle='-', markersize=15)
    grey_arrow_tail_tail = mlines.Line2D([], [], color=grey, marker='', linestyle='--', markersize=15)
    ercis_arrow_head = mlines.Line2D([], [], color=ercis, linestyle='', marker='<', markersize=5)
    ercis_arrow_tail = mlines.Line2D([], [], color=ercis, marker='', linestyle='-', markersize=15)
    ercis_arrow_tail_tail = mlines.Line2D([], [], color=ercis, marker='', linestyle='--', markersize=15)

    handles = [(grey_arrow_head, grey_arrow_tail, grey_arrow_tail_tail),
               (ercis_arrow_head, ercis_arrow_tail, ercis_arrow_tail_tail)]
    labels = ['Principal component 1', 'Principal component 2']

    ax.legend(handles=handles, labels=labels, loc='upper left',
              handler_map={tuple: mpl.legend_handler.HandlerTuple(None)}, fontsize=15)

    sns.despine(left=True, right=True, bottom=True, top=True)
    plt.savefig("pca_components_before", bbox_inches="tight")
    plt.show()
    plt.close(fig)

    #fig, ax = plt.subplots(figsize=(8, 8))

    # plot principal components
    X_pca = pca.transform(X)

    sns.set(rc={'figure.figsize': (8, 8)})
    sns.set_style("whitegrid")
    g = sns.JointGrid(X[:, 0], X[:, 1], xlim=(-3.1, 3.1), ylim=(-3.1, 3.1), height=8, space=0, ratio=16)
    g.ax_joint.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.4, label="PCA transformed data")

    g.ax_marg_x.scatter(X_pca[:, 0], [0] * X_pca.shape[0], alpha=0.4, c=grey, label='Projection on PC 1')
    g.ax_marg_x.legend()
    handles1, labels1 = g.ax_marg_x.get_legend_handles_labels()
    g.ax_marg_x.get_legend().remove()
    g.ax_marg_y.scatter([0] * X_pca.shape[0], X_pca[:, 1], alpha=0.4, c=ercis, label='Projection on PC 2')
    g.ax_marg_y.legend()
    handles2, labels2 = g.ax_marg_y.get_legend_handles_labels()
    g.ax_marg_y.get_legend().remove()

    handles1.extend(handles2)
    labels1.extend(labels2)

    draw_vector([0, 0], [0, 3], color=ercis, ax=g.ax_joint)
    draw_vector([0, 0], [0, -3], color=ercis, arrow=False, ax=g.ax_joint)
    draw_vector([0, 0], [3, 0], ax=g.ax_joint)
    draw_vector([0, 0], [-3, 0], arrow=False, ax=g.ax_joint)

    #ax.scatter(X_pca[:, 0], [0] * X_pca.shape[0], alpha=0.4, c=grey, label='Projection on PC 1')
    #ax.scatter([0] * X_pca.shape[0], X_pca[:, 1], alpha=0.4, c=ercis, label='Projection on PC 2')

    exp_var1 = round(pca.explained_variance_ratio_[0], 3)
    exp_var2 = round(pca.explained_variance_ratio_[1], 3)
    #ax.axis('equal')
    g.ax_joint.set(xlabel='Principal component 1 (explained variance ratio: {})'.format(str(exp_var1)),
           ylabel='Principal component 2 (explained variance ratio: {})'.format(str(exp_var2)),)
           #xlim=(-3.21, 3.2), ylim=(-3.21, 3.2))
    g.ax_joint.xaxis.label.set_size(15)
    g.ax_joint.yaxis.label.set_size(15)

    g.ax_joint.legend()
    handles, labels = g.ax_joint.get_legend_handles_labels()
    g.ax_joint.legend().remove()

    handles.extend(handles1)
    labels.extend(labels1)

    g.ax_joint.legend(handles=handles, labels=labels, loc='upper right', fontsize=15)

    g.ax_marg_x.grid(color="w")
    g.ax_marg_x.spines['bottom'].set_color('w')
    g.ax_marg_x.spines['top'].set_color('w')
    g.ax_marg_x.spines['right'].set_color('w')
    g.ax_marg_x.spines['left'].set_color('w')

    g.ax_marg_y.grid(color="w")
    g.ax_marg_y.spines['bottom'].set_color('w')
    g.ax_marg_y.spines['top'].set_color('w')
    g.ax_marg_y.spines['right'].set_color('w')
    g.ax_marg_y.spines['left'].set_color('w')

    sns.despine(left=True, right=True, bottom=True, top=True)
    plt.savefig("pca_components_after", bbox_inches="tight")
    plt.show()


def get_intersection(f, g):
    return np.argwhere(np.diff(np.sign(f - g))).flatten()


def plot_gaussian_student_distance_shift(x_lim_low, x_lim_high, *x_values, alpha=1, high_dim_pdf="gaussian", low_dim_pdf="student1"):
    mu = 0
    variance = 1
    sigma = math.sqrt(variance)
    x = np.linspace(x_lim_low, mu + x_lim_high * sigma, 10000)

    if high_dim_pdf == "gaussian":
        hfunc = partial(stats.norm.pdf, loc=mu, scale=sigma)
        hlabel = "Gaussian"
        htitle = "gaussian"
    elif high_dim_pdf == "laplacian":
        hfunc = partial(stats.laplace.pdf, loc=mu, scale=sigma)
        hlabel = "Laplace"
        htitle = "laplacian"
    elif high_dim_pdf == "student50":
        hfunc = partial(stats.t.pdf, df=50)
        hlabel = "Student-t ($\\alpha=50$)"
        htitle = "student50"
    else:
        hfunc = partial(stats.norm.pdf, loc=mu, scale=sigma)
        hlabel = "Gaussian"
        htitle = "gaussian"

    if low_dim_pdf == "chi":
        lfunc = partial(stats.chi2.pdf, df=2)
        llabel = "Chi-squared"
        ltitle = "chi2"
    elif low_dim_pdf == "student05":
        lfunc = partial(stats.t.pdf, df=0.5)
        llabel = "Student-t ($\\alpha=0.5$)"
        ltitle = "student05"
    elif low_dim_pdf == "student01":
        lfunc = partial(stats.t.pdf, df=0.1)
        llabel = "Student-t ($\\alpha=0.1$)"
        ltitle = "student01"
    else:
        lfunc = partial(stats.t.pdf, df=1)
        llabel = "Student-t ($\\alpha=1$)"
        ltitle = "student1"

    idx = get_intersection(alpha * hfunc(x), 0.5 * lfunc(x))[0]


    y_min = -0.019945552964581167
    y_max = 0.4188893200855286

    y_range = y_max-y_min
    y_min = y_min + y_range * 0.05
    y_max = y_max - y_range * 0.05

    x_1 = x[:idx]
    x_2 = x[idx:]

    sns.set(style="whitegrid")

    marker_adjustment = .03

    plt.figure(figsize=(10, 5))
    plt.plot(x, alpha * hfunc(x), linestyle="-", color=blue, linewidth=1.5, label="{} distribution".format(hlabel))
    plt.plot(x, 0.5 * lfunc(x), linestyle="-", color=orange, linewidth=1.5, label="{} distribution".format(llabel))
    plt.fill_between(x_1, alpha * hfunc(x_1), 0.5 * lfunc(x_1), alpha=0.4, color=green)
    plt.fill_between(x_2, alpha * hfunc(x_2), 0.5 * lfunc(x_2), alpha=0.4, color=red)
    for x_val in x_values:
        d = None
        if x_val < x[idx]:
            a, =plt.plot([x_val, find_x_given_y(x_lim_low, x_lim_high, alpha * hfunc(x_val), lfunc) + marker_adjustment],
                     [alpha * hfunc(x_val), alpha * hfunc(x_val)], linestyle='--', color=green)
            b, =plt.plot(find_x_given_y(x_lim_low, x_lim_high, alpha * hfunc(x_val), lfunc) + marker_adjustment,
                        alpha * hfunc(x_val), linestyle='',
                        marker='<', color=green, markersize=6, zorder=100)
            lowdim, = plt.plot(find_x_given_y(x_lim_low, x_lim_high, alpha * hfunc(x_val), lfunc), 0, linestyle='', markeredgewidth=1,
                         marker="o", markerfacecolor=orange, markeredgecolor=orange, markersize=6, zorder=100, alpha=0.4)
            highdim, = plt.plot(x_val, alpha * hfunc(0), linestyle='', markeredgewidth=1,
                         marker="o", markerfacecolor=blue, markeredgecolor=blue, markersize=6, zorder=100, alpha=0.4)

        else:
            d, =plt.plot([x_val, find_x_given_y(x_lim_low, x_lim_high, alpha * hfunc(x_val), lfunc)- marker_adjustment],
                     [alpha * hfunc(x_val), alpha * hfunc(x_val)], linestyle='--', color=red)
            e, =plt.plot(find_x_given_y(x_lim_low, x_lim_high, alpha * hfunc(x_val), lfunc) - marker_adjustment,
                        alpha * hfunc(x_val), linestyle='',
                        marker='>', color=red, markersize=6, zorder=100)
            lowdim, = plt.plot(
                find_x_given_y(x_lim_low, x_lim_high, alpha * hfunc(x_val), lfunc), 0,
                linestyle='', markeredgewidth=1,
                marker="o", markerfacecolor=orange, markeredgecolor=orange, markersize=6, zorder=100, alpha=0.4)
            highdim, = plt.plot(x_val, alpha * hfunc(0), linestyle='', markeredgewidth=1,
                               marker="o", markerfacecolor=blue, markeredgecolor=blue, markersize=6, zorder=100,
                               alpha=0.4)

        #linedict = dict(linestyles="-", linewidths=1, alpha=0.5)

        # line to bottom
        #plt.gca().vlines(x_val, y_min, stats.norm.pdf(x_val, mu, sigma), **linedict)

        # line to top
        #plt.gca().vlines(find_x_given_y(x_lim_low, x_lim_high, stats.norm.pdf(x_val, mu, sigma), stats.t.pdf, 1),
        #                     stats.norm.pdf(x_val, mu, sigma), y_max, **linedict)


    ax = plt.gca()
    ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
    for label in ax.yaxis.get_ticklabels()[::2]:
        label.set_visible(False)

    handles = [(b,a,a)]
    labels = ["Attractive force"]
    if d is not None:
        handles.append((d,d,e))
        labels.append("Repulsive force")
    handles_, labels_ = ax.get_legend_handles_labels()

    handles.extend([highdim, lowdim])
    labels.extend(["Neighboring observations' distances\naccording to {} distribution".format(hlabel),
                   "Neighboring observations' distances\naccording to {} distribution".format(llabel)])

    handles_.extend(handles)
    labels_.extend(labels)


    ax.legend(handles=handles_, labels=labels_, loc='center right', fontsize=15,
              handler_map={tuple: mpl.legend_handler.HandlerTuple(None)})

    #plt.legend()

    mpl.rc('text', usetex=True)
    mpl.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]

    plt.xlabel(r"Pairwise distances $|| \mathbf{y}_i - \mathbf{y}_j ||^2$", fontsize=15)
    plt.ylabel("Neighboring probabilities", fontsize=15)
    ax.tick_params(right=False, top=False, labelright=False, labeltop=True)
    plt.title("Pairwise distances $|| \mathbf{x}_i - \mathbf{x}_j ||^2$", fontsize=15)
    sns.despine(left=True, right=True, bottom=True, top=True)
    plt.savefig("{}_{}_distance_shift_alpha_{}".format(htitle, ltitle, str(alpha)), bbox_inches="tight")
    plt.show()


def plot_fashion_mnist():

    fashion_labeldict = {
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

    data, labels = load_fashion_mnist_data(False, len_sample=400)
    n = 20
    fig, axes = plt.subplots(2, n, figsize=(2 * n, 5))
    for label in fashion_labeldict.keys():
        for i in range(n):
            axes[0, i].imshow(data[i].reshape(28, 28), cmap='gray')
            axes[1, i].imshow(data[n + i].reshape(28, 28), cmap='gray')
        for ax in axes.flatten():
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.savefig("fashion_test", bbox_inches="tight")
        plt.show()


def metric_plot(*metric_json_list, stoplying_iter=[], restartlying_iter=[], legend_labels=[], plot_title="metric_dict"):
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(16, 4))

    ax2 = ax.twinx()
    ax2.set_zorder(15)
    ax2.grid(False)

    linestyles = ['solid', 'dashed', 'dotted']

    for metric_json, stoplying, restartlying, label, linestyle in zip(metric_json_list, stoplying_iter, restartlying_iter, legend_labels, linestyles[:len(metric_json_list)]):
        for metric in metric_json.keys():
            if metric != "1NNgeneralization_error":
                x = np.array(list(metric_json[metric].keys()))
                x = x.astype(int)
                y = np.array(list(metric_json[metric].values()))

                # correct error values
                if metric == "cost_function_value":
                    for i, x_val in enumerate(x):
                        if int(x_val) < stoplying or int(x_val) >= restartlying:
                            y[i] = y[i]/12 - np.log(12)

                    ax.plot(x, y, linestyle=linestyle, color=blue, linewidth=1.5,
                            label="/", zorder=100)

                    #ax.annotate('Plateau indicates infinitesimal improvement\nin global organization (iteration 250)',
                    #            xy=(x[6], y[6]), fontsize=15, backgroundcolor="w", alpha=.9,
                    #            xytext=(x[9]+7, 3.7), arrowprops={'facecolor': 'black', 'shrink': 0.1, 'alpha':.6})

                    ax.set_ylabel('KL-cost', color=blue, fontsize=15)
                else:
                    ax2.plot(x, y, linestyle=linestyle, color=orange, linewidth=1.5,
                             #label="Restart exaggerating at iteration {}".format(str(restartlying)) if restartlying <= 1000 else "No exaggeration restart", zorder=100)
                             label=label, zorder=100)
                    ax2.set_ylabel('Trustworthiness T(12)', color=orange, fontsize=15)
                #if metric == "cost_function_value":
                #    plt.plot(x, np.repeat(1.17, len(x)), linestyle="--", color=blue, linewidth=1.5,
                #             label="Approximate local optimum")
    ax.set_xlabel('Iteration', fontsize=15)

    x_pad = 1000 * 0.05
    y_pad1 = 5 * 0.05
    y_pad2 = 0.5 * 0.05

    ax.set(xlim=(0-x_pad, 1000+x_pad), ylim=(0-y_pad1, 5+y_pad1))
    ax2.set(ylim=(0.5-y_pad2, 1+y_pad2))

    ax.tick_params(
        axis='y',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        left=False,  # ticks along the bottom edge are off
        labelleft=True,
        labelsize=12,
        labelcolor=blue)

    ax.tick_params(
        axis='x',  # changes apply to the x-axis
        labelsize=12)

    ax2.tick_params(
        axis='y',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        right=False,  # ticks along the bottom edge are off
        labelright=True,
        labelsize=12,
        labelcolor=orange)

    sns.despine(left=True, right=True, bottom=True, top=True)

    handlesL, labelsL = ax.get_legend_handles_labels()
    handlesR, labelsR = ax2.get_legend_handles_labels()
    handles = handlesL + handlesR
    labels = labelsL + labelsR
    plt.legend(handles, labels, loc='upper right', bbox_to_anchor=[.9, .9], ncol=2, fontsize=15,
               handletextpad=0.4, columnspacing=0.4)

    plt.savefig(plot_title, bbox_inches="tight")
    plt.show()


def plot_perplexity_trustworthiness():
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(16, 4))

    perplexity_x = np.array([2,5,10,20,30,40,50,100,200,500,1000,2333])
    perplexity_y = np.array([0.9295, 0.9688, 0.9760, 0.9799, 0.9808, 0.9811, 0.9895, 0.9881, 0.9821,0.9798 ,0.9764 ,0.9673])

    ax.plot(perplexity_x, perplexity_y, linestyle="-", color=orange, linewidth=1.5)

    ax.set_xlabel('Perplexity', fontsize=15)
    ax.set_ylabel('Trustworthiness T(12)', color=orange, fontsize=15)

    ax.tick_params(
        axis='x',  # changes apply to the x-axis
        labelsize=12)

    ax.tick_params(
        axis='y',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        labelsize=12,
        labelcolor=orange)

    #ax.set(xlim=(0, 2333), ylim=(0.925, 0.99))

    ax.annotate('Maximum Trustworthiness\nat perplexity = 50', xy=(perplexity_x[6],perplexity_y[6]),
                alpha=.9, backgroundcolor='w',
                xytext=(perplexity_x[8]+10, 0.956), arrowprops={'facecolor':'black', 'shrink':0.1, 'alpha':.6}, fontsize=15)
    sns.despine(left=True, right=True, bottom=True, top=True)
    plt.savefig("perplexity_trustworthiness", bbox_inches="tight")
    plt.show()


def plot_perplexity_time():

    x = np.array([2,5,10,20,30,40,50,100,200,500,1000,5000])
    y_tree = [76.36, 94.20, 107.90, 132.41, 160.06, 167.18, 189.13, 274.35, 315.30, 858.99, 2081.80, 32856.28]
    y_fit = [521.09, 527.09, 545.90, 589.03, 672.80, 703.97, 761.61, 958.56, 1062.05, 1895.86, 3235.10, 11936.49]

    y_total = [sum(x) for x in zip(y_tree, y_fit)]

    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(16, 4))

    ax.plot(x, y_total, linestyle="-", color='black', linewidth=1.5, label="Total time")

    tree = plt.fill_between(x, y_total, y_tree, alpha=0.4, color=blue)
    fit = plt.fill_between(x, y_tree, 0, alpha=0.4, color=orange)

    handles, labels = ax.get_legend_handles_labels()

    handles.extend((tree, fit))
    labels.extend(("Fitting time", "Input similarity approximation time"))

    ax.legend(handles=handles, labels=labels, loc='upper left',
              fontsize=15)

    #plt.yscale('log')

    ax.tick_params(
        axis='both',  # changes apply to the x-axis
        labelsize=12)

    ax.set_xlabel('Perplexity', fontsize=15)
    ax.set_ylabel('Time in seconds', fontsize=15)

    sns.despine(left=True, right=True, bottom=True, top=True)
    plt.savefig("perplexity_time", bbox_inches="tight")
    plt.show()


def plot_theta_run_time():

    num_samples = [500, 1000, 2500, 5000, 7500, 10000, 20000, 30000, 40000, 50000, 60000, 70000]
    theta_val = np.round(np.linspace(0,1,11),1)
    num_samples_exact = [200, 500, 1000, 2500, 5000, 7500, 10000, 20000]

    theta = np.array([  [ 3.8, 18.91, 136.18, 581.74, 1421.88, 2601.82, 10409.27, np.NaN, np.NaN, np.NaN, np.NaN , np.NaN],
                        [ 3.79, 9.94, 33.83, 90.63, 162.7, 239.57, 668.85, 1416.34, 2115.96, 2790.85, 3726.12, 4467.96],
                        [ 2.64, 5.76, 17.72, 43.68, 73.67, 107.16, 262.31, 457.36, 680.26, 935.22, 1205.91, 1510.96],
                        [ 1.99, 4.49, 12.92, 30.31, 53.89, 71.38, 169.19, 290.82, 432.21, 536.36, 727.62, 826.76],
                        [ 1.72, 3.81, 10.42, 23.95, 41.61, 56.46, 132.08, 218.09, 308.32, 413.97, 519.27, 598.7],
                        [ 1.56, 3.35, 9.25, 20.98, 34.32, 48.13, 112.23, 181.06, 270.19, 359.66, 493.43, 553.2],
                        [ 1.51, 3.05, 8.43, 19.89, 30.79, 44.58, 105.06, 167.62, 223.79, 293.17, 359.89, 433.65],
                        [ 1.35, 2.9, 8, 17.66, 29.23, 41.48, 95.17, 146.32, 204.74, 260.65, 338.86, 391.34],
                        [ 1.31, 2.65, 7.27, 17.02, 28.13, 36.68, 91.23, 138.04, 190.88, 243.85, 318.89, 351.19],
                        [ 1.28, 2.61, 7.01, 15.41, 24.81, 34.73, 81.37, 124.7, 168.31, 212.97, 265.08, 316.06],
                        [ 1.19, 2.54, 6.67, 14.26, 23.55, 30.79, 71.54, 114.15, 150.89, 190.96, 240.32, 283.32]])

    theta_log = np.round(np.log10(theta),2)


    fig, ax = plt.subplots(figsize=(8, 8))

    sns.set_style("whitegrid")
    ax = sns.heatmap(theta_log, annot=True, linewidths=.5, vmin=0, vmax=4, xticklabels=num_samples, yticklabels=theta_val,
                     cbar_kws=dict(ticks=[0, 1, 2, 3, 4], shrink=.77), square=True)
    cbar = ax.collections[0].colorbar
    # here set the labelsize by 20
    cbar.ax.tick_params(labelsize=12)
    cbar.ax.set_ylabel(r'$\log_{10}$ Time in seconds', fontsize=15)

    plt.xticks(rotation='vertical', fontsize=12)
    plt.yticks(rotation='horizontal', fontsize=12)
    plt.tick_params(axis='both', which='both', length=0)
    plt.ylabel(r"Gradient accuracy $\theta$", fontsize=15)
    plt.xlabel("Number of samples in database", fontsize=15)
    #ax.set_aspect("equal")

    plt.savefig("theta_numsample_time", bbox_inches="tight")
    plt.show()

def plot_theta_1nn_tradeoff():

    x = np.round(np.linspace(0.1,1,10),1)

    y_acc = np.array([0.7802,0.7797,0.7791,0.7781,0.7743,0.7731,0.7606,0.7573,0.7447,0.7294])

    y_nn = np.round(1-y_acc,4)
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(16, 4))

    y_range = y_nn[-1] - y_nn[0]
    y_pad = y_range * 0.05

    plt.plot(x, y_nn, linestyle="--", marker='o', color='black', linewidth=1.5)

    for i, txt in enumerate(y_nn):
        ax.annotate(txt, xy=(x[i], y_nn[i]), xytext=(x[i]-0.05, y_nn[i]+0.001), fontsize=12)

    plt.ylabel("1-NN error", fontsize=15)
    plt.xlabel(r"Gradient accuracy $\theta$", fontsize=15)

    ax.set(xlim=(0, 1.05), ylim=(y_nn[0] - y_pad, y_nn[-1] + y_pad*2))

    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.tick_params(
        axis='both',  # changes apply to the x-axis
        labelsize=12)

    plt.savefig("theta_1nnerror", bbox_inches="tight")
    plt.show()


def plot_initialization_comparison(t_sne=True):

    x = np.round(np.linspace(0, 1000, 21), 0)
    if t_sne:
        with open(
                'C:\\Users\\Tobi\\git\\bhtsne\\results\\tSNE\\buildingblocks\\initial_embeddings\\pca\\exaggeration\\1\\fashion_mnist7000\\1\\bh_tsne_result-18-08-2019_06-25-33-metrics.json','r') as f:
            metric_json = json.load(f)

        y_pca = np.delete(np.array(list(metric_json['cost_function_value'].values())), 1)

        with open(
                'C:\\Users\\Tobi\\git\\bhtsne\\results\\tSNE\\buildingblocks\\initial_embeddings\\mds\\exaggeration\\1\\fashion_mnist7000\\1\\bh_tsne_result-18-08-2019_10-35-52-metrics.json','r') as f:
            metric_json = json.load(f)

        y_mds = np.delete(np.array(list(metric_json['cost_function_value'].values())), 1)

        with open(
                'C:\\Users\\Tobi\\git\\bhtsne\\results\\tSNE\\buildingblocks\\initial_embeddings\\lle\\exaggeration\\1\\fashion_mnist7000\\1\\bh_tsne_result-18-08-2019_07-48-11-metrics.json','r') as f:
            metric_json = json.load(f)

        y_lle = np.delete(np.array(list(metric_json['cost_function_value'].values())), 1)

        with open(
                'C:\\Users\\Tobi\\git\\bhtsne\\results\\tSNE\\buildingblocks\\initial_embeddings\\autoencoder\\exaggeration\\1\\fashion_mnist7000\\1\\bh_tsne_result-18-08-2019_09-10-53-metrics.json','r') as f:
            metric_json = json.load(f)

        y_auto = np.delete(np.array(list(metric_json['cost_function_value'].values())),1 )
    else:
        with open(
                'C:\\Users\\Tobi\\git\\bhtsne\\results\\BHtSNE\\buildingblocks\\initial_embeddings\\pca\\exaggeration\\1\\fashion_mnist\\1\\bh_tsne_result-22-07-2019_11-17-43-metrics.json',
                'r') as f:
            metric_json = json.load(f)

        y_pca = np.delete(np.array(list(metric_json['cost_function_value'].values())), 1)

        with open(
                'C:\\Users\\Tobi\\git\\bhtsne\\results\\BHtSNE\\buildingblocks\\initial_embeddings\\lle\\exaggeration\\1\\fashion_mnist\\1\\bh_tsne_result-22-07-2019_12-26-44-metrics.json',
                'r') as f:
            metric_json = json.load(f)

        y_lle = np.delete(np.array(list(metric_json['cost_function_value'].values())), 1)

        with open(
                'C:\\Users\\Tobi\\git\\bhtsne\\results\\BHtSNE\\buildingblocks\\initial_embeddings\\autoencoder\\exaggeration\\1\\fashion_mnist\\1\\bh_tsne_result-23-07-2019_13-32-56-metrics.json',
                'r') as f:
            metric_json = json.load(f)

        y_auto = np.delete(np.array(list(metric_json['cost_function_value'].values())), 1)

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(16, 4))
    palette = sns.color_palette()

    plt.plot(x, y_pca, linestyle="-", linewidth=1.5, label="PCA", color=palette[0])
    if t_sne:
        plt.plot(x, y_mds, linestyle="-", linewidth=1.5, label="MDS", color=palette[1])
    plt.plot(x, y_lle, linestyle="-", linewidth=1.5, label="LLE", color=palette[2])
    plt.plot(x, y_auto, linestyle="-", linewidth=1.5, label="Autoencoder", color=palette[3])

    ax.tick_params(
        axis='both',  # changes apply to the x-axis
        labelsize=12)

    plt.ylabel("KL-cost", fontsize=15)
    plt.xlabel("Iteration", fontsize=15)



    axins = inset_axes(ax, width="30%", height="30%", loc=10 if t_sne else 7)
    axins.plot(x, y_pca, linestyle="-", linewidth=1.5, label="PCA", color=palette[0])
    if t_sne:
        axins.plot(x, y_mds, linestyle="-", linewidth=1.5, label="MDS", color=palette[1])
    axins.plot(x, y_lle, linestyle="-", linewidth=1.5, label="LLE", color=palette[2])
    axins.plot(x, y_auto, linestyle="-", linewidth=1.5, label="Autoencoder", color=palette[3])
    axins.set_xlim([890, 1000.5])
    if t_sne:
        axins.set_ylim([1.1801, 1.2099])
    else:
        axins.set_ylim([2.9064, 3.135])

    axins.xaxis.tick_top()
    axins.tick_params(axis='x', which='both', length=0)
    axins.xaxis.set_major_locator(ticker.MultipleLocator(50))
    mark_inset(ax, axins, loc1=1 if t_sne else 3, loc2=3 if t_sne else 4, fc="2", ec="0.5")

    # ax2 = plt.axes([0.4, 0.4, .35, .2], facecolor=None)
    # ax2.plot(x, y_pca, linestyle="-", linewidth=1.5, label="PCA")
    # ax2.plot(x, y_mds, linestyle="-", linewidth=1.5, label="MDS")
    # ax2.plot(x, y_lle, linestyle="-", linewidth=1.5, label="LLE")
    # ax2.plot(x, y_auto, linestyle="-", linewidth=1.5, label="Autoencoder")
    # ax2.set_xlim([890, 1000])
    # ax2.set_ylim([1.1801, 1.2099])
    # ax2.xaxis.set_major_locator(ticker.MultipleLocator(50))

    # ax3 = plt.axes([0.3, 0.4, .25, .2], facecolor=None)
    # ax3.plot(x, y_pca, linestyle="-", linewidth=1.5, label="PCA")
    # ax3.plot(x, y_mds, linestyle="-", linewidth=1.5, label="MDS")
    # ax3.plot(x, y_lle, linestyle="-", linewidth=1.5, label="LLE")
    # ax3.plot(x, y_auto, linestyle="-", linewidth=1.5, label="Autoencoder")
    # ax3.set_xlim([100, 200])
    # ax3.set_ylim([1.8295, 2.5649])
    # ax3.xaxis.set_major_locator(ticker.MultipleLocator(50))

    ax.tick_params(
        axis='both',  # changes apply to the x-axis
        labelsize=12)
    ax.legend(fontsize=15, ncol=4, handletextpad=0.4, columnspacing=0.4, loc="upper center")
    sns.despine(left=True, right=True, bottom=True, top=True)

    plt.savefig("initial_embeddings_cost_{}".format("tSNE" if t_sne else "BHtSNE"), bbox_inches="tight")

    plt.show()


def plot_blacklist():

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(8, 8))
    palette = sns.color_palette()

    # reference point
    plt.plot([0.4], [0.4], c=palette[0], linestyle="", marker='o', zorder=100, label="Reference Observation")

    # neighboring points
    neighbors = [[0.51, 0.64, 0.6], [0.63, 0.6, 0.52]]
    neighbor1, = plt.plot(neighbors[0], neighbors[1], c=palette[3], linestyle="", marker='o', zorder=100, label="Neighboring Observation")

    # edge point
    neighbor1, = plt.plot([0.55], [0.575], c=palette[2], linestyle="", marker='o', zorder=100, label="Neighboring Observation")

    # Create a Rectangle patch
    rect = patches.Rectangle((0.5, 0.5), 0.15, 0.15, linewidth=.7, edgecolor="black", facecolor='none', label="Splittree cell")

    # Add the patch to the Axes
    ax.add_patch(rect)

    # connections
    #no
    for i in range(3):
        x = [0.4] + [neighbors[0][i]]
        y = [0.4] + [neighbors[1][i]]
        plt.plot(x, y, linestyle='--', linewidth=0.7, c=palette[3], label="$p_{ij}$ not given")

    #yes
    plt.plot([0.4] + [0.55],[0.4] + [0.575], linestyle='-', linewidth=0.7, c=palette[2], label="$p_{ij}$ given")
    handles, labels = ax.get_legend_handles_labels()

    _handles = [handles[0]]
    _labels = [labels[0]]

    _handles.append((handles[1], handles[2]))
    _labels.append(labels[1])

    _handles.append(handles[3])
    _labels.append(labels[3])

    _handles.append(handles[6])
    _labels.append(labels[6])

    _handles.append(handles[7])
    _labels.append(labels[7])

    ax.legend(handles=_handles, labels=_labels, loc='lower right', fontsize=15,
              handler_map={tuple: mpl.legend_handler.HandlerTuple(None)})
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    #plt.legend(fontsize=15)
    ax.set_xlim([0.35, 0.7])
    ax.set_ylim([0.35, 0.7])

    plt.savefig("blacklist_splittree", bbox_inches="tight")

    plt.show()


if __name__ == '__main__':

    plot_blacklist()
    #plot_initialization_comparison(t_sne=False)
    #plot_theta_1nn_tradeoff()
    #plot_theta_run_time()
    #plot_gaussian_student_distance_shift(0, 6, 1, 1.5, 2, 2.5, alpha=1, high_dim_pdf="gausian", low_dim_pdf="chi")

    #plot_perplexity_trustworthiness()
    #plot_perplexity_time()

    #plot_pca_new_directions()
    #plot_lda()
    #plot_lle()

    #plot_fashion_mnist()


    # restartlying
    #with open('C:\\Users\\Tobi\\git\\bhtsne\\results\\tSNE\\parametertuning\\iterations\\1000\\fashion_mnist7000\\1\\bh_tsne_result-14-08-2019_09-23-14-metrics.json', 'r') as f:
    #    metric_json = json.load(f)

    #with open('C:\\Users\\Tobi\\git\\bhtsne\\results\\tSNE\\parametertuning\\restartlying\\750\\fashion_mnist7000\\1\\bh_tsne_result-16-08-2019_10-17-22-metrics.json', 'r') as f:
    #    metric_json2 = json.load(f)


    # stoplying
    #with open('C:\\Users\\Tobi\\git\\bhtsne\\results\\tSNE\\parametertuning\\stoplying\\500\\fashion_mnist7000\\1\\bh_tsne_result-16-08-2019_05-14-42-metrics.json', 'r') as f:
    #    metric_json2 = json.load(f)

    #with open('C:\\Users\\Tobi\\git\\bhtsne\\results\\tSNE\\parametertuning\\stoplying\\750\\fashion_mnist7000\\1\\bh_tsne_result-16-08-2019_06-54-48-metrics.json', 'r') as f:
    #    metric_json3 = json.load(f)

    #with open('C:\\Users\\Tobi\\git\\bhtsne\\results\\tSNE\\parametertuning\\stoplying\\1000\\fashion_mnist7000\\1\\bh_tsne_result-16-08-2019_08-35-54-metrics.json', 'r') as f:
    #    metric_json4 = json.load(f)

    # learningrate
    #with open('C:\\Users\\Tobi\\git\\bhtsne\\results\\tSNE\\parametertuning\\learningrate\\50\\fashion_mnist7000\\1\\bh_tsne_result-15-08-2019_05-28-59-metrics.json', 'r') as f:
    #     metric_json2 = json.load(f)

    #with open('C:\\Users\\Tobi\\git\\bhtsne\\results\\tSNE\\parametertuning\\iterations\\1000\\fashion_mnist7000\\1\\bh_tsne_result-14-08-2019_09-23-14-metrics.json', 'r') as f:
    #     metric_json3 = json.load(f)

    #with open('C:\\Users\\Tobi\\git\\bhtsne\\results\\tSNE\\parametertuning\\learningrate\\1000\\fashion_mnist7000\\1\\bh_tsne_result-15-08-2019_10-35-11-metrics.json', 'r') as f:
    #    metric_json4 = json.load(f)

    #metric_plot(metric_json2, metric_json3, metric_json4, stoplying_iter=[250, 250, 250],
    #            restartlying_iter=[1001, 1001, 1001],
    #            legend_labels=["Learning rate: 50", "Learning rate: 200", "Learning rate: 1000"], plot_title="KLT12tsnelearningrate")

    # momentum
    # with open(
    #         'C:\\Users\\Tobi\\git\\bhtsne\\results\\tSNE\\parametertuning\\momentum\\0.0\\fashion_mnist7000\\1\\bh_tsne_result-15-08-2019_12-17-32-metrics.json',
    #         'r') as f:
    #     metric_json2 = json.load(f)
    #
    # with open(
    #         'C:\\Users\\Tobi\\git\\bhtsne\\results\\tSNE\\parametertuning\\iterations\\1000\\fashion_mnist7000\\1\\bh_tsne_result-14-08-2019_09-23-14-metrics.json',
    #         'r') as f:
    #     metric_json3 = json.load(f)
    #
    # with open(
    #         'C:\\Users\\Tobi\\git\\bhtsne\\results\\tSNE\\parametertuning\\momentum\\0.8\\fashion_mnist7000\\1\\bh_tsne_result-15-08-2019_19-05-22-metrics.json',
    #         'r') as f:
    #     metric_json4 = json.load(f)
    #
    # metric_plot(metric_json4, metric_json3, metric_json2, stoplying_iter=[250, 250, 250],
    #             restartlying_iter=[1001, 1001, 1001],
    #             legend_labels=["Momentum: 0.0", "Momentum: 0.5", "Momentum: 0.8"],
    #             plot_title="KLT12tsnemomentum")

    # finalmomentum
    # with open(
    #         'C:\\Users\\Tobi\\git\\bhtsne\\results\\tSNE\\parametertuning\\finalmomentum\\0.0\\fashion_mnist7000\\1\\bh_tsne_result-15-08-2019_20-47-08-metrics.json',
    #         'r') as f:
    #     metric_json2 = json.load(f)
    #
    # with open(
    #         'C:\\Users\\Tobi\\git\\bhtsne\\results\\tSNE\\parametertuning\\finalmomentum\\0.5\\fashion_mnist7000\\1\\bh_tsne_result-16-08-2019_01-52-16-metrics.json',
    #         'r') as f:
    #     metric_json3 = json.load(f)
    #
    # with open(
    #         'C:\\Users\\Tobi\\git\\bhtsne\\results\\tSNE\\parametertuning\\iterations\\1000\\fashion_mnist7000\\1\\bh_tsne_result-14-08-2019_09-23-14-metrics.json',
    #         'r') as f:
    #     metric_json4 = json.load(f)
    #
    #
    #
    # metric_plot(metric_json2, metric_json3, metric_json4, stoplying_iter=[250, 250, 250],
    #             restartlying_iter=[1001, 1001, 1001],
    #             legend_labels=["Final momentum: 0.0", "Final momentum: 0.5", "Final momentum: 0.8"],
    #             plot_title="KLT12tsnefinalmomentum")
    #
    # # momentumswitch
    #
    # with open(
    #         'C:\\Users\\Tobi\\git\\bhtsne\\results\\tSNE\\parametertuning\\iterations\\1000\\fashion_mnist7000\\1\\bh_tsne_result-14-08-2019_09-23-14-metrics.json',
    #         'r') as f:
    #     metric_json2 = json.load(f)
    #
    # with open(
    #         'C:\\Users\\Tobi\\git\\bhtsne\\results\\tSNE\\parametertuning\\momentumswitch\\500\\fashion_mnist7000\\1\\bh_tsne_result-16-08-2019_11-58-46-metrics.json',
    #         'r') as f:
    #     metric_json3 = json.load(f)
    #
    # with open(
    #         'C:\\Users\\Tobi\\git\\bhtsne\\results\\tSNE\\parametertuning\\momentumswitch\\750\\fashion_mnist7000\\1\\bh_tsne_result-16-08-2019_13-40-01-metrics.json',
    #         'r') as f:
    #     metric_json4 = json.load(f)
    #
    # metric_plot(metric_json2, metric_json3, metric_json4, stoplying_iter=[250, 250, 250],
    #             restartlying_iter=[1001, 1001, 1001],
    #             legend_labels=["Momentum switch iteration: 250", "Momentum switch iteration: 500", "Momentum switch iteration: 750"],
    #             plot_title="KLT12tsnemomentumswitch")


#!/usr/bin/env python

import matplotlib.pyplot as plt
import math
import numpy as np
import scipy.stats as stats
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib as mpl
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
import matplotlib.lines as mlines

blue = (57 / 256, 106 / 256, 177 / 256)
orange = (218 / 256, 124 / 256, 48 / 256)
green = (132 / 256, 186 / 256, 91 / 256)
red = (211 / 256, 94 / 256, 96 / 256)
grey = (83 / 256, 81 / 256, 84 / 256)
ercis = (146 / 256, 36 / 256, 40 / 256)


def find_x_given_y(x_lim_low, x_lim_high, y_value, y_function, *y_args):
    x = np.linspace(x_lim_low, x_lim_high, 10000)
    y = y_function(x, *y_args)

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


def plot_gaussian_student_distance_shift(x_lim_low, x_lim_high, *x_values):
    mu = 0
    variance = 1
    sigma = math.sqrt(variance)
    x = np.linspace(x_lim_low, mu + x_lim_high * sigma, 10000)

    idx = get_intersection(stats.norm.pdf(x, mu, sigma), stats.t.pdf(x, 1))[0]


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
    plt.plot(x, stats.norm.pdf(x, mu, sigma), linestyle="-", color=blue, linewidth=1.5, label="Standard normal distribution")
    plt.plot(x, stats.t.pdf(x, 1), linestyle="-", color=orange, linewidth=1.5, label="Student-t (df = 1) distribution")
    plt.fill_between(x_1, stats.norm.pdf(x_1, mu, sigma), stats.t.pdf(x_1, 1), alpha=0.4, color=green)
    plt.fill_between(x_2, stats.norm.pdf(x_2, mu, sigma), stats.t.pdf(x_2, 1), alpha=0.4, color=red)
    for x_val in x_values:
        if x_val < 1.83:
            a, =plt.plot([x_val, find_x_given_y(x_lim_low, x_lim_high, stats.norm.pdf(x_val, mu, sigma), stats.t.pdf, 1) + marker_adjustment],
                     [stats.norm.pdf(x_val, mu, sigma), stats.norm.pdf(x_val, mu, sigma)], linestyle='--', color=green)
            b, =plt.plot(find_x_given_y(x_lim_low, x_lim_high, stats.norm.pdf(x_val, mu, sigma), stats.t.pdf, 1) + marker_adjustment,
                        stats.norm.pdf(x_val, mu, sigma), linestyle='',
                        marker='<', color=green, markersize=6, zorder=100)



            #c, =plt.plot(x_val, stats.norm.pdf(x_val, mu, sigma), linestyle='', markeredgewidth=1,
            #            marker="o", markerfacecolor='none', markeredgecolor=green, markersize=6, zorder=100)
        else:
            d, =plt.plot([x_val, find_x_given_y(x_lim_low, x_lim_high, stats.norm.pdf(x_val, mu, sigma), stats.t.pdf, 1)- marker_adjustment],
                     [stats.norm.pdf(x_val, mu, sigma), stats.norm.pdf(x_val, mu, sigma)], linestyle='--', color=red)
            e, =plt.plot(find_x_given_y(x_lim_low, x_lim_high, stats.norm.pdf(x_val, mu, sigma), stats.t.pdf, 1) - marker_adjustment,
                        stats.norm.pdf(x_val, mu, sigma), linestyle='',
                        marker='>', color=red, markersize=6, zorder=100)
            #f, =plt.plot(x_val, stats.norm.pdf(x_val, mu, sigma), linestyle='', markeredgewidth = 1,
            #            marker="o", markerfacecolor='none', markeredgecolor=red, markersize=6, zorder=100)

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

    handles = [(b,a,a), (d,d,e)]
    handles_, labels = ax.get_legend_handles_labels()

    handles_.extend(handles)
    labels.extend(("Attractive force", "Repulsive force"))


    ax.legend(handles=handles_, labels=labels, loc='upper right',
              handler_map={tuple: mpl.legend_handler.HandlerTuple(None)})

    #plt.legend()

    mpl.rc('text', usetex=True)
    mpl.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]

    plt.xlabel(r"Pairwise distances  $|| \mathbf{x}_i - \mathbf{x}_j ||^2$")
    plt.ylabel("Neighboring probabilities")
    ax.tick_params(right=False, top=False, labelright=False, labeltop=True)
    plt.title("Pairwise distances $|| \mathbf{y}_i - \mathbf{y}_j ||^2$")
    sns.despine(left=True, right=True, bottom=True, top=True)
    plt.savefig("gaussian_student_distance_shift", bbox_inches="tight")
    plt.show()


if __name__ == '__main__':
    plot_gaussian_student_distance_shift(0, 5, 1, 1.5, 2, 2.5)

    #plot_pca_new_directions()
    #plot_lda()
    #plot_lle()
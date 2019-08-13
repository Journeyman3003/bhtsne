#!/usr/bin/env python

import matplotlib.pyplot as plt
import math
import numpy as np
import scipy.stats as stats
import matplotlib as mpl
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter


def format_y_axis(value, tick_number):
    None


def find_x_given_y(x_lim_low, x_lim_high, y_value, y_function, *y_args):
    x = np.linspace(x_lim_low, x_lim_high, 100)
    y = y_function(x, *y_args)

    rev_x = x[::-1]
    rev_y = y[::-1]

    return np.interp(y_value, rev_y, rev_x)


def plot_gaussian_student_distance_shift(x_lim_low, x_lim_high, *x_values):
    mu = 0
    variance = 1
    sigma = math.sqrt(variance)
    x = np.linspace(x_lim_low, mu + x_lim_high * sigma, 100)
    sns.set(style="whitegrid")

    plt.figure(figsize=(10, 5))
    plt.plot(x, stats.norm.pdf(x, mu, sigma), linestyle="-", linewidth=1, label="Standard normal distribution")
    plt.plot(x, stats.t.pdf(x, 1), linestyle="-", linewidth=1, label="Student-t (df = 1) distribution")
    for x_val in x_values:
        if x_val < 1.83:
            plt.plot([x_val, find_x_given_y(x_lim_low, x_lim_high, stats.norm.pdf(x_val, mu, sigma), stats.t.pdf, 1)],
                     [stats.norm.pdf(x_val, mu, sigma), stats.norm.pdf(x_val, mu, sigma)], 'g--')
            plt.scatter(find_x_given_y(x_lim_low, x_lim_high, stats.norm.pdf(x_val, mu, sigma), stats.t.pdf, 1),
                        stats.norm.pdf(x_val, mu, sigma),
                        marker=mpl.markers.CARETLEFT, color='g', s=20, zorder=100)
            plt.scatter(x_val, stats.norm.pdf(x_val, mu, sigma),
                        marker="o", facecolors='w', edgecolors='g', s=20, zorder=100)
        else:
            plt.plot([x_val, find_x_given_y(x_lim_low, x_lim_high, stats.norm.pdf(x_val, mu, sigma), stats.t.pdf, 1)],
                     [stats.norm.pdf(x_val, mu, sigma), stats.norm.pdf(x_val, mu, sigma)], 'r--')
            plt.scatter(find_x_given_y(x_lim_low, x_lim_high, stats.norm.pdf(x_val, mu, sigma), stats.t.pdf, 1),
                        stats.norm.pdf(x_val, mu, sigma),
                        marker=mpl.markers.CARETRIGHT, color='r', s=20, zorder=100)
            plt.scatter(x_val, stats.norm.pdf(x_val, mu, sigma),
                        marker="o", facecolors='w', edgecolors='r', s=20, zorder=100)


    ax = plt.gca()
    ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
    for label in ax.yaxis.get_ticklabels()[::2]:
        label.set_visible(False)
    plt.legend()
    plt.xlabel("Pairwise distances")
    plt.ylabel("Neighboring probabilities")
    sns.despine(left=True, right=True, bottom=True, top=True)
    plt.savefig("gaussian_student_distance_shift", bbox_inches="tight")
    plt.show()


if __name__ == '__main__':
    plot_gaussian_student_distance_shift(0, 5, 1, 2, 2.5)


#!/usr/bin/env python

import matplotlib.pyplot as plt
import math
import numpy as np
import scipy.stats as stats
import matplotlib as mpl
import seaborn as sns


def plot_gaussian_student_distance_shift():
    mu = 0
    variance = 1
    sigma = math.sqrt(variance)
    x = np.linspace(0, mu + 10 * sigma, 100)
    sns.set()
    plt.plot(x, stats.norm.pdf(x, mu, sigma), linestyle="-", label="Standard normal distribution")
    plt.plot(x, stats.t.pdf(x, df=1), linestyle="-", label="Student-t (df = 1) distribution")
    plt.plot([ , ], [, ], 'ro-')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    plot_gaussian_student_distance_shift()
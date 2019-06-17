import numpy as np
import bhtsne
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

PLOT_DIR = "plots"

if __name__ == "__main__":
    mnist = np.loadtxt("MNIST/mnist2500_X.txt", dtype=float)
    embedding_array = bhtsne.run_bh_tsne(mnist, initial_dims=mnist.shape[1], verbose=True)

    mnist_labels = np.loadtxt("MNIST/mnist2500_labels.txt", dtype=float)
    mnist_latent = np.hstack((embedding_array, np.reshape(mnist_labels, (2500, 1))))

    # plots
    # fig, ax = plt.subplots()
    # scatter = ax.scatter(x=mnist_latent[:, 0], y=mnist_latent[:, 1], c=mnist_latent[:, 2])
    # produce a legend with the unique colors from the scatter
    # legend1 = ax.legend(*scatter.legend_elements(),
    #                    loc="lower left", title="MNIST")
    # ax.add_artist(legend1)
    # plt.show()

    # seaborn
    g = sns.scatterplot(x=mnist_latent[:, 0],
                        y=mnist_latent[:, 1],
                        hue=mnist_latent[:, 2],
                        legend="full",
                        palette=sns.color_palette("bright"))

    fig = g.get_figure()
    figure_name = "tSNE-MNIST-" + str(datetime.now()).replace(":", "_").replace(".", "_")
    fig.savefig(os.path.join(PLOT_DIR, figure_name))





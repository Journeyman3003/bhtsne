import mnist
import bhtsne

data, _ = mnist.load_mnist_data(True)
bhtsne.debug_bh_tsne_pre(data)
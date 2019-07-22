from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Flatten, Reshape
from keras.models import Model, load_model
from keras.callbacks import ReduceLROnPlateau
from keras import optimizers
import matplotlib.pyplot as plt
import os
import pickle
import numpy as np

import mnist

# directory structure
CWD = os.path.dirname(os.path.realpath(__file__))
INIT = os.path.join(CWD, "initial_solutions")

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


def make_and_fit(data, train_test_split_threshold=60000):

    x_train = data[:train_test_split_threshold, :]
    x_test = data[train_test_split_threshold:, :]

    x_train = np.reshape(x_train, (-1, 28, 28, 1))
    x_test = np.reshape(x_test, (-1, 28, 28, 1))

    inputs = Input(shape=(28, 28, 1))

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = BatchNormalization()(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(32, (2, 2), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = BatchNormalization()(x)
    x = Conv2D(16, (2, 2), activation='relu', padding='same')(x)
    x = Conv2D(4, (2, 2), activation='relu', padding='same')(x)
    x = Conv2D(1, (2, 2), activation='relu', padding='same')(x)
    x = Flatten()(x)
    encoded = Dense(2, activation='relu')(x)

    encoder = Model(inputs=inputs, outputs=encoded)

    encoded_inputs = Input(shape=(2,))

    x = Dense(4, activation='relu')(encoded_inputs)
    x = Reshape((2, 2, 1))(x)
    x = Conv2D(4, (2, 2), activation='relu', padding='same')(x)
    x = Conv2D(16, (2, 2), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D((7, 7))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    decoder = Model(inputs=encoded_inputs, outputs=decoded)

    x = encoder(inputs)
    x = decoder(x)
    model = Model(inputs=inputs, outputs=x)
    model.compile(optimizer=optimizers.Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy', 'mse'])

    print(model.summary())

    clr = ReduceLROnPlateau(
        monitor='loss',
        factor=0.5,
        patience=3,
        min_delta=0.01,
        cooldown=0,
        min_lr=1e-7,
        verbose=1)

    model.fit(
        x_train,
        x_train,
        batch_size=256,
        epochs=50,
        shuffle=True,
        validation_data=(x_test, x_test),
        callbacks=[clr])

    return model, encoder, decoder


def get_triple(inputs):
    latent_repr = encoder.predict(inputs)
    outputs = decoder.predict(latent_repr)
    latent_repr = latent_repr.reshape((latent_repr.shape[0], latent_repr.shape[1], 1))

    return inputs, latent_repr, outputs


def show_encodings(inputs, latent_repr, outputs, filename="autoencoder-2dim-fashion_mnist.png"):
    n = len(inputs)
    fig, axes = plt.subplots(2, n, figsize=(2 * n, 5))
    for i in range(n):
        axes[0, i].set_title(str(i))
        axes[1, i].set_title('({0:.2f}, {1:.2f})'.format(float(latent_repr[i, 0]), float(latent_repr[i, 1])))
        axes[0, i].imshow(inputs[i].reshape(28, 28), cmap='gray')
        axes[1, i].imshow(outputs[i].reshape(28, 28), cmap='gray')
    for ax in axes.flatten():
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':

    data_name = "fashion_mnist"
    #data_name = "mnist"
    data, labels = mnist.load_fashion_mnist_data()
    # #data, labels = mnist.load_mnist_data(True)
    #
    # # data = mnist.mnist_1d_to_2d(data)
    # #
    # # #model, encoder, decoder = make_and_fit(data)
    # # #
    # # #model.save("autoencoder.h5")
    # # #encoder.save("encoder.h5")
    # # #decoder.save("decoder.h5")
    # #
    # autoencoder = load_model(os.path.join("autoencoders", "fashion_mnist", "autoencoder.h5"))
    # encoder = load_model(os.path.join("autoencoders", "fashion_mnist","encoder.h5"))
    # decoder = load_model(os.path.join("autoencoders", "fashion_mnist","decoder.h5"))
    # #
    # data = np.reshape(data, (-1, 28, 28, 1)) / 255.0
    # # data = np.reshape(data, (-1, 28, 28, 1))
    # #
    # inputs, latent_embedding, outputs = get_triple(data)
    # #
    # indexes = list(map(lambda x: np.argmax(labels == x), np.arange(10)))
    # #
    # show_encodings(inputs[indexes], latent_embedding[indexes], outputs[indexes])
    # # #show_encodings(inputs[indexes], latent_embedding[indexes], outputs[indexes], filename="autoencoder-2dim-mnist.png")
    # #
    filename = "initial_solution_" + data_name + "_autoencoder.pickle"
    # #
    file_abspath = os.path.join(INIT, filename)
    #
    #latent_embedding = np.reshape(latent_embedding, (-1, 2))
    #
    # #
    # with open(file_abspath, 'wb') as pickle_file:
    #    pickle.dump(latent_embedding, pickle_file)

    with open(file_abspath, 'rb') as pickle_file:
        latent_embedding = pickle.load(pickle_file)

    from matplotlib import pyplot as plt
    import seaborn as sns
    sns.scatterplot(x=latent_embedding[:, 0],
                    y=latent_embedding[:, 1],
                    hue=labels,
                    legend="full",
                    palette=sns.color_palette("bright"))

    plt.show()



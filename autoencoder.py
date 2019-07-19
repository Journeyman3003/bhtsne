from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Flatten, Reshape
from keras.models import Model
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


def make_and_fit(data, train_test_split_threshold=60000):

    data = mnist.mnist_1d_to_2d(data)

    x_train = data[:train_test_split_threshold, :]
    x_test = data[train_test_split_threshold:, :]

    x_train = np.reshape(x_train, (-1, 28, 28, 1)) / 255.0
    x_test = np.reshape(x_test, (-1, 28, 28, 1)) / 255.0

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


def plot_autoencoder_results(original, predicted, filename="autoencoder-2dim-fashion_mnist.png"):
    for i in range(10):
        # display original images
        ax = plt.subplot(2, 10, i + 1)
        plt.imshow(original[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstructed images
        ax = plt.subplot(2, 10, 10 + i + 1)
        plt.imshow(predicted[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':

    data_name = "fashion_mnist"
    data, _ = mnist.load_fashion_mnist_data()

    model, encoder, decoder = make_and_fit(data)

    encoded_imgs = encoder.predict(data)
    predicted_imgs = model.predict(data)

    filename = "initial_solution_" + data_name + "_autoencoder.pickle"

    file_abspath = os.path.join(INIT, filename)

    with open(file_abspath, 'wb') as pickle_file:
        pickle.dump(encoded_imgs, pickle_file)

    plot_autoencoder_results(data, predicted_imgs)



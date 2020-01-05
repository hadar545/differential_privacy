import keras
from keras.datasets import mnist
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout, \
    UpSampling2D, Reshape, Conv2DTranspose
from keras.utils import np_utils
import numpy as np
from matplotlib import pyplot as plt


def plot_noacc(history, name):
    plt.figure()
    plt.plot(history.history['loss'], linewidth=2, label='train')
    plt.plot(history.history['val_loss'], linewidth=2, label='test')
    plt.xlabel('epochs')
    plt.ylabel('mean squared error loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(name)


def plot(history, name):
    f, ax = plt.subplots()
    ax.plot(history.history['acc'], label='accuracy', color='red')
    ax.tick_params(axis='y', labelcolor='red')
    ax2 = ax.twinx()
    ax2.plot(history.history['loss'], label='train loss')
    ax2.plot(history.history['val_loss'], label='test loss')
    plt.xlabel('epochs')
    ax.set_ylabel('accuracy', color='red')
    ax2.set_ylabel('loss')
    plt.legend(loc=2)
    f.tight_layout()
    plt.savefig(name)
    # plt.show()


def conv_autoencoder(data, num_epochs=2, batch_size=32, im_size=28, latent_size=2,
                     save_path=''):
    (X_train, y_train), (X_test, y_test) = data

    # encoder
    inputs = keras.layers.Input((im_size, im_size, 1))
    norm = keras.layers.BatchNormalization()(inputs)
    E0 = Conv2D(8, kernel_size=(5, 5), activation='relu', padding='same')(norm)
    E1 = MaxPooling2D(pool_size=(2, 2))(E0)
    E2 = Conv2D(16, kernel_size=(5, 5), activation='relu', padding='same')(E1)
    E3 = MaxPooling2D(pool_size=(2, 2))(E2)
    E4 = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(E3)
    E5 = Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same')(E4)
    E6 = Conv2D(4, kernel_size=(2, 2), activation='relu', padding='same')(E5)
    E7 = Flatten()(E6)
    E_fin = Dense(latent_size, activation='relu', activity_regularizer=keras.regularizers.l2(10e-3))(E7)

    # decoder
    D_inp = keras.layers.Input((latent_size))
    D0 = Dense(196, activation='relu')(E_fin)
    D1 = Reshape((7, 7, 4))(D0)
    D2 = Conv2D(4, kernel_size=(2, 2), activation='relu', padding='same')(D1)
    D3 = Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same')(D2)
    D4 = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(D3)
    D5 = UpSampling2D(size=(2, 2))(D4)
    D6 = Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same')(D5)
    D7 = UpSampling2D(size=(2, 2))(D6)
    D8 = Conv2D(8, kernel_size=(3, 3), activation='relu', padding='same')(D7)
    D_fin = Conv2D(1, kernel_size=(2, 2), activation='relu', padding='same')(D8)
    model = keras.models.Model(inputs=inputs, outputs=D_fin)
    model.compile(loss='mse', metrics=['loss'],
                  optimizer='adam')
    model.summary()
    # keras.utils.plot_model(model, 'conv_autoencoder_model.png', show_shapes=True, show_layer_names=False)

    encoder_hist = model.fit(X_train[..., np.newaxis], X_train[..., np.newaxis],
                             batch_size=batch_size, epochs=num_epochs,
                             validation_data=(X_test[..., np.newaxis], X_test[..., np.newaxis]), verbose=2)
    plot_noacc(encoder_hist, 'Conv Autoencoder')
    model.save(save_path + 'autoencoder.h5')

    encoder = keras.models.Model(inputs=inputs, outputs=E_fin)
    encoder.compile(loss='mse', metrics='loss', optimizer='adam')
    encoder.save(save_path + 'encoder.h5')

    decoder = keras.models.Model(inputs=D_inp, outputs=D_fin)
    decoder.compile(loss='mse', metrics='loss', optimizer='adam')
    decoder.save(save_path + 'decoder.h5')

    ims = X_train[:25]
    plt.figure()
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.imshow(ims[i], cmap='gray')
        plt.axis('off')
    plt.tight_layout(h_pad=0, w_pad=0)
    plt.savefig('orignals.png')

    recon = model.predict(ims[:, :, :, None])[:, :, :, 0]
    plt.figure()
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.imshow(recon[i], cmap='gray')
        plt.axis('off')
    plt.tight_layout(h_pad=0, w_pad=0)
    plt.savefig('reconstruction.png')


if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # preprocess data
    n_classes = 10
    test_labels = y_test
    X_train = X_train.astype('float32')/255
    X_test = X_test.astype('float32')/255
    y_train = np_utils.to_categorical(y_train, n_classes)
    y_test = np_utils.to_categorical(y_test, n_classes)

    train_amnt, test_amnt = 6000, 1000
    data = ((X_train[:train_amnt], y_train[:train_amnt]),
            (X_test[:test_amnt], y_test[:test_amnt]))

    conv_autoencoder(data, num_epochs=10, batch_size=20)


from sklearn.decomposition import PCA
import numpy as np
from matplotlib import pyplot as plt
import keras
from keras.datasets import mnist
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, UpSampling2D, Reshape
from keras.utils import np_utils


fig_path = 'presentation/'


def get_encoded(data, model):
    encoder = keras.models.Model(inputs=model.input,
                                 outputs=model.get_layer('dense_1').output)
    return encoder.predict(data)


def conv_autoencoder(data, num_epochs=10, batch_size=32, im_size=28, latent_size=50):
    (X_train, y_train), (X_test, _) = data
    activ = 'relu'
    inp = keras.layers.Input((im_size, im_size, 1))
    # encoder
    E0 = Conv2D(8, kernel_size=(5, 5), activation=activ, padding='same')(inp)
    E1 = MaxPooling2D(pool_size=(2, 2))(E0)
    E2 = Conv2D(16, kernel_size=(5, 5), activation=activ, padding='same')(E1)
    E3 = MaxPooling2D(pool_size=(2, 2))(E2)
    E4 = Conv2D(16, kernel_size=(3, 3), activation=activ, padding='same')(E3)
    # E5 = Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same')(E4)
    # E6 = Conv2D(4, kernel_size=(2, 2), activation='relu', padding='same')(E5)
    # E7 = Flatten()(E6)
    E7 = Flatten()(E4)
    # E_fin = Dense(latent_size, activation='relu', activity_regularizer=keras.regularizers.l2(10e-3))(E7)
    E_fin = Dense(latent_size, activation=activ)(E7)

    # decoder
    # D0 = Dense(196, activation='relu')(E_fin)
    D0 = Dense(7*7*16, activation=activ)(E_fin)
    # D1 = Reshape((7, 7, 4))(D0)
    D1 = Reshape((7, 7, 16))(D0)
    # D2 = Conv2D(4, kernel_size=(2, 2), activation='relu', padding='same')(D1)
    # D3 = Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same')(D2)
    # D4 = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(D3)
    D4 = Conv2D(16, kernel_size=(3, 3), activation=activ, padding='same')(D1)
    D5 = UpSampling2D(size=(2, 2))(D4)
    D6 = Conv2D(8, kernel_size=(5, 5), activation=activ, padding='same')(D5)
    D7 = UpSampling2D(size=(2, 2))(D6)
    D_fin = Conv2D(1, kernel_size=(5, 5), activation=activ, padding='same')(D7)

    model = keras.models.Model(inputs=inp, outputs=D_fin)
    model.compile(loss='mse', optimizer='adam')
    model.summary()
    model.fit(X_train[..., np.newaxis], X_train[..., np.newaxis],
              batch_size=batch_size, epochs=num_epochs,
              validation_data=(X_test[..., np.newaxis], X_test[..., np.newaxis]), verbose=2)

    model.save(fig_path + 'autoencoder_z{}.h5'.format(latent_size))
    ims = X_train[:25]
    # plt.figure()
    # for i in range(25):
    #     plt.subplot(5, 5, i+1)
    #     plt.imshow(ims[i], cmap='gray')
    #     plt.axis('off')
    # plt.tight_layout(h_pad=0, w_pad=0)
    # plt.savefig(fig_path + 'ae_orignals_z{}.png'.format(latent_size))

    recon = model.predict(ims[:, :, :, None])[:, :, :, 0]
    plt.figure()
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.imshow(recon[i], cmap='gray')
        plt.axis('off')
    plt.tight_layout(h_pad=0, w_pad=0)
    plt.savefig(fig_path + 'ae_reconstruction_z{}.png'.format(latent_size))


def plot_with_images(X, images, fig, ax, image_num=25, x_size=None):
    '''
    A plot function for viewing images in their embedded locations. The
    function receives the embedding (X) and the original images (images) and
    plots the images along with the embeddings.

    :param X: Nxd embedding matrix (after dimensionality reduction).
    :param images: NxD original data matrix of images.
    :param title: The title of the plot.
    :param num_to_plot: Number of images to plot along with the scatter plot.
    :return: the figure object.
    '''

    n, pixels = np.shape(images)
    img_size = int(pixels**0.5)

    if x_size is None:
        # get the size of the embedded images for plotting:
        x_size = (max(X[:, 0]) - min(X[:, 0])) * 0.08
    # y_size = (max(X[:, 1]) - min(X[:, 1])) * 0.08
    y_size = x_size

    # draw random images and plot them in their relevant place:
    for i in range(image_num):
        img_num = np.random.choice(n)
        x0, y0 = X[img_num, 0] - x_size / 2., X[img_num, 1] - y_size / 2.
        x1, y1 = X[img_num, 0] + x_size / 2., X[img_num, 1] + y_size / 2.
        img = images[img_num, :].reshape(img_size, img_size)
        ax.imshow(img, aspect='auto', cmap=plt.cm.gray, zorder=100000,
                  extent=(x0, x1, y0, y1))

    # draw the scatter plot of the embedded data points:
    ax.scatter(X[:, 0], X[:, 1], s=10, marker='.', alpha=0.9)

    return fig


def mosaic(images, reshape: tuple=None, gap: int=1,
           normalize: bool=True, clip: bool=False, cols: int=-1):
    def _factors(num: int):
        return np.where((num % np.arange(1, np.floor(np.sqrt(num) + 1))) == 0)[0] + 1
    if cols > 0: assert len(images) % cols == 0, 'Bad number of columns given to mosaic'
    else: cols = len(images)//_factors(len(images))[-1]
    ims = images

    if normalize:
        ims = [(I-np.min(I))/(np.max(I)-np.min(I)) if np.max(I)!=np.min(I) else I for I in ims]

    if clip:
        ims = [np.clip(I, 0, 1) for I in ims]

    if reshape is not None:
        ims = [I.reshape(reshape) for I in ims]

    max_val = np.max(ims) if not clip else np.max(ims)/np.max(ims)

    if gap > 0:
        sh = (ims[0].shape[0], gap) if ims[0].ndim < 3 else (ims[0].shape[0], gap, 3)
        ims = [np.concatenate([max_val * np.ones(sh), I], axis=1) for I in ims]

    rows = [np.concatenate(ims[i*cols: (i+1)*cols], axis=1) for i in range(len(ims)//cols)]

    if gap > 0:
        sh = (gap, rows[0].shape[1]) if rows[0].ndim < 3 else (gap, rows[0].shape[1], 3)
        rows = [np.concatenate([max_val * np.ones(sh), I], axis=0) for I in rows]

    ret = np.concatenate(rows, axis=0)

    if gap > 0:
        sh = (gap, ret.shape[1]) if ims[0].ndim < 3 else (gap, ret.shape[1], 3)
        ret = np.concatenate([ret, max_val*np.ones(sh)], axis=0)
        sh = (ret.shape[0], gap) if ims[0].ndim < 3 else (ret.shape[0], gap, 3)
        ret = np.concatenate([ret, max_val*np.ones(sh)], axis=1)

    return ret


def MNIST_PCA():
    (X, y), _ = mnist.load_data()
    print('Training PCA on {} MNIST images'.format(X.shape[0]), flush=True)
    flattened = X.reshape((X.shape[0], np.prod(X.shape[1:])))
    im_shape = X.shape[1:]

    choices = np.random.choice(X.shape[0], 50, replace=False)
    plt.figure(figsize=(10, 16))
    plt.imshow(mosaic(X[choices], gap=1, normalize=True), cmap='gray')
    plt.axis('off')
    plt.savefig(fig_path + 'MNIST_examples.png')

    pca = PCA(n_components=2)
    latent = pca.fit_transform(flattened)
    latent = (latent - np.mean(latent, axis=0)[None, :])/\
             np.std(latent - np.mean(latent, axis=0)[None, :], axis=0)[None, :]
    plt.figure()
    for i in range(10):
        inds = np.where(y == i)[0]
        inds = np.random.choice(inds, 300, replace=False)
        plt.scatter(latent[inds, 0], latent[inds, 1], s=10, label=str(i))
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path + 'MNIST_latent.png')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plot_with_images(latent[y == 0], flattened[y == 0], fig, ax, image_num=15)
    plot_with_images(latent[y == 1], flattened[y == 1], fig, ax, image_num=15)
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.savefig(fig_path + 'MNIST_meaning.png')

    ims = []
    names = []

    new_inds = np.random.choice(X.shape[0], 10, replace=False)

    ims2 = pca.inverse_transform(latent).reshape(X.shape)[new_inds]
    ims.append(mosaic(ims2, normalize=True, cols=1))
    names.append('z=2')

    pca5 = PCA(n_components=5)
    lat5 = pca5.fit_transform(flattened)
    ims5 = pca5.inverse_transform(lat5).reshape(X.shape)[new_inds]
    ims.append(mosaic(ims5, normalize=True, cols=1))
    names.append('z=5')

    pca25 = PCA(n_components=25)
    lat25 = pca25.fit_transform(flattened)
    ims25 = pca25.inverse_transform(lat25).reshape(X.shape)[new_inds]
    ims.append(mosaic(ims25, normalize=True, cols=1))
    names.append('z=25')

    pca50 = PCA(n_components=50)
    lat50 = pca50.fit_transform(flattened)
    ims50 = pca50.inverse_transform(lat50).reshape(X.shape)[new_inds]
    ims.append(mosaic(ims50, normalize=True, cols=1))
    names.append('z=50')

    pca150 = PCA(n_components=150)
    lat150 = pca150.fit_transform(flattened)
    ims150 = pca150.inverse_transform(lat150).reshape(X.shape)[new_inds]
    ims.append(mosaic(ims150, normalize=True, cols=1))
    names.append('z=150')

    pca500 = PCA(n_components=500)
    lat500 = pca500.fit_transform(flattened)
    ims500 = pca500.inverse_transform(lat500).reshape(X.shape)[new_inds]
    ims.append(mosaic(ims500, normalize=True, cols=1))
    names.append('z=500')

    ims.append(mosaic(X[new_inds], normalize=True, cols=1))
    names.append('Original')

    plt.figure()
    for i, im in enumerate(ims):
        plt.subplot(1, len(ims), i+1)
        plt.imshow(im, cmap='gray')
        plt.axis('off')
        plt.title(names[i])
    plt.tight_layout()
    plt.savefig(fig_path + 'reconstruction.png')


def MNIST_AE():
    (X, y), _ = mnist.load_data()
    X = X.astype('float32') / 255

    ae2 = keras.models.load_model('presentation/autoencoder_z2.h5')
    latent = get_encoded(X[..., None], ae2)
    latent = (latent - np.mean(latent, axis=0)[None, :]) / \
             np.std(latent - np.mean(latent, axis=0)[None, :], axis=0)[None, :]

    plt.figure()
    for i in range(10):
        inds = np.where(y == i)[0]
        inds = np.random.choice(inds, 300, replace=False)
        plt.scatter(latent[inds, 0], latent[inds, 1], s=10, label=str(i))
    plt.xlabel('z_1')
    plt.ylabel('z_2')
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path + 'MNIST_latent_ae.png')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.axis('equal')
    flattened = X.reshape((X.shape[0], np.prod(X.shape[1:])))
    plot_with_images(latent[y == 6], flattened[y == 6], fig, ax, image_num=15, x_size=.3)
    plot_with_images(latent[y == 8], flattened[y == 8], fig, ax, image_num=15, x_size=.3)
    plt.xlabel('z 1')
    plt.ylabel('z 2')
    plt.savefig(fig_path + 'MNIST_meaning_ae.png')

    ims = []
    names = []

    new_inds = np.random.choice(X.shape[0], 10, replace=False)

    ims2 = [im for im in ae2.predict(X[new_inds, ..., None])[:, :, :, 0]]
    ims.append(mosaic(ims2, normalize=True, cols=1))
    names.append('z=2')

    ae5 = keras.models.load_model('presentation/autoencoder_z5.h5')
    ims5 = [im for im in ae5.predict(X[new_inds, ..., None])[:, :, :, 0]]
    ims.append(mosaic(ims5, normalize=True, cols=1))
    names.append('z=5')

    ae25 = keras.models.load_model('presentation/autoencoder_z25.h5')
    ims25 = [im for im in ae25.predict(X[new_inds, ..., None])[:, :, :, 0]]
    ims.append(mosaic(ims25, normalize=True, cols=1))
    names.append('z=25')

    ae50 = keras.models.load_model('presentation/autoencoder_z50.h5')
    ims50 = [im for im in ae50.predict(X[new_inds, ..., None])[:, :, :, 0]]
    ims.append(mosaic(ims50, normalize=True, cols=1))
    names.append('z=50')

    ims.append(mosaic(X[new_inds], normalize=True, cols=1))
    names.append('Original')

    plt.figure()
    for i, im in enumerate(ims):
        plt.subplot(1, len(ims), i+1)
        plt.imshow(im, cmap='gray')
        plt.axis('off')
        plt.title(names[i])
    plt.tight_layout()
    plt.savefig(fig_path + 'reconstruction_ae.png')


if __name__ == '__main__':
    # MNIST_PCA()

    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # preprocess data
    n_classes = 10
    test_labels = y_test
    X_train = X_train.astype('float32')/255
    X_test = X_test.astype('float32')/255
    y_train = np_utils.to_categorical(y_train, n_classes)
    y_test = np_utils.to_categorical(y_test, n_classes)

    train_amnt, test_amnt = 60000, 1000
    data = ((X_train[:train_amnt], y_train[:train_amnt]),
            (X_test[:test_amnt], y_test[:test_amnt]))
    # conv_autoencoder(data, num_epochs=20, latent_size=2, batch_size=16)
    # conv_autoencoder(data, num_epochs=10, latent_size=5, batch_size=32)
    # conv_autoencoder(data, num_epochs=10, latent_size=25, batch_size=32)
    # conv_autoencoder(data, num_epochs=10, latent_size=50, batch_size=32)
    MNIST_AE()

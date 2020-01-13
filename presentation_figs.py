from sklearn.decomposition import PCA
import numpy as np
from matplotlib import pyplot as plt
import keras
from keras.datasets import mnist
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, UpSampling2D, Reshape
from keras.utils import np_utils


fig_path = '/cs/labs/yweiss/roy.friedmam/DP_project/presentation/'


def conv_autoencoder(data, num_epochs=2, batch_size=32, im_size=28, latent_size=50,
                     save_path=''):
    (X_train, y_train), (X_test, y_test) = data

    encoder = []

    inp = keras.layers.Input((im_size, im_size, 1))
    # encoder
    encoder.append(Conv2D(8, kernel_size=(5, 5), activation='relu', padding='same'))
    encoder.append(MaxPooling2D(pool_size=(2, 2)))
    encoder.append(Conv2D(16, kernel_size=(5, 5), activation='relu', padding='same'))
    encoder.append(MaxPooling2D(pool_size=(2, 2)))
    encoder.append(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
    encoder.append(Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same'))
    encoder.append(Conv2D(4, kernel_size=(2, 2), activation='relu', padding='same'))
    encoder.append(Flatten())
    # encoder.append(Dense(latent_size, activation='relu', activity_regularizer=keras.regularizers.l2(10e-3)))
    encoder.append(Dense(latent_size, activation='relu'))

    # decoder
    decoder = []
    decoder.append(Dense(196, activation='relu'))
    decoder.append(Reshape((7, 7, 4)))
    decoder.append(Conv2D(4, kernel_size=(2, 2), activation='relu', padding='same'))
    decoder.append(Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same'))
    decoder.append(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
    decoder.append(UpSampling2D(size=(2, 2)))
    decoder.append(Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same'))
    decoder.append(UpSampling2D(size=(2, 2)))
    decoder.append(Conv2D(8, kernel_size=(3, 3), activation='relu', padding='same'))
    decoder.append(Conv2D(1, kernel_size=(2, 2), activation='relu', padding='same'))

    x = inp
    for e in encoder:
        x = e(x)
    for d in decoder:
        x = d(x)

    model = keras.models.Model(inputs=inp, outputs=x)
    model.compile(loss='mse', metrics=['loss'],
                  optimizer='adam')
    model.summary()
    # keras.utils.plot_model(model, 'conv_autoencoder_model.png', show_shapes=True, show_layer_names=False)

    # encoder_hist = model.fit(X_train[..., np.newaxis], X_train[..., np.newaxis],
    #                          batch_size=batch_size, epochs=num_epochs,
    #                          validation_data=(X_test[..., np.newaxis], X_test[..., np.newaxis]), verbose=2)
    #
    # model.save(save_path + 'autoencoder.h5')
    #
    # encoder = keras.models.Model(inputs=inp, outputs=E_fin)
    # encoder.compile(loss='mse', metrics='loss', optimizer='adam')
    # encoder.save(save_path + 'encoder.h5')
    #
    # d_inp = keras.layers.Input((latent_size, 1))
    # decoder = keras.models.Model(inputs=D_inp, outputs=D_fin)
    # decoder.compile(loss='mse', metrics='loss', optimizer='adam')
    # decoder.save(save_path + 'decoder.h5')
    #
    # ims = X_train[:25]
    # plt.figure()
    # for i in range(25):
    #     plt.subplot(5, 5, i+1)
    #     plt.imshow(ims[i], cmap='gray')
    #     plt.axis('off')
    # plt.tight_layout(h_pad=0, w_pad=0)
    # plt.savefig('orignals.png')
    #
    # recon = model.predict(ims[:, :, :, None])[:, :, :, 0]
    # plt.figure()
    # for i in range(25):
    #     plt.subplot(5, 5, i + 1)
    #     plt.imshow(recon[i], cmap='gray')
    #     plt.axis('off')
    # plt.tight_layout(h_pad=0, w_pad=0)
    # plt.savefig('reconstruction.png')


def plot_with_images(X, images, fig, ax, image_num=25):
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


def _factors(num: int):
    return np.where((num % np.arange(1, np.floor(np.sqrt(num) + 1))) == 0)[0] + 1


def mosaic(images, reshape: tuple=None, gap: int=1,
           normalize: bool=True, clip: bool=False, cols: int=-1):
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

    train_amnt, test_amnt = 6000, 1000
    data = ((X_train[:train_amnt], y_train[:train_amnt]),
            (X_test[:test_amnt], y_test[:test_amnt]))
    conv_autoencoder(data)
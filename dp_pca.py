from sklearn.decomposition import PCA
import numpy as np
from matplotlib import pyplot as plt
from typing import Union
import tensorflow as tf
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train/255.0, x_test/255.0

_imarray = Union[list, np.ndarray]


def _factors(num: int):
    return np.where((num % np.arange(1, np.floor(np.sqrt(num) + 1))) == 0)[0] + 1


def mosaic(images: _imarray):
    ims = images[:25]
    cols = 5
    rows = [np.concatenate(ims[i*cols: (i+1)*cols], axis=1) for i in range(len(ims)//cols)]
    ret = np.concatenate(rows, axis=0)
    return ret


def PCA_privatize(images, latent_dim: int=50, eps: float=.1, ret_latent: bool=False):
    if type(images) is list: images = np.array(images)
    sz = images.shape
    if images.ndim > 2: images = images.reshape((images.shape[0], np.prod(images.shape[1:])))
    pca = PCA(n_components=latent_dim)
    latent = pca.fit_transform(images)
    mu, std = np.mean(latent, axis=0), np.std(latent, axis=0)
    latent = (latent - mu[None, :]) / std[None, :]
    latent += eps*np.random.randn(latent.shape[0], latent.shape[1])
    latent = latent*std[None, :] + mu[None, :]
    if not ret_latent: return pca.inverse_transform(latent).reshape(sz)
    else: return pca.inverse_transform(latent).reshape(sz), latent


if __name__ == '__main__':
    sz = 64
    f_path1 = "C:/Users/Roy/University/UniGoogleDrive/Studies" \
             "/Master/project/CelebA/numpy_CelebA/CelebA_c1_{}.npy".format(sz)
    f_path2 = "C:/Users/Roy/University/UniGoogleDrive/Studies" \
              "/Master/project/CelebA/numpy_CelebA/CelebA_c2_{}.npy".format(sz)
    ims1 = np.load(f_path1)
    ims2 = np.load(f_path2)
    ims = np.concatenate([ims1, ims2])
    labels = np.zeros(ims.shape[0])
    labels[ims1.shape[0]:] = 1
    N = ims.shape[0]
    ims = ims.reshape((N, sz, sz, 3))

    # plt.figure()
    # plt.imshow(mosaic(ims))
    # plt.axis('off')
    # plt.savefig('PCA/original images.png')
    #
    # ims_noised = PCA_privatize(ims, eps=.01)
    # plt.figure()
    # plt.imshow(mosaic(ims_noised))
    # plt.axis('off')
    # plt.savefig('PCA/eps=01.png')
    #
    # ims_noised = PCA_privatize(ims, eps=.1, latent_dim=150)
    # plt.figure()
    # plt.imshow(mosaic(ims_noised))
    # plt.axis('off')
    # plt.savefig('PCA/eps=1.png')
    #
    # ims_noised = PCA_privatize(ims, eps=.2, latent_dim=150)
    # plt.figure()
    # plt.imshow(mosaic(ims_noised))
    # plt.axis('off')
    # plt.savefig('PCA/eps=2.png')
    #
    # ims_noised = PCA_privatize(ims, eps=.5, latent_dim=150)
    # plt.figure()
    # plt.imshow(mosaic(ims_noised))
    # plt.axis('off')
    # plt.savefig('PCA/eps=5.png')

    ims_noised, lat = PCA_privatize(ims, eps=.1, latent_dim=2, ret_latent=True)
    plt.figure()
    plt.scatter(lat[:, 0], lat[:, 1], c=labels, s=10)
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.grid()
    plt.savefig('PCA/latent.png')

    plt.show()

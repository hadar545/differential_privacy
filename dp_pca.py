# from sklearn.decomposition import PCA
import numpy as np
from matplotlib import pyplot as plt
from typing import Union
import pickle
from utils import *
import datetime


def _rmse(x: np.ndarray, y: np.ndarray, y_is_mean: bool=False):
    flat_shape = (x.shape[0], np.prod(x.shape[1:]))
    if not y_is_mean: return np.sqrt(np.mean((x.reshape(flat_shape) - y.reshape(flat_shape))**2, axis=1))
    return np.sqrt(np.mean((x.reshape(flat_shape) - y.reshape(np.prod(y.shape))[None, :])**2, axis=1))


def mosaic(images: Union[list, np.ndarray], reshape: tuple=None, gap: int=1,
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


class pPCA:

    def __init__(self, latent_dim: int=50):
        """
        Initialize a probabilistic PCA model with the given latent dimension
        :param latent_dim: an int depicting the latent dimension the model should learn
        """
        self.latent = latent_dim
        self.mu, self.W, self.phi = np.array([]), np.array([]), np.array([])
        self.M, self.M_inv = np.array([]), np.array([])
        self._shape, self._d = [], 0
        self._trained = False

    def __str__(self): return 'pPCA_z{}'.format(self.latent)

    def __repr__(self): return 'pPCA_z{}'.format(self.latent)

    def _update_M(self):
        """
        Updates inner parameters (which shouldn't be needed by anyone outside of the class)
        """
        self.M = self.W.T @ self.W
        self.M.flat[::self.M_inv.shape[0] + 1] += self.phi
        self.M_inv = np.linalg.inv(self.M)

    def fit(self, X: np.ndarray, y=None):
        """
        Fit the probabilistic PCA model to the given data.
        :param X: a numpy array with dimensions [N, ...] where N is the number of samples to fit to
        :param y: not used (defaults to None) - a parameter to fit the API of a standard sklearn model
        :return: the fitted model
        """
        self._shape = list(X.shape[1:])
        print('Fitting a pPCA model to {} samples with latent dimension {}'.format(X.shape[0], self.latent))
        self._d = np.prod(self._shape)
        if X.ndim > 2:
            X = X.copy().reshape((X.shape[0], np.prod(X.shape[1:])))
        self.mu = X.sum(axis=0)/X.shape[0]
        _, s, v = np.linalg.svd(X - self.mu, full_matrices=False)
        sig = (s ** 2) / X.shape[0]
        self.phi = np.maximum(np.sum(sig[self.latent:]) / (self._d - self.latent), 10e-8)
        self.W = v[:self.latent, :].T * np.sqrt(sig[:self.latent] - self.phi)[None, :]
        self._update_M()
        self._trained = True
        return self

    def fit_transform(self, X: np.ndarray, y=None):
        """
        Fit a probabilistic PCA model to the given data and return the encodings of the data
        :param X: a numpy array with dimensions [N, ...] where N is the number of samples to fit to
        :param y: not used (defaults to None) - a parameter to fit the API of a standard sklearn model
        :return: the encoded data points of X as a [N, latent_dim] numpy array
        """
        self.fit(X)
        self._trained = True
        return self.encode(X)

    def encode(self, X: np.ndarray):
        """
        Encode the datapoints of X using the learned pPCA model. If the model wasn't trained before calling
        this step, it is trained on the data points of X (essentially the same as calling
        model.fit_transform(X))
        :param X: a numpy array with dimensions [N, ...] where N is the number of samples to fit to
        :return: the encoded data points of X as a [N, latent_dim] numpy array
        """
        if not self._trained:
            self._trained = True
            return self.fit_transform(X)
        Z = X.copy().reshape((X.shape[0], np.prod(X.shape[1:])))
        return (self.M_inv @ self.W.T @ (Z.T - self.mu[:, None])).T

    def decode(self, Z: np.ndarray):
        """
        Decode the latent vectors Z into the original space
        :param Z: a numpy array with dimensions [N, latent_dim] where N is the number of samples to decode
        :return: a numpy array with dimensions [N, ...] where '...' stand for the original dimensions of
                 the data (the model reshapes the decoded vectors to the shapes of the original data)
        """
        assert self._trained, "Model must be trained before trying to decode"
        inved = self.W.T @ self.W
        inved.flat[::inved.shape[0] + 1] += 10e-8
        inved = np.linalg.inv(inved)
        X = (self.W @ inved @ self.M @ Z.T).T + self.mu[None, :]
        return X.reshape(np.hstack([X.shape[0], self._shape]))

    def generate(self, N: int):
        """
        Generate data from the learned model
        :param N: number of samples to generate
        :return: a numpy array of dimension [N, ...] of generated samples (where '...' stands for
                 the original shape of the data the model was trained on)
        """
        assert self._trained, "Model must be trained before trying to generate data from it"
        return self.decode(np.random.randn(N, self.latent))

    def privatize(self, X: np.ndarray, noise: float=.1, return_encodings: bool=False):
        """
        Privatize the datapoints in X by using a DP Gaussian Mechanism on the latent encodings of X as
        trained by the model. If the model was not trained before trying to privatize X, it will be trained.
        :param X: a numpy array with dimensions [N, ...] where N is the number of samples to fit to
        :param noise: the amount of noise to add to the encodings using the Gaussian mechanism on the
                      latent encodings of the data
        :param return_encodings: a boolean indicating whether the encodings and their noisy representations
                                 should be returned as well
        :return: if return_encodings=False, only the privatized images as a [N, ...] dimensional numpy array is
                 returned (where '...' stands for the original shape of each image). Otherwise, Z, Z_noisy and
                 the privatized images are returned, where Z are the original encodings, Z_noisy are the noised
                 encodings and both Z as well as Z_noisy have the dimensions of [N, latent_dim]
        """
        if self._trained: Z = self.encode(X)
        else: Z = self.fit_transform(X)
        Z_noise = Z + np.sqrt(noise)*np.random.randn(Z.shape[0], Z.shape[1])
        if return_encodings:
            return Z, Z_noise, self.decode(Z_noise)
        return self.decode(Z_noise)

    def prop_privatize(self, X: np.ndarray, noise: float=.1, return_encodings: bool=False):
        """
        Privatize the data points in X by using a DP Gaussian Mechanism on the latent encodings of X as
        trained by the model. If the model was not trained before trying to privatize X, it will be trained.
        :param X: a numpy array with dimensions [N, ...] where N is the number of samples to fit to
        :param noise: the amount of noise to add to the encodings using the Gaussian mechanism on the
                      latent encodings of the data
        :param return_encodings: a boolean indicating whether the encodings and their noisy representations
                                 should be returned as well
        :return: if return_encodings=False, only the privatized images as a [N, ...] dimensional numpy array is
                 returned (where '...' stands for the original shape of each image). Otherwise, Z, Z_noisy and
                 the privatized images are returned, where Z are the original encodings, Z_noisy are the noised
                 encodings and both Z as well as Z_noisy have the dimensions of [N, latent_dim]
        """
        if self._trained: Z = self.encode(X)
        else: Z = self.fit_transform(X)
        Z_noise = (Z + np.sqrt(noise)*np.random.randn(Z.shape[0], Z.shape[1]))/np.sqrt(1 + noise)
        if return_encodings:
            return Z, Z_noise, self.decode(Z_noise)
        return self.decode(Z_noise)

    def wiener_privatize(self, X: np.ndarray, noise: float=.1, return_encodings: bool=False):
        """
        Privatize the data points in X by using a DP Gaussian Mechanism on the latent encodings of X as
        trained by the model. If the model was not trained before trying to privatize X, it will be trained.
        :param X: a numpy array with dimensions [N, ...] where N is the number of samples to fit to
        :param noise: the amount of noise to add to the encodings using the Gaussian mechanism on the
                      latent encodings of the data
        :param return_encodings: a boolean indicating whether the encodings and their noisy representations
                                 should be returned as well
        :return: if return_encodings=False, only the privatized images as a [N, ...] dimensional numpy array is
                 returned (where '...' stands for the original shape of each image). Otherwise, Z, Z_noisy and
                 the privatized images are returned, where Z are the original encodings, Z_noisy are the noised
                 encodings and both Z as well as Z_noisy have the dimensions of [N, latent_dim]
        """
        if self._trained: Z = self.encode(X)
        else: Z = self.fit_transform(X)
        Z_noise = (Z + np.sqrt(noise)*np.random.randn(Z.shape[0], Z.shape[1]))/(1+noise)
        if return_encodings:
            return Z, Z_noise, self.decode(Z_noise)
        return self.decode(Z_noise)

    def save(self, path: str=None):
        """
        Saves a pPCA model
        :param path: the path to save the model to; if no path was given, the model is saved in the current
                     directory with a default name format "pPCA_z<latent_dim>.pkl". The model is saved
                     using pickle
        """
        params = {'mu': self.mu.copy(),
                  'W': self.W.copy(),
                  'phi': self.phi.copy(),
                  'latent': self.latent,
                  'shape': self._shape.copy(),
                  'd': self._d}
        if path is None: path = str(self)
        np.savez(path, **params)
        print('Saved model as file {}'.format(path))

    @staticmethod
    def load(path: str):
        """
        Loads a pPCA model
        :param path: the path to load the model from
        :return: the model if loading was successful, otherwise raises an error
        """
        try:
            params = np.load(path)
            mod = pPCA(latent_dim=params['latent'])
            mod._d, mod._shape = params['d'], params['shape']
            mod.mu, mod.W, mod.phi = params['mu'], params['W'], params['phi']
        except:
            raise Exception('Either the path {} is not a saved pPCA model or '
                            'there was a problem reading the file.'.format(path))
        mod._update_M()
        mod._trained = True
        return mod


def create_models(images, latent_dims=(2, 5, 10, 25, 50, 100, 200, 300, 500, 750), save_path="models/pPCA/",
                  fit=True):
    from pathlib import Path
    Path(save_path).mkdir(parents=True, exist_ok=True)
    model_paths = {i: "" for i in latent_dims}
    for l in latent_dims:
        mod = pPCA(latent_dim=l)
        if fit:
            mod = mod.fit(images)
            mod.save(save_path + str(mod))
        model_paths[l] = save_path + str(mod) + '.npz'
    return model_paths


def reconstruction_errs(images, model_paths: dict, save_path="models/pPCA/"):
    chosen = np.random.choice(images.shape[0], 10, replace=False)
    latent_dims = list(model_paths.keys())
    recon_images = []
    rmse_mean = []
    rmse_std = []

    for l in latent_dims:
        print(l)
        mod = pPCA.load(model_paths[l])
        rec = mod.decode(mod.encode(images))
        recon_images.append(mosaic(rec[chosen], normalize=False, clip=True, cols=1))
        m = _rmse(rec, images)
        rmse_mean.append(np.mean(m))
        rmse_std.append(np.std(m))

    plt.figure()
    plt.subplot(1, len(recon_images)+1, 1)
    plt.imshow(mosaic(images[chosen], normalize=False, clip=True, cols=1))
    plt.axis('off')
    # plt.title('originals')
    for i, im in enumerate(recon_images):
        plt.subplot(1, len(recon_images)+1, i+2)
        plt.imshow(im)
        plt.axis('off')
        # plt.title('z = ' + str(latent_dims[i]))
    # plt.tight_layout()
    plt.savefig(save_path + 'reconstructed_images.png', dpi=300)

    plt.figure()
    plt.errorbar(latent_dims, rmse_mean, rmse_std, capsize=5, elinewidth=2, lw=2)
    plt.xlabel('size of latent dimension')
    plt.ylabel('RMSE')
    plt.savefig(save_path + 'reconstructed_stats.png', dpi=300)


def privatization(images, model_paths: dict, save_path="models/pPCA/", latent_dim=300,
                  noise=(0, .001, .01, .1, .15, .3, .5, .75, 1, 1.5)):
    chosen = np.random.choice(images.shape[0], 10, replace=False)

    privatized = []
    prop_privatized = []
    wien = []

    priv_rmse = []
    priv_rmse_std = []
    priv_mean = []
    priv_mean_std = []

    prop_rmse = []
    prop_rmse_std = []
    prop_mean = []
    prop_mean_std = []

    mod = pPCA.load(model_paths[latent_dim])
    for e in noise:
        print(e)
        priv = mod.privatize(images, noise=e)
        prop = mod.prop_privatize(images, noise=e)
        privatized.append(mosaic(priv[chosen], normalize=False, clip=True, cols=1))
        prop_privatized.append(mosaic(prop[chosen], normalize=False, clip=True, cols=1))
        wien.append(mosaic(mod.wiener_privatize(images[chosen], noise=e), normalize=False, clip=True, cols=1))

        m = _rmse(priv, images)
        priv_rmse.append(np.mean(m))
        priv_rmse_std.append(np.std(m))
        m = _rmse(priv, mod.mu, y_is_mean=True)
        priv_mean.append(np.mean(m))
        priv_mean_std.append(np.std(m))

        m = _rmse(prop, images)
        prop_rmse.append(np.mean(m))
        prop_rmse_std.append(np.std(m))
        m = _rmse(prop, mod.mu, y_is_mean=True)
        prop_mean.append(np.mean(m))
        prop_mean_std.append(np.std(m))

    plt.figure()
    plt.subplot(1, len(privatized)+1, 1)
    plt.imshow(mosaic(images[chosen], normalize=False, clip=True, cols=1))
    plt.axis('off')
    # plt.title('originals')
    for i, im in enumerate(privatized):
        plt.subplot(1, len(privatized)+1, i+2)
        plt.imshow(im)
        plt.axis('off')
        # plt.title(r'$\epsilon = ' + str(noise[i]) + '$')
    plt.savefig(save_path + 'privatized_images.png', dpi=300)

    plt.figure()
    plt.subplot(1, len(wien)+1, 1)
    plt.imshow(mosaic(images[chosen], normalize=False, clip=True, cols=1))
    plt.axis('off')
    # plt.title('originals')
    for i, im in enumerate(wien):
        plt.subplot(1, len(wien)+1, i+2)
        plt.imshow(im)
        plt.axis('off')
        # plt.title(r'$\epsilon = ' + str(noise[i]) + '$')
    plt.savefig(save_path + 'wien_images.png', dpi=300)

    plt.figure()
    plt.subplot(1, len(prop_privatized)+1, 1)
    plt.imshow(mosaic(images[chosen], normalize=False, clip=True, cols=1))
    plt.axis('off')
    # plt.title('originals')
    for i, im in enumerate(prop_privatized):
        plt.subplot(1, len(prop_privatized)+1, i+2)
        plt.imshow(im)
        plt.axis('off')
        # plt.title(r'$\epsilon = ' + str(noise[i]) + '$')
    plt.savefig(save_path + 'prop_privatized_images.png', dpi=300)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.errorbar(noise, priv_rmse, priv_rmse_std, capsize=5, elinewidth=2, lw=2, label='pPCA')
    plt.errorbar(noise, prop_rmse, prop_rmse_std, capsize=5, elinewidth=2, lw=2, label='npPCA')
    plt.xlabel('standard deviation of added noise')
    plt.ylabel('RMSE from originals')

    plt.subplot(1, 2, 2)
    plt.errorbar(noise, priv_mean, priv_mean_std, capsize=5, elinewidth=2, lw=2, label='pPCA')
    plt.errorbar(noise, prop_mean, prop_mean_std, capsize=5, elinewidth=2, lw=2, label='npPCA')
    plt.xlabel('standard deviation of added noise')
    plt.ylabel('RMSE from mean image')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path + 'privatization_stats.png')


def save_noisy(images, model_paths: dict, save_path="models/pPCA/noised/",
               noise=(0, .001, .01, .1, .15, .3, .5, .75, 1, 1.5, 2, 3)):
    from pathlib import Path
    Path(save_path).mkdir(parents=True, exist_ok=True)
    l_dims = list(model_paths.keys())
    np.save(save_path + 'original_images.npy', images)
    for l in l_dims:
        mod = pPCA.load(model_paths[l])
        print(mod)
        for e in noise:
            print('\t{}'.format(e))
            np.save(save_path + 'priv_{}_e{}.npy'.format(str(mod), e), mod.privatize(images, noise=e))
            np.save(save_path + 'prop_{}_e{}.npy'.format(str(mod), e), mod.prop_privatize(images, noise=e))


def simple_query(images, model_paths: dict, mid_amnt: int=15, save_path="models/pPCA/",
                 noise=(0.001, 0.01, .1, .5, 1, 2, 5), lat_dims=(50, 100, 300)):
    _query = lambda x: np.mean(x[:, mid-mid_amnt:mid+mid_amnt, mid-mid_amnt:mid+mid_amnt], axis=(1, 2))
    _bin_query = lambda q: np.sum(np.where(q > .5)[0])
    mid = images.shape[1]//2
    true_queries = _query(images)
    binary_queries = _bin_query(true_queries)
    _to_range = lambda x: np.clip((x - np.min(true_queries))/(np.max(true_queries) - np.min(true_queries)), 0, 1)
    true_queries = _to_range(true_queries)
    glob_acc = np.zeros((len(lat_dims), len(noise), 2))
    loc_acc = np.zeros((len(lat_dims), len(noise), 2))
    for i, l in enumerate(lat_dims):
        mod = pPCA.load(model_paths[l])
        for j, e in enumerate(noise):
            noised = _to_range(_query(mod.privatize(images, e)))
            n_noised = _to_range(_query(mod.prop_privatize(images, e)))
            loc_acc[i, j, 0] = 1 - np.mean(np.abs(true_queries - noised))
            loc_acc[i, j, 1] = 1 - np.mean(np.abs(true_queries - n_noised))

            glob_acc[i, j, 0] = 1 - np.abs(binary_queries - _bin_query(noised))
            glob_acc[i, j, 1] = 1 - np.abs(binary_queries - _bin_query(n_noised))

    for n, s in ['pPCA', 'npPCA']:
        plt.figure()
        for i, l in enumerate(lat_dims):
            plt.plot(noise, glob_acc[i, :, n], lw=2, label='global z={}'.format(l))
            plt.plot(noise, loc_acc[i, :, n], '--', lw=2, label='local z={}'.format(l))
        plt.xlabel('noise')
        plt.ylabel('accuracy')
        plt.xscale('log')
        plt.legend(loc='lower right')
        plt.savefig(save_path + '{}_simplequery_acc.png'.format(s))


if __name__ == '__main__':
    ims = np.load('/cs/labs/yweiss/roy.friedmam/full128_10k.npy')[:5000]/255.0
    N = ims.shape[0]

    paths = create_models(ims, fit=False)
    # reconstruction_errs(ims[:500], paths)
    # privatization(ims[:500], paths)
    # save_noisy(ims[:1000], model_paths=paths)
    simple_query(ims[:1000], model_paths=paths)

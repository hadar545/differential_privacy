from sklearn.decomposition import PCA
import numpy as np
from matplotlib import pyplot as plt
from typing import Union
import pickle


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
        print('Fitting a pPCA model to {} samples with latent dimension {}\n'
              'The original shape of the data points is {}'
              .format(X.shape[0], self.latent, X.shape[1:]))
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
        return X.reshape([X.shape[0]] + self._shape)

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
        assert 0 < noise < 1
        if self._trained: Z = self.encode(X)
        else: Z = self.fit_transform(X)
        Z_noise = (1-noise)*Z + noise*np.random.randn(Z.shape[0], Z.shape[1])
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
        params = {'mu': self.mu,
                  'W': self.W,
                  'phi': self.phi,
                  'latent': self.latent,
                  'shape': self._shape,
                  'd': self._d}
        if path is None: path = str(self) + '.pkl'
        with open(path, 'wb') as f:
            pickle.dump(params, f)
        print('Saved model as file {}'.format(path))

    @staticmethod
    def load(path: str):
        """
        Loads a pPCA model
        :param path: the path to load the model from
        :return: the model if loading was successful, otherwise raises an error
        """
        try:
            with open(path, 'rb') as f:
                params = pickle.load(f)
            mod = pPCA(latent_dim=params['latent'])
            mod._d, mod._shape = params['d'], params['shape']
            mod.mu, mod.W, mod.phi = params['mu'], params['W'], params['phi']
            mod._update_M()
            mod._trained = True
            return mod
        except:
            raise Exception('Either the path {} is not a saved pPCA model or '
                            'there was a problem reading the file.'.format(path))


if __name__ == '__main__':
    sz = 100
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
    # mod = pPCA(latent_dim=256).fit(ims)
    # mod.save()

    mod = pPCA.load('pPCA_z256.pkl')
    en = mod.encode(ims)
    ch = np.random.choice(en.shape[0], 2, replace=False)
    part = ims[:50]
    noise = .5
    noised = mod.privatize(part, noise=noise)
    prop_noised = mod.prop_privatize(part, noise=noise)

    # interp = np.array([(1-a)*en[ch[0]] + a*en[ch[1]] for a in np.linspace(0, 1.1, 50)])
    # plt.figure()
    # plt.imshow(mosaic(mod.decode(interp), normalize=False, clip=True))
    # plt.axis('off')
    # plt.show()

    plt.figure()
    plt.imshow(mosaic(part, normalize=False))
    plt.axis('off')
    plt.title('original images')

    plt.figure()
    plt.imshow(mosaic(mod.decode(mod.encode(part)), normalize=False, clip=True))
    plt.axis('off')
    plt.title('reconstructed images')

    plt.figure()
    plt.imshow(mosaic(noised, normalize=False, clip=True))
    plt.axis('off')
    plt.title('privatized images')
    plt.show()

    plt.figure()
    plt.imshow(mosaic(prop_noised, normalize=False, clip=True))
    plt.axis('off')
    plt.title('properly privatized images')
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
# from __future__ import absolute_import, division, print_function, unicode_literals
import os
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Reshape, UpSampling2D, InputLayer, BatchNormalization
from tensorflow.keras import Model
import datetime
import sys

import utils


EPOCHS = 1
LAMBDA = 0.00001
LATENT_SIZE = 2
BATCH_SIZE = 32
LEARNING_RATE = 0.001
SD = 0.1


class basic_AE(Model):

    def __init__(self, save_encodings=True, learning_rate=LEARNING_RATE):
        super(basic_AE, self).__init__()

        global LAMBDA, LATENT_SIZE, SD
        self.lambda_, self.latent_, self.sd_ = LAMBDA, LATENT_SIZE, SD

        self.save_encodings = save_encodings

        self.encodings = []
        self.noisy_encodings = []

        self.encoding_layers = self._create_encoder()
        self.decoding_layers = self._create_decoder()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')

    def __str__(self):
        return 'basic_AE_z{}_la{}'.format(self.latent_, self.lambda_)

    def _create_encoder(self) -> list:
        layers = []
        layers.append(Conv2D(8, 3, activation='relu', padding='same'))
        layers.append(MaxPooling2D())
        layers.append(Conv2D(16, 3, activation='relu', padding='same'))
        layers.append(MaxPooling2D())
        layers.append(Conv2D(8, 3, activation='relu', padding='same'))
        layers.append(Conv2D(4, 3, activation='relu', padding='same'))
        layers.append(Flatten())
        layers.append(Dense(self.latent_, activation='linear',
                            activity_regularizer=tf.keras.regularizers.l2(self.lambda_)))
        return layers

    def _create_decoder(self) -> list:
        layers = []
        layers.append(InputLayer((28,28,1)))
        # layers.append(InputLayer((self.latent_,1)))
        layers.append(Dense(196, activation='relu'))
        layers.append(Reshape((7, 7, 4)))
        layers.append(Conv2D(8, 3, activation='relu', padding='same'))
        layers.append(Conv2D(16, 3, activation='relu', padding='same'))
        layers.append(UpSampling2D())
        layers.append(Conv2D(8, 3, activation='relu', padding='same'))
        layers.append(UpSampling2D())
        layers.append(Conv2D(1, 3, activation='relu', padding='same'))
        return layers

    def call_noisy(self, x, return_encodings_too=False):
        z = self.encode(x)
        z_noised = z + self.sd_*np.random.randn(self.latent_)
        if self.save_encodings:
            self.encodings.append(z)
            self.noisy_encodings.append(z_noised)
        x_decoded = self.decode(z_noised)
        if return_encodings_too:
            return z, z_noised, x_decoded
        return x_decoded

    def call(self, z):
        z = self.encode(z)
        if self.save_encodings:
            self.encodings.append(z)
            self.noisy_encodings.append(z + self.sd_ * np.random.randn(self.latent_))
        x_decoded = self.decode(z)
        return x_decoded

    def encode(self, x):
        for l in self.encoding_layers:
            x = l(x)
        return x

    def decode(self, x):
        for l in self.decoding_layers:
            x = l(x)
        return x

    def mse_loss(self, images, labels=None):
        _cast = lambda x: tf.cast(x, tf.float32)
        return tf.reduce_mean(tf.square(tf.subtract(_cast(self.call(images)), _cast(images))))

    def train_step(self, images):
        with tf.GradientTape() as tape:
            gradients = tape.gradient(self.mse_loss(images), self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
            self.train_loss(self.mse_loss(images))

    def test_step(self, images):
        self.test_loss(self.mse_loss(images))

    def reset_metrics(self):
        self.train_loss.reset_states()
        self.test_loss.reset_states()

    def save_encodings_npy(self, save_dir):
        np.save(save_dir + '/encodings.npy', np.array(self.encodings))
        np.save(save_dir + '/encodings_noisy_sd{}.npy'.format(self.sd_), np.array(self.noisy_encodings))
        print('saved encodings at ' + save_dir)

    def save(self, model_dir):
        model_path = model_dir + '/model_weights'
        self.save_weights(model_path, save_format='tf')
        print('saved model at: ' + model_dir)

    def train_epoch(self, train_ds, test_ds, logger, iter_counter_start=0, epoch_counter=0):

        iter_counter = iter_counter_start

        for images, _ in train_ds:

            reconstructions = self.call(images)
            self.train_step(images)

            if iter_counter % 100 == 0:

                for test_images, _ in test_ds:
                    test_reconstructions = self.call(test_images)
                    self.test_step(test_images)

                logger.log(iter_counter, images, reconstructions, test_images, test_reconstructions)
                print('     Step {}, Loss: {}, Test Loss: {}'.format(iter_counter, self.train_loss.result(), self.test_loss.result()))

            iter_counter += 1

        template = 'finished epoch {}, Step {}, Loss: {:.5f}, Test Loss: {:.5f}     [time stamp: {}]'
        print(template.format(epoch_counter + 1, iter_counter,
                              self.train_loss.result(),
                              self.test_loss.result(),
                              datetime.datetime.now().strftime("%H:%M:%S")))

        # Reset the metrics for the next epoch
        self.reset_metrics()

        return iter_counter

    def call_on_images(self, imgs, labels=None):

        encodings, noisy_encodings, decoded_images = [], [], np.zeros_like(imgs)
        curr_i = 0
        def run(images):
            z, z_noised, x_decoded = self.call_noisy(images, return_encodings_too=True)
            encodings.append(z)
            noisy_encodings.append(z + self.sd_ * np.random.randn(self.latent_))
            decoded_images[curr_i] = x_decoded

        if labels is None:
            ds = tf.data.Dataset.from_tensor_slices(imgs).batch(1, drop_remainder=True)
            for images in ds:
                run(images)
                curr_i += 1
        else:
            ds = tf.data.Dataset.from_tensor_slices((imgs, labels)).batch(1, drop_remainder=True)
            for images, labels in ds:
                run(images)
                curr_i += 1

        return np.array(encodings).reshape(imgs.shape[0],self.latent_), \
               np.array(noisy_encodings).reshape(imgs.shape[0],self.latent_), \
               decoded_images

    def run_noisy_and_create_new_ds(self, ds, save_dir):

        self.encodings = []
        self.noisy_encodings = []
        self.reconstructions = None
        self.save_encodings = True

        for images, _ in ds:
            curr_reconstructions = np.array(self.call_noisy(images))
            if self.reconstructions is None:
                self.reconstructions = curr_reconstructions
            else:
                self.reconstructions = np.concatenate((self.reconstructions, curr_reconstructions), axis=0)

        self.save_encodings_npy(save_dir)
        np.save(save_dir + '/noisy_reconstructions_images.npy', np.clip(self.reconstructions, 0,1))
        print('saved noisy_reconstructions_images at ' + save_dir)

class ae_logger():

    def __init__(self, tb_dir, loss_func):

        self.time_of_creation = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tb_dir = utils.check_dir_exists(tb_dir)
        print("tensorboard logdir: {}".format(self.tb_dir + '/' + self.time_of_creation))
        print("     Reminder, to open tensorboard - open the link returned from the "
              "terminal command: %tensorboard --logdir {}".format(self.tb_dir + '/' + self.time_of_creation))

        self.loss_func = loss_func

        self.iter_log = []
        self.train_loss_log = []
        self.test_loss_log = []

        # tensorboard logs - Set up summary writers to write the summaries to disk
        # (reminder to myself) to open tensorboard - open the link returned from the terminal command: %tensorboard --logdir logs/gradient_tape
        self.train_log_dir = self.tb_dir + '/' + self.time_of_creation + '/train'
        self.test_log_dir = self.tb_dir + '/' + self.time_of_creation + '/test'
        self.train_summary_writer = tf.summary.create_file_writer(self.train_log_dir)
        self.test_summary_writer = tf.summary.create_file_writer(self.test_log_dir)

    def log(self, iter, images, reconstructions, test_images, test_reconstructions):
        self.iter_log.append(iter)
        train_loss, test_loss = self.loss_func(images), self.loss_func(test_images)
        self.train_loss_log.append(train_loss)
        self.test_loss_log.append(test_loss)
        with self.train_summary_writer.as_default():
            # tf.summary.scalar('loss', self.train_loss.result(), step=iter)
            tf.summary.scalar('loss', train_loss, step=iter)
            tf.summary.image('original', images, max_outputs=10, step=iter)
            tf.summary.image('reconstructed', reconstructions, max_outputs=10, step=iter)
        with self.test_summary_writer.as_default():
            # tf.summary.scalar('loss', self.test_loss.result(), step=iter)
            tf.summary.scalar('loss', test_loss, step=iter)
            tf.summary.image('original', test_images, max_outputs=10, step=iter)
            tf.summary.image('reconstructed', test_reconstructions, max_outputs=10, step=iter)


class ae_plotter():

    def __init__(self):
        self.images_before = None
        self.final_before_after_images_to_save_later = []

    def plot_loss(self, model, logger, do_save=1, save_dir=None, iter=None):
        plt.figure()
        plt.title(str(model) + ' - Loss per step\n' + logger.time_of_creation)
        plt.plot(logger.iter_log, logger.train_loss_log, 'b', label='train')
        plt.plot(logger.iter_log, logger.test_loss_log, 'r', label='test')
        plt.legend()
        plt.xlabel('iteration')
        plt.ylabel('loss')

        if do_save:
            if save_dir is not None:
                path_to_save = save_dir+'/loss.png'
                if iter is not None:
                    path_to_save = save_dir+'/loss_iter{}.png'.format(iter)
                plt.savefig(path_to_save)

        plt.show()

    def plot_encodings_2Dlatent(self, model, encodings=None, labels=None):

        if encodings is None:
            encodings, noisy_encodings = np.array(model.encodings), np.array(model.noisy_encodings)
        # else:
        #     encodings, noisy_encodings = model.encode(encodings), model.encode(encodings)

        if labels is None:
            plt.figure()
            sd_title = 'ax0_std=%.2f ax1_std=%.2f' % (np.std(encodings[:,0]), np.std(encodings[:,1]))
            plt.title(str(model) + ' - Encodings\n' + sd_title)
            plt.scatter(encodings[:, 0], encodings[:,1], s=10)
            plt.xlabel('x')
            plt.ylabel('y')

            plt.figure()
            sd_title = 'ax0_std=%.2f ax1_std=%.2f' % (np.std(noisy_encodings[:, 0]), np.std(noisy_encodings[:, 1]))
            plt.title(str(model) + ' - Noisy Encodings\n' + sd_title)
            plt.scatter(noisy_encodings[:, 0], noisy_encodings[:,1], s=10)
            plt.xlabel('x')
            plt.ylabel('y')
            plt.show()

        else:
            plt.figure()
            plt.scatter(encodings[:, 0], encodings[:, 1], s=10, c=labels, cmap='hsv')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title('')
            # plt.legend()
            plt.show()

    def plot_before_after(self, model, single_color=0, save_dir=None, iter=None, n_images=10, do_save=1, save_later=0):

        if self.images_before is None:
            indx = np.random.choice(BATCH_SIZE, n_images, replace=False)
            for imgs, _ in test_ds:
                self.images_before = np.array(imgs)[indx]
                break
        encodings, noisy_encodings, images_after = model.call_on_images(self.images_before)

        originals_wide = np.hstack(tuple([self.images_before[i] for i in range(self.images_before.shape[0])]))
        decoded_wide = np.hstack(tuple([images_after[i] for i in range(images_after.shape[0])]))
        before_after = np.clip(np.vstack((originals_wide, decoded_wide))[:,:,0 if single_color else 0:3], 0,1)
        plt.imshow(before_after)
        plt.axis('off')

        path_to_save = 'before_after.png'
        if save_dir is not None:
            path_to_save = save_dir + '/before_after.png'
            if iter is not None:
                path_to_save = save_dir + '/before_after_iter{}.png'.format(iter)

        if save_later:
            self.final_before_after_images_to_save_later.append((iter, before_after))

        if do_save and not save_later:
            plt.imsave(path_to_save, before_after)

        plt.show()

    def save_all_the_save_later(self, save_dir):

        np.save(save_dir + '/images_before.npy', self.images_before)

        for t in self.final_before_after_images_to_save_later:
            iter, before_after = t[0], t[1]
            path_to_save = save_dir + '/before_after.png'
            if iter is not None:
                path_to_save = save_dir + '/before_after_iter{}.png'.format(iter)
            plt.imsave(path_to_save, before_after)


def prepare_data_mnist(portion=1.0, batch_size=BATCH_SIZE, return_labels=False):

    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if portion < 1.0:
        indx = np.random.choice(x_train.shape[0], int(portion*x_train.shape[0]), replace=False)
        test_indx = np.random.choice(x_test.shape[0], int(portion*x_test.shape[0]), replace=False)
        x_train, y_train, x_test, y_test =x_train[indx],y_train[indx],x_test[test_indx],y_test[test_indx]

    x_train, x_test = x_train / 255.0, x_test / 255.0
    # Add a channels dimension
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    # Use tf.data to batch and shuffle the dataset:
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(batch_size, drop_remainder=True)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size, drop_remainder=True)
    if return_labels:
        return train_ds, test_ds, (x_train, y_train), (x_test, y_test)
    return train_ds, test_ds


def load_pre_trained_model(model_class, saved_models_dir, train_ds):
    saved_models = [x for x in os.listdir(saved_models_dir) if os.path.isdir(saved_models_dir + '/' + x)]
    if len(saved_models):
        saved_models.sort(key=lambda p: int(p.split('iter')[-1].split('_')[0])) # based on the model titles format of: '/model_at_epoch{}_iter{}_date{}' we want to sort by iteration
        pre_trained_model_path = saved_models[-1]
        full_pre_trained_model_path = saved_models_dir + '/' + saved_models[-1] + '/model_weights'
        print("full_pre_trained_model_path = ", full_pre_trained_model_path)

        model = model_class()
        model.compile(loss=model.mse_loss, optimizer=model.optimizer)
        # run a single step in order to initialize the model (necessary before loading weights)
        for images, _ in train_ds:
            model.train_step(images)
            break

        model.load_weights(full_pre_trained_model_path)
        pre_trained_epochs = int(pre_trained_model_path.split('epoch')[-1].split('_')[0])
        iter_counter = int(pre_trained_model_path.split('iter')[-1].split('_')[0])
        return model, pre_trained_model_path, pre_trained_epochs, iter_counter
    return model_class(), None, 0, 0


def main_mnist(epochs=3, mnist_portion=1, use_pretrained=1, do_plot=1):
    EPOCHS = epochs
    # train_ds, test_ds = prepare_data_mnist(portion=1)
    train_ds, test_ds, train, test = prepare_data_mnist(portion=mnist_portion, return_labels=True)
    # images_original, images_labels = tf.convert_to_tensor(train[0][:32]), tf.convert_to_tensor(train[1][:32])
    model = basic_AE()
    model.compile(loss=model.mse_loss, optimizer=model.optimizer)
    tb_dir = utils.check_dir_exists('logs/gradient_tape/' + str(model))
    saved_models_dir = utils.check_dir_exists('models/' + str(model))
    logger = ae_logger(tb_dir, model.mse_loss)
    epoch_counter, iter_counter = 0, 0

    # load pre-trained models
    pre_trained_epochs, pre_trained_model_path = 0, None  # number of epochs this model already trained for
    if use_pretrained:
        saved_models_dir = utils.check_dir_exists('models/' + str(model))
        model, pre_trained_model_path, pre_trained_epochs, iter_counter = load_pre_trained_model(basic_AE,
                                                                                                 saved_models_dir,
                                                                                                 train_ds)
    if pre_trained_model_path is None:
        print('did not use any pre-trained model.')
    else:
        print('loaded model: ' + saved_models_dir + '/' + pre_trained_model_path)
        logger = ae_logger(tb_dir, model.mse_loss)

    # train epochs
    for epoch in range(EPOCHS):
        iter_counter = model.train_epoch(train_ds, test_ds, logger, iter_counter_start=iter_counter,
                                         epoch_counter=epoch_counter)
        epoch_counter += 1

    # save model
    model_title = saved_models_dir + '/model_at_epoch_{}_iter_{}'.format(pre_trained_epochs + EPOCHS, iter_counter)
    model.save(model_title)

    # plots
    if do_plot:
        plotter = ae_plotter()
        plotter.plot_loss(model, logger)
        n_images = 10
        indx = np.random.choice(train[0].shape[0], n_images, replace=False)
        images_original, images_labels = train[0][indx], train[1][indx]
        encodings, noisy_encodings, images_decoded = model.call_on_images(images_original, images_labels)
        plotter.plot_before_after(images_original, images_decoded)
        plotter.plot_encodings_2Dlatent(model, encodings=encodings, labels=np.array(images_labels))

    return model








class celebA_AE(basic_AE):

    def __init__(self, save_encodings=True, learning_rate=LEARNING_RATE):
        super(celebA_AE, self).__init__(save_encodings, learning_rate)

    def __str__(self):
        return 'celebA_AE_z{}_la{}'.format(self.latent_, self.lambda_)

    def _create_encoder(self) -> list:  # 128*128*3
        layers = []
        layers.append(InputLayer((128, 128, 3)))
        layers.append(Conv2D(8, 7, activation='relu', padding='same'))
        layers.append(MaxPooling2D())   # 64*64*8
        layers.append(Conv2D(16, 5, activation='relu', padding='same'))
        layers.append(MaxPooling2D())   # 32*32*16
        layers.append(Conv2D(32, 3, activation='relu', padding='same'))
        layers.append(MaxPooling2D())  # 16*16*32
        layers.append(Conv2D(32, 3, activation='relu', padding='same'))
        layers.append(MaxPooling2D())  # 8*8*32
        layers.append(Conv2D(32, 3, activation='relu', padding='same'))
        layers.append(MaxPooling2D())  # 4*4*32 = 512
        layers.append(Flatten())
        layers.append(Dense(100, activation='relu'))
        layers.append(Dense(self.latent_, activation='linear',
                            activity_regularizer=tf.keras.regularizers.l2(self.lambda_)))
        return layers

    def _create_decoder(self) -> list:
        layers = []
        # layers.append(InputLayer((128,128,3))) # todo
        # layers.append(InputLayer((self.latent_,1)))
        layers.append(Dense(100, activation='relu'))
        layers.append(Dense(512, activation='relu'))
        layers.append(Reshape((4,4,32)))
        layers.append(UpSampling2D())   # 8*8*32
        layers.append(Conv2D(32, 3, activation='relu', padding='same'))
        layers.append(UpSampling2D()) # 16*16*32
        layers.append(Conv2D(32, 3, activation='relu', padding='same'))
        layers.append(UpSampling2D())   #32*32*32
        layers.append(Conv2D(16, 3, activation='relu', padding='same'))
        layers.append(UpSampling2D())   #64*64*16
        layers.append(Conv2D(8, 5, activation='relu', padding='same')) # 64*64*8
        layers.append(UpSampling2D())   #128*128*8
        layers.append(Conv2D(3, 7, activation='relu', padding='same'))  #128*128*3
        return layers


class celebA_AE_BN(basic_AE):

    def __init__(self,  save_encodings=True, learning_rate=LEARNING_RATE):
        super(celebA_AE_BN, self).__init__(save_encodings, learning_rate)

    def __str__(self):
        return 'celebA_AE_BN_z{}_la{}'.format(self.latent_, self.lambda_)

    def _create_encoder(self) -> list:  # 128*128*3
        layers = []
        layers.append(InputLayer((128, 128, 3)))
        layers.append(Conv2D(32, 5, activation='relu', padding='same'))
        layers.append(MaxPooling2D()) # 64,64,32
        layers.append(BatchNormalization())
        layers.append(Conv2D(64, 5, activation='relu', padding='same'))
        layers.append(MaxPooling2D())   # 32,32,64
        layers.append(BatchNormalization())
        layers.append(Conv2D(128, 5, activation='relu', padding='same'))
        layers.append(MaxPooling2D())  # 16,16,128
        layers.append(BatchNormalization())
        layers.append(Conv2D(256, 5, activation='relu', padding='same'))
        layers.append(MaxPooling2D())  # 8,8,256
        layers.append(BatchNormalization())
        layers.append(Conv2D(256, 5, activation='relu', padding='same'))
        layers.append(MaxPooling2D())  # 4,4,256 = 4096
        layers.append(BatchNormalization())
        layers.append(Flatten())
        layers.append(Dense(self.latent_, activation='linear',
                            activity_regularizer=tf.keras.regularizers.l2(self.lambda_)))
        return layers

    def _create_decoder(self) -> list:
        layers = []
        # layers.append(InputLayer((128,128,3))) # todo
        # layers.append(InputLayer((self.latent_,1)))
        layers.append(Dense(4096, activation='relu'))
        layers.append(Reshape((4,4,256)))
        layers.append(UpSampling2D())   # 8*8*256
        layers.append(Conv2D(256, 5, activation='relu', padding='same'))
        layers.append(BatchNormalization())
        layers.append(UpSampling2D()) # 16*16*256
        layers.append(Conv2D(128, 5, activation='relu', padding='same'))
        layers.append(BatchNormalization())
        layers.append(UpSampling2D())   #32*32*128
        layers.append(Conv2D(64, 5, activation='relu', padding='same'))
        layers.append(BatchNormalization())
        layers.append(UpSampling2D())   #64*64*16
        layers.append(Conv2D(32, 5, activation='relu', padding='same')) # 64*64*32
        layers.append(BatchNormalization())
        layers.append(UpSampling2D())   #128*128*32
        layers.append(Conv2D(3, 5, activation='relu', padding='same'))  #128*128*3
        return layers


def prepare_data_celebA(portion=1.0, batch_size=BATCH_SIZE, train_portion=0.9):
    all_images = np.load('data/full128_10k.npy').astype(np.float32) / 255.0
    indx = np.random.choice(all_images.shape[0], int(portion * all_images.shape[0]), replace=False)
    train_indx = indx[:int(train_portion * indx.shape[0])]
    test_indx = indx[int(train_portion * indx.shape[0]):]
    # Use tf.data to batch and shuffle the dataset:
    test_ds = tf.data.Dataset.from_tensor_slices((all_images[test_indx], np.zeros(test_indx.shape[0]))).batch(batch_size, drop_remainder=True)
    train_ds = tf.data.Dataset.from_tensor_slices((all_images[train_indx], np.zeros(train_indx.shape[0]))).shuffle(10000).batch(batch_size, drop_remainder=True)
    return train_ds, test_ds


def main_AE(model_class, train_ds, test_ds, use_pretrained=0, do_training=1, do_save=1, do_save_encodings=0, do_plot=0):

    model, model_dir = model_class(), None
    model.compile(loss=model.mse_loss, optimizer=model.optimizer)
    tb_dir = utils.check_dir_exists('logs/gradient_tape/' + str(model))
    saved_models_dir = utils.check_dir_exists('models/' + str(model))
    logger = ae_logger(tb_dir, model.mse_loss)
    plotter = ae_plotter()
    epoch_counter, iter_counter = 0, 0

    # load pre-trained models
    pre_trained_epochs, pre_trained_model_path = 0, None  # number of epochs this model already trained for
    if use_pretrained:
        saved_models_dir = utils.check_dir_exists('models/' + str(model))
        model, pre_trained_model_path, pre_trained_epochs, iter_counter = load_pre_trained_model(model_class, saved_models_dir, train_ds)
    if pre_trained_model_path is None:
        print('(did not use any pre-trained model)')
    else:
        model_dir = saved_models_dir + '/' + pre_trained_model_path
        print('loaded model: ' + model_dir)
        logger = ae_logger(tb_dir, model.mse_loss)
        plotter.images_before = np.load(model_dir + '/' + 'images_before.npy')

    if do_training:

        # train epochs
        for epoch in range(EPOCHS):
            iter_counter = model.train_epoch(train_ds, test_ds, logger, iter_counter_start=iter_counter, epoch_counter=epoch_counter)
            epoch_counter += 1
            if do_plot:
                plotter.plot_before_after(model, save_later=1, iter=iter_counter)

        # save model
        model_dir = saved_models_dir + '/model_at_epoch{}_iter{}_date{}'.format(pre_trained_epochs + EPOCHS, iter_counter, datetime.datetime.now().strftime('%d%m%Y-%H%M'))
        if do_save:
            model.save(model_dir)
        if do_save_encodings:
            model.save_encodings_npy(model_dir)

        # plots
        if do_plot:
            plotter.plot_loss(model, logger, do_save=do_save, save_dir=model_dir, iter=iter_counter)
            plotter.save_all_the_save_later(save_dir=model_dir)

    return model, model_dir







def mosaic(images: list, reshape: tuple=None, gap: int=1,
           normalize: bool=True, clip: bool=False, cols: int=-1):
    """
    :param images:
    :param reshape:
    :param gap:
    :param normalize:
    :param clip:
    :param cols:
    :return:
    """
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
        ret = np.concatenate([ret, max_val * np.ones(sh)], axis=0)
        sh = (ret.shape[0], gap) if ims[0].ndim < 3 else (ret.shape[0], gap, 3)
        ret = np.concatenate([ret, max_val * np.ones(sh)], axis=1)

    return ret





################# main #################

EPOCHS = 1
LAMBDA = 0.00001
LATENT_SIZE = 50

# # run on MNIST
# train_ds, test_ds, train, test = prepare_data_mnist(portion=0.05, return_labels=True)
# model = main_AE(basic_AE, train_ds, test_ds, epochs=1, use_pretrained=0, do_plot=0)
#
# n_images = 10
# indx = np.random.choice(train[0].shape[0], n_images, replace=False)
# images_original = train[0][indx]
# encodings, noisy_encodings, images_decoded = model.call_on_images(images_original)


# # Mosaic for celebA
# all_images = np.load('data/full128_10k.npy').astype(np.float32) / 255.0
# indx = np.random.choice(all_images.shape[0], 100, replace=False)
# celeb_mosaic = mosaic(list(all_images[indx]),normalize=False, cols=10)
# plt.imshow(celeb_mosaic)
# plt.show()

# run on celebA
train_ds, test_ds = prepare_data_celebA(portion=1)
# model, model_dir = main_AE(celebA_AE_BN, train_ds, test_ds, use_pretrained=0, do_save=1, do_save_encodings=0, do_plot=1)

# model, model_dir = main_AE(celebA_AE, train_ds, test_ds, use_pretrained=1, do_training=0)
# model.run_noisy_and_create_new_ds(test_ds, model_dir)

latent_sizes = [2,20,50,100]
lambdas = [0.0001, 0.00001, 0.000001]
for z in latent_sizes:
    for l in lambdas:
        LATENT_SIZE, LAMBDA = z, l
        print('\n'*5, "*** LATENT_SIZE={}  LAMBDA={} ***".format(LATENT_SIZE, LAMBDA))
        model, model_dir = main_AE(celebA_AE_BN, train_ds, test_ds, use_pretrained=0, do_save=1, do_save_encodings=0, do_plot=1)

# # create new ds for Hadar:
# all_images = np.load('data/full128_10k.npy').astype(np.float32) / 255.0
# ds = tf.data.Dataset.from_tensor_slices((all_images[:1000], np.zeros(1000))).batch(BATCH_SIZE, drop_remainder=False)
# model, model_dir = main_AE(celebA_AE, ds, ds, use_pretrained=1, do_training=0)
# model.run_noisy_and_create_new_ds(ds, model_dir)
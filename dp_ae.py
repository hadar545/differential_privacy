import numpy as np
import matplotlib.pyplot as plt
# from __future__ import absolute_import, division, print_function, unicode_literals
import os
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Reshape, UpSampling2D
from tensorflow.keras import Model
import datetime

import utils


EPOCHS = 1
BATCH_SIZE = 32
LEARNING_RATE = 0.001
LATENT_SIZE = 2


class basic_AE(Model):

    def __init__(self, lambda_: float=.001, latent_: int=LATENT_SIZE,  sd_:float=.1, save_encodings=True, learning_rate=LEARNING_RATE):
        super(basic_AE, self).__init__()

        self.latent_ = latent_
        self.sd_ = sd_
        self.save_encodings = save_encodings

        self.encodings = []
        self.noisy_encodings = []

        self.lambda_ = lambda_
        self.encoding_layers = self._create_encoder()
        self.decoding_layers = self._create_decoder()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')

    def __str__(self):
        return 'basic_AE_z{}'.format(self.latent_)

    def _create_decoder(self) -> list:
        layers = []
        layers.append(Dense(196, activation='relu'))
        layers.append(Reshape((7, 7, 4)))
        layers.append(Conv2D(8, 3, activation='relu', padding='same'))
        layers.append(Conv2D(16, 3, activation='relu', padding='same'))
        layers.append(UpSampling2D())
        layers.append(Conv2D(8, 3, activation='relu', padding='same'))
        layers.append(UpSampling2D())
        layers.append(Conv2D(1, 3, activation='relu', padding='same'))
        return layers

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

    def call_noisy(self, x):
        x = self.encode(x)
        z = x.copy()
        if self.save_encodings: self.encodings.append(z)
        z += self.sd_*np.random.randn(self.latent_)
        if self.save_encodings: self.noisy_encodings.append(z)
        x = self.decode(z)
        return x

    def call(self, x):
        x = self.encode(x)
        if self.save_encodings:
            self.encodings.append(x)
            self.noisy_encodings.append(x + self.sd_*np.random.randn(self.latent_))
        x = self.decode(x)
        return x

    def encode(self, x):
        for l in self.encoding_layers:
            x = l(x)
        return x

    def decode(self, x):
        for l in self.decoding_layers:
            x = l(x)
        return x

    def mse_loss(self, images):
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
        np.save(save_dir + '/encodings', np.array(self.encodings))
        np.save(save_dir + '/noisy_encodings', np.array(self.noisy_encodings))
        print('saved encodings at ' + save_dir)

    def save(self, model_title):
        model.save_weights(model_title, save_format='tf')
        print('saved model: ' + model_title)

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

            iter_counter += 1

        template = 'finished epoch {}, Step {}, Loss: {}, Test Loss: {}'
        print(template.format(epoch_counter + 1, iter_counter,
                              self.train_loss.result(),
                              self.test_loss.result()))

        # Reset the metrics for the next epoch
        self.reset_metrics()

        return iter_counter


class ae_logger():

    def __init__(self, tb_dir, loss_func):

        self.time_of_creation = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tb_dir = utils.check_dir_exists(tb_dir)
        print("tensorboard logdir: {}".format(self.tb_dir + '/' + self.time_of_creation))

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
        with self.test_summary_writer.as_default():
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
        pass

    def plot_loss(self, model, logger):
        plt.figure()
        plt.title(str(model) + ' - Loss per step\n' + logger.time_of_creation)
        plt.plot(logger.iter_log, logger.train_loss_log, 'b', label='train')
        plt.plot(logger.iter_log, logger.test_loss_log, 'r', label='test')
        plt.legend()
        plt.xlabel('iteration')
        plt.ylabel('loss')
        plt.show()

    def plot_encodings_2Dlatent(self, model):

        encodings, noisy_encodings = np.array(model.encodings), np.array(model.noisy_encodings)

        plt.figure()
        sd_title = 'ax0_std=%.2f ax1_std=%.2f' % (np.std(encodings[:,0]), np.std(encodings[:,1]))
        plt.title(str(model) + ' - Encodings\n' + sd_title)
        plt.scatter(encodings[:, 0], encodings[:,1], s=10)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

        plt.figure()
        sd_title = 'ax0_std=%.2f ax1_std=%.2f' % (np.std(noisy_encodings[:, 0]), np.std(noisy_encodings[:, 1]))
        plt.title(str(model) + ' - Noisy Encodings\n' + sd_title)
        plt.scatter(noisy_encodings[:, 0], noisy_encodings[:,1], s=10)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()


def prepare_data_mnist(portion=1.0, batch_size=BATCH_SIZE):

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

    return train_ds, test_ds


def load_pre_trained_model(model_class, saved_models_dir, train_ds):
    saved_models = [x.split('.')[0] for x in os.listdir(saved_models_dir) if x.endswith('.index')]
    if len(saved_models):
        saved_models.sort()
        pre_trained_model_path = saved_models[-1]
        full_pre_trained_model_path = saved_models_dir + '/' + pre_trained_model_path
        print("full_pre_trained_model_path = ", full_pre_trained_model_path)

        model = model_class()
        model.compile(loss=model.mse_loss, optimizer=model.optimizer)
        # run a single step in order to initialize the model (necessary before loading weights)
        for images, _ in train_ds:
            model.train_step(images)
            break

        model.load_weights(full_pre_trained_model_path)
        pre_trained_epochs = int(pre_trained_model_path.split('_')[-3])  # -3 because model_title later on
        iter_counter = int(pre_trained_model_path.split('_')[-1])  # -1 because model_title later on
        return model, pre_trained_model_path, pre_trained_epochs, iter_counter
    return model_class(), None, 0, 0


if __name__ == '__main__':

    train_ds, test_ds = prepare_data_mnist(portion=0.1)
    model = basic_AE()
    tb_dir = utils.check_dir_exists('logs/gradient_tape/' + str(model))
    saved_models_dir = utils.check_dir_exists('models/' + str(model))
    logger = ae_logger(tb_dir, model.mse_loss)
    epoch_counter, iter_counter = 0, 0

    # load pre-trained models
    use_pretrained, pre_trained_model_path = 1, None
    pre_trained_epochs = 0  # number of epochs this model already trained for
    if use_pretrained:
        saved_models_dir = utils.check_dir_exists('models/' + str(model))
        model, pre_trained_model_path, pre_trained_epochs, iter_counter = load_pre_trained_model(basic_AE, saved_models_dir, train_ds)
    if pre_trained_model_path is None:
        print('(did not use any pre-trained model)')
    else:
        print('loaded model: ' + saved_models_dir + '/' + pre_trained_model_path)

    # train epochs
    for epoch in range(EPOCHS):
        iter_counter = model.train_epoch(train_ds, test_ds, logger, iter_counter_start=iter_counter, epoch_counter=epoch_counter)
        epoch_counter += 1

    # save model
    model_title = saved_models_dir + '/model_at_epoch_{}_iter_{}'.format(pre_trained_epochs + EPOCHS, iter_counter)
    model.save(model_title)

    # plots
    plotter = ae_plotter()
    plotter.plot_loss(model, logger)
    plotter.plot_encodings_2Dlatent(model)


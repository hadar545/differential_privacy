import numpy as np
import matplotlib.pyplot as plt
# from __future__ import absolute_import, division, print_function, unicode_literals
import os
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Reshape, UpSampling2D, InputLayer
from tensorflow.keras import Model
import datetime

import utils


EPOCHS = 0
BATCH_SIZE = 32
LEARNING_RATE = 0.001
LATENT_SIZE = 3


class basic_AE(Model):

    def __init__(self, lambda_: float=.001, latent_: int=LATENT_SIZE,  sd_: float=.1, save_encodings=True,
                 learning_rate=LEARNING_RATE):
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
        z = self.encode(x)
        z_noised = z + self.sd_*np.random.randn(self.latent_)
        x_decoded = self.decode(z_noised)
        return z, z_noised, x_decoded

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
                print('Step {}, Loss: {}, Test Loss: {}'.format(iter_counter, self.train_loss.result(), self.test_loss.result()))

            iter_counter += 1

        template = 'finished epoch {}, Step {}, Loss: {}, Test Loss: {}'
        print(template.format(epoch_counter + 1, iter_counter,
                              self.train_loss.result(),
                              self.test_loss.result()))

        # Reset the metrics for the next epoch
        self.reset_metrics()

        return iter_counter

    def call_on_images(self, imgs, labels=None):

        encodings, noisy_encodings, decoded_images = [], [], np.zeros_like(imgs)
        i = 0
        def run(images):
            z, z_noised, x_decoded = self.call_noisy(images)
            encodings.append(z)
            noisy_encodings.append(z + self.sd_ * np.random.randn(self.latent_))
            decoded_images[i] = x_decoded
            i += 1

        if labels is None:
            ds = tf.data.Dataset.from_tensor_slices(imgs).batch(1, drop_remainder=True)
            for images in ds:
                run(images)
        else:
            ds = tf.data.Dataset.from_tensor_slices((imgs, labels)).batch(1, drop_remainder=True)
            for images, labels in ds:
                run(images)

        return np.array(encodings).reshape(imgs.shape[0],self.latent_), \
               np.array(noisy_encodings).reshape(imgs.shape[0],self.latent_), \
               decoded_images






class ae_logger():

    def __init__(self, tb_dir, loss_func):

        self.time_of_creation = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tb_dir = utils.check_dir_exists(tb_dir)
        print("tensorboard logdir: {}".format(self.tb_dir + '/' + self.time_of_creation))
        print("Reminder, to open tensorboard - open the link returned from the "
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

    def plot_before_after(self, images_original, images_decoded):
        originals_wide = np.hstack(tuple([images_original[i] for i in range(images_original.shape[0])]))
        decoded_wide = np.hstack(tuple([images_decoded[i] for i in range(images_decoded.shape[0])]))
        plt.imshow(np.vstack((originals_wide, decoded_wide))[:,:,0])
        plt.axis('off')
        plt.show()

        # all_images_stacked = np.concatenate((images_original[:, :, :, 0], images_decoded[:, :, :, 0]))
        # # grid in matplotlib, taken from https://stackoverflow.com/questions/46615554/how-to-display-multiple-images-in-one-figure-correctly/46616645
        # fig, ax = plt.subplots(nrows=2, ncols=images_decoded.shape[0])
        # for i, axi in enumerate(ax.flat):
        #     img = all_images_stacked[i, :, :]
        #     axi.imshow(img)
        # plt.tight_layout(True)
        # plt.show()




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
    saved_models = [x.split('.')[0] for x in os.listdir(saved_models_dir) if x.endswith('.index')]
    if len(saved_models):
        saved_models.sort(key=lambda p: int(p.split('_')[-1]))
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
    EPOCHS = 7
    # train_ds, test_ds = prepare_data_mnist(portion=1)
    train_ds, test_ds, train, test = prepare_data_mnist(portion=0.1, return_labels=True)
    images_original, images_labels = tf.convert_to_tensor(train[0][:32]), tf.convert_to_tensor(train[1][:32])
    model = basic_AE()
    model.compile(loss=model.mse_loss, optimizer=model.optimizer)
    tb_dir = utils.check_dir_exists('logs/gradient_tape/' + str(model))
    saved_models_dir = utils.check_dir_exists('models/' + str(model))
    logger = ae_logger(tb_dir, model.mse_loss)
    epoch_counter, iter_counter = 0, 0

    # load pre-trained models
    use_pretrained = 1
    pre_trained_epochs, pre_trained_model_path = 0, None  # number of epochs this model already trained for
    if use_pretrained:
        saved_models_dir = utils.check_dir_exists('models/' + str(model))
        model, pre_trained_model_path, pre_trained_epochs, iter_counter = load_pre_trained_model(basic_AE, saved_models_dir, train_ds)
    if pre_trained_model_path is None:
        print('(did not use any pre-trained model)')
    else:
        print('loaded model: ' + saved_models_dir + '/' + pre_trained_model_path)
        logger = ae_logger(tb_dir, model.mse_loss)

    # train epochs
    for epoch in range(EPOCHS):
        iter_counter = model.train_epoch(train_ds, test_ds, logger, iter_counter_start=iter_counter, epoch_counter=epoch_counter)
        epoch_counter += 1

    # save model
    model_title = saved_models_dir + '/model_at_epoch_{}_iter_{}'.format(pre_trained_epochs + EPOCHS, iter_counter)
    model.save(model_title)

    # plots
    if 0:
        plotter = ae_plotter()
        plotter.plot_loss(model, logger)
        n_images = 10
        indx = np.random.choice(train[0].shape[0], n_images, replace=False)
        images_original, images_labels = train[0][indx], train[1][indx]
        encodings, noisy_encodings, images_decoded = model.call_on_images(images_original, images_labels)
        plotter.plot_before_after(images_original, images_decoded)
        plotter.plot_encodings_2Dlatent(model, encodings=encodings, labels=np.array(images_labels))



    # for images, labels in train_ds:
    #     plotter.plot_encodings_2Dlatent(model, images=images, labels=labels)
    #     break

# finished epoch 14, Step 2618, Loss: 0.04205966740846634, Test Loss: 0.04324441775679588
# saved model: models/basic_AE_z3/model_at_epoch_14_iter_2618

# finished epoch 7, Step 1309, Loss: 0.04657045751810074, Test Loss: 0.04678208380937576
# saved model: models/basic_AE_z3/model_at_epoch_7_iter_1309

# finished epoch 7, Step 2618, Loss: 0.04348893463611603, Test Loss: 0.04404647275805473
# saved model: models/basic_AE_z3/model_at_epoch_14_iter_2618
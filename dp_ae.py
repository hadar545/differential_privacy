import numpy as np
import matplotlib.pyplot as plt
# from __future__ import absolute_import, division, print_function, unicode_literals
import os
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Reshape, UpSampling2D
from tensorflow.keras import Model
import datetime

import utils

############# basic hyperparameters #############

EPOCHS = 2
BATCH_SIZE = 32
LEARNING_RATE = 0.001

############# data #############

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
# Add a channels dimension
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

# for an AE, the y is just the original images...
y_train = x_train.copy()
y_test = x_test.copy()

BATCH_SIZE = 32

# Use tf.data to batch and shuffle the dataset:
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(BATCH_SIZE, drop_remainder=True)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(BATCH_SIZE, drop_remainder=True)


############# models #############

class basic_AE(Model):

    def __init__(self, lambda_: float=.001):
        super(basic_AE, self).__init__()
        self.lambda_ = lambda_
        self.encoding_layers = self._create_encoder()
        self.decoding_layers = self._create_decoder()

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
        layers.append(Dense(50, activation='relu',
                            activity_regularizer=tf.keras.regularizers.l2(self.lambda_)))
        return layers

    def call(self, x):
        x = self.encode(x)
        # z = x.copy()
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

    def __str__(self):
        return 'basic_AE'


model_of_choice = basic_AE

# Create an instance of the model
model = model_of_choice()

# Choose an optimizer and loss function for training:
# loss_object = tf.keras.losses.mean_squared_error()
def loss(images, reconstructions=None):
    """ calc the reconstruction error """
    if reconstructions is None:
        reconstructions = model(images)
    return tf.reduce_mean(tf.square(tf.subtract(reconstructions, images)))
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

# metrics
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.mean_squared_error()
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.mean_squared_error()
def reset_metrics():
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()


# training
@tf.function
def train_step(images, reconstructions=None):
    if reconstructions is None:
        reconstructions = model(images)
    with tf.GradientTape() as tape:
        gradients = tape.gradient(loss(images, reconstructions), model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss(loss)
        train_accuracy(images, reconstructions)


# testing
@tf.function
def test_step(images, reconstructions=None):
    if reconstructions is None:
        reconstructions = model(images)
    t_loss = loss(images)
    test_loss(t_loss)
    test_accuracy(images, reconstructions)


############# logs #############

# local logs:
iter_counter = 0
iter_log = []
train_accuracy_log = []
test_accuracy_log = []

# tensorboard logs - Set up summary writers to write the summaries to disk
# (reminder to myself) to open tensorboard - open the link returned from the terminal command: %tensorboard --logdir logs/gradient_tape
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tb_dir = utils.check_dir_exists('logs/gradient_tape/' + str(model))
print("tensorboard logdir: {}".format(tb_dir + '/' + current_time))
train_log_dir = tb_dir + '/' + current_time + '/train'
test_log_dir = tb_dir + '/' + current_time + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)


def log(iter, images, reconstructions, test_images, test_reconstructions):

    iter_log.append(iter)
    train_accuracy_log.append(train_accuracy.result())
    test_accuracy_log.append(test_accuracy.result())

    with train_summary_writer.as_default():
        tf.summary.scalar('loss', train_loss.result(), step=iter)
        tf.summary.scalar('accuracy', train_accuracy.result(), step=iter)
        tf.summary.image('original', images, max_outputs=10, step=iter)
        tf.summary.image('reconstructed', reconstructions, max_outputs=10, step=iter)
    with test_summary_writer.as_default():
        tf.summary.scalar('loss', test_loss.result(), step=iter)
        tf.summary.scalar('accuracy', test_accuracy.result(), step=iter)
        tf.summary.image('original', test_images, max_outputs=10, step=iter)
        tf.summary.image('reconstructed', test_reconstructions, max_outputs=10, step=iter)


############# training #############

EPOCHS = 2

# load pre trained models
pre_trained_epochs = 0  # number of epochs this model already trained for
use_pretrained = 0
pretrained_model_used = None
saved_models_dir = utils.check_dir_exists('models/' + str(model))
# if use_pretrained:
#     saved_models = [x.split('.')[0] for x in os.listdir(saved_models_dir) if x.endswith('.index')]
#     if len(saved_models):
#         saved_models.sort()
#         pre_trained_model_path = saved_models[-1]
#
#         model = model_of_choice()
#         model.compile(loss=loss_object, optimizer=optimizer)
#         # run a single step in order to initialize the model (necessary before loading weights)
#         # train_step(x_train[:1], y_train[:1])
#         for images, labels in train_ds:
#             train_step(images, labels)
#             break
#
#         model.load_weights(saved_models_dir + '/' + pre_trained_model_path)
#         pre_trained_epochs = int(pre_trained_model_path.split('_')[-3])  # -3 because model_title later on
#         iter_counter = int(pre_trained_model_path.split('_')[-1])  # -1 because model_title later on
#         pretrained_model_used = pre_trained_model_path
if pretrained_model_used is None:
    print('(did not use any pre-trained model)')
else:
    print('loaded model: ' + saved_models_dir + '/' + pre_trained_model_path)

# training, yalla balagan
for epoch in range(pre_trained_epochs, pre_trained_epochs + EPOCHS):
    print('started epoch %d/%d' % (epoch + 1, pre_trained_epochs + EPOCHS))
    for images, _ in train_ds:

        reconstructions = model(images)
        train_step(images, reconstructions)

        if iter_counter % 100 == 0:

            for test_images, _ in test_ds:
                test_reconstructions = model(test_images)
                test_step(test_images, test_reconstructions)

            log(iter_counter, images, reconstructions, test_images, test_reconstructions)

        iter_counter += 1

    template = 'finished epoch {}, Step {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch + 1, iter_counter,
                          train_loss.result(),
                          train_accuracy.result(),
                          test_loss.result(),
                          test_accuracy.result()))

    # Reset the metrics for the next epoch
    reset_metrics()

# save model:
model_title = saved_models_dir + '/model_at_epoch_{}_iter_{}'.format(pre_trained_epochs + EPOCHS, iter_counter)
# model.save(model_title + '.h5')
model.save_weights(model_title, save_format='tf')
print('saved model: ' + model_title)

############# plot #############


# plot acc:
plt.figure()
plt.title(str(model) + ' - Accuracy per step\n' + current_time)
plt.plot(iter_log, train_accuracy_log, 'b', label='train')
plt.plot(iter_log, test_accuracy_log, 'r', label='test')
plt.legend()
plt.xlabel('iteration')
plt.ylabel('accuracy')
plt.show()

print("\nEND")

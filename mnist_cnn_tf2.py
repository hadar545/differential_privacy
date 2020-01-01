import numpy as np
import matplotlib.pyplot as plt
# from __future__ import absolute_import, division, print_function, unicode_literals
import os
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
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
x_train, x_test = x_train/255.0, x_test/255.0
# Add a channels dimension
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

# Use tf.data to batch and shuffle the dataset:
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(BATCH_SIZE, drop_remainder=True)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(BATCH_SIZE, drop_remainder=True)


############# models #############

class basic_CNN(Model):

    def __init__(self):
        super(basic_CNN, self).__init__()
        self.conv1 = Conv2D(32, 5, activation='relu')
        self.pool1 = MaxPooling2D()
        self.conv2 = Conv2D(64, 5, activation='relu')
        self.pool2 = MaxPooling2D()
        self.flatten = Flatten()
        self.d1 = Dense(1024, activation='relu')
        # self.d1_ = Dense(254, activation='relu')
        self.d2 = Dense(10, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.d1(x)
        # x = self.d1_(x)
        x = self.d2(x)
        return x

    def __str__(self):
        return 'basic_CNN_for_mnist'


class shallow_CNN(Model):

    def __init__(self):
        super(shallow_CNN, self).__init__()
        self.conv1 = Conv2D(32, 5, activation='relu')
        self.pool1 = MaxPooling2D()
        # self.conv2 = Conv2D(64, 5, activation='relu')
        # self.pool2 = MaxPooling2D()
        self.flatten = Flatten()
        self.d1 = Dense(1024, activation='relu')
        # self.d1_ = Dense(254, activation='relu')
        self.d2 = Dense(10, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        # x = self.conv2(x)
        # x = self.pool2(x)
        x = self.flatten(x)
        x = self.d1(x)
        # x = self.d1_(x)
        x = self.d2(x)
        return x

    def __str__(self):
        return 'shallow_CNN_for_mnist'


model_of_choice = basic_CNN

# Create an instance of the model
model = model_of_choice()

# Choose an optimizer and loss function for training:
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

# metrics
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
def reset_metrics():
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

# training
@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    train_accuracy(labels, predictions)

# testing
@tf.function
def test_step(images, labels):
    predictions = model(images)
    t_loss = loss_object(labels, predictions)
    test_loss(t_loss)
    test_accuracy(labels, predictions)


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

def log(iter):

    iter_log.append(iter)
    train_accuracy_log.append(train_accuracy.result())
    test_accuracy_log.append(test_accuracy.result())

    with train_summary_writer.as_default():
        tf.summary.scalar('loss', train_loss.result(), step=iter_counter)
        tf.summary.scalar('accuracy', train_accuracy.result(), step=iter_counter)
    with test_summary_writer.as_default():
        tf.summary.scalar('loss', test_loss.result(), step=iter_counter)
        tf.summary.scalar('accuracy', test_accuracy.result(), step=iter_counter)


############# training #############

# load pre trained models
pre_trained_epochs = 0 # number of epochs this model already trained for
use_pretrained = 0
pretrained_model_used = None
saved_models_dir = utils.check_dir_exists('models/' + str(model))
if use_pretrained:
    saved_models = [x.split('.')[0] for x in os.listdir(saved_models_dir) if x.endswith('.index')]
    if len(saved_models):
        saved_models.sort()
        pre_trained_model_path = saved_models[-1]

        model = model_of_choice()
        model.compile(loss = loss_object, optimizer = optimizer)
        # run a single step in order to initialize the model (necessary before loading weights)
        # train_step(x_train[:1], y_train[:1])
        for images, labels in train_ds:
            train_step(images, labels)
            break

        model.load_weights(saved_models_dir + '/' + pre_trained_model_path)
        pre_trained_epochs = int(pre_trained_model_path.split('_')[-3]) # -3 because model_title later on
        iter_counter = int(pre_trained_model_path.split('_')[-1]) # -1 because model_title later on
        pretrained_model_used = pre_trained_model_path
if pretrained_model_used is None:
    print('(did not use any pre-trained model)')
else:
    print('loaded model: ' + saved_models_dir + '/' + pre_trained_model_path)

# training, yalla balagan
for epoch in range(pre_trained_epochs, pre_trained_epochs + EPOCHS):
    print('started epoch %d/%d' % (epoch+1, pre_trained_epochs + EPOCHS))
    for images, labels in train_ds:

        train_step(images, labels)

        if iter_counter % 100 == 0:

            for test_images, test_labels in test_ds:
                test_step(test_images, test_labels)

            log(iter_counter)

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
from datetime import datetime
import time
import tensorflow as tf
import os
import math
import numpy as np
from VGGnet_CNN_model import cnn


def distorted_inputs(data_dir, batch_size):
    filenames = [os.path.join(data_dir)]
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    # Create a queue that produces the filenames to read.
    filename_queue = tf.train.string_input_producer(filenames)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
      features={"label": tf.FixedLenFeature([], tf.int64),
          "image": tf.FixedLenFeature([], tf.string)})
    label = tf.cast(features["label"], tf.int32)
    imgin = tf.reshape(tf.decode_raw(features["image"], tf.uint8), tf.stack([96, 96, 3]))
    reshaped_image = tf.cast(imgin, tf.float32)

    height = 64
    width = 64

    distorted_image = tf.random_crop(reshaped_image, [height, width, 3])
    distorted_image = tf.image.random_flip_left_right(distorted_image)
    distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)
    distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)

    float_image = tf.image.per_image_standardization(distorted_image)
    float_image.set_shape([height, width, 3])

    data_size = 25300
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(data_size*min_fraction_of_examples_in_queue)
    print ('Filling queue with %d CIFAR images before starting to train. '
         'This will take a few minutes.' % min_queue_examples)

    num_preprocess_threads = 16
    images, label_batch = tf.train.shuffle_batch([float_image, label], batch_size=batch_size,
        num_threads=num_preprocess_threads,capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)

    tf.summary.image('images', images)
    return images, tf.reshape(label_batch, [batch_size])


# In[3]: # loss


def loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
    # The total loss is defined as the cross entropy loss plus all of the weight(L2 loss)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


# In[4]: # train using AdamOptimizer


def train(total_loss, global_step):

    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    for l in losses + [total_loss]:
        tf.summary.scalar(l.op.name+' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.AdamOptimizer() # instead of tf.train.AdamOptimizer()
        grads = opt.compute_gradients(total_loss)

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(0.9999, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op # Op for training.


# In[ ]: # Session


with tf.Graph().as_default():
    global_step = tf.contrib.framework.get_or_create_global_step()
    # for train
    train_image, train_label = distorted_inputs('/Users/Downloads/CNN, DCGANコード/cnn_train.tfrecords',128)
    c_logits = cnn(train_image)
    loss = loss(c_logits, train_label)
    train_op = train(loss, global_step)

    class _LoggerHook(tf.train.SessionRunHook):
        def begin(self):
            self._step = -1
            self._start_time = time.time()

        def before_run(self, run_context):
            self._step += 1
            return tf.train.SessionRunArgs(loss)  # Asks for loss value.

        def after_run(self, run_context, run_values):
            if self._step % 10 == 0:
                current_time = time.time()
                duration = current_time - self._start_time
                self._start_time = current_time

                loss_value = run_values.results
                examples_per_sec = 10 * 128 / duration
                sec_per_batch = float(duration / 10)

                format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                        'sec/batch)')
                print (format_str % (datetime.now(), self._step, loss_value,
                               examples_per_sec, sec_per_batch))
    # checkpoint_dir must be directory
    with tf.train.MonitoredTrainingSession(checkpoint_dir='/Users/Downloads/cnn_tbdir',
                                           hooks=[tf.train.StopAtStepHook(last_step=4000),
               tf.train.NanTensorHook(loss), _LoggerHook()], config=tf.ConfigProto(log_device_placement=False)) as mon_sess:
        while not mon_sess.should_stop():
            mon_sess.run(train_op)

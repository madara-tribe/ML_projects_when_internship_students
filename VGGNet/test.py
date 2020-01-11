from datetime import datetime
import time
import tensorflow as tf
import os
import math
import numpy as np
from VGGnet_CNN_model import cnn

def inputs(data_dir, batch_size):
    filenames = [os.path.join(data_dir)]
    num_examples_per_epoch = 540

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
    resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image, height, width)
    float_image = tf.image.per_image_standardization(resized_image)
    float_image.set_shape([height, width, 3])

    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(num_examples_per_epoch*min_fraction_of_examples_in_queue)

    num_preprocess_threads = 16
    images, label_batch = tf.train.batch([float_image, label], batch_size=batch_size,
        num_threads=num_preprocess_threads,capacity=min_queue_examples + 3 * batch_size)

    tf.summary.image('images', images)
    return images, tf.reshape(label_batch, [batch_size])


# In[2]:


def eval_once(saver, summary_writer, top_k_op, summary_op):
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state('/Users/Downloads/cnn_tbdir')
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print('No checkpoint file found')
            return

        # Start the queue runners.
        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))

            num_iter = int(math.ceil(540 / 128))
            true_count = 0  # Counts the number of correct predictions.
            total_sample_count = num_iter * 128
            step = 0
            while step < num_iter and not coord.should_stop():
                predictions = sess.run([top_k_op])
                true_count += np.sum(predictions)
                step += 1

                # Compute precision @ 1.
            precision = true_count / total_sample_count
            print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
            summary.value.add(tag='Precision @ 1', simple_value=precision)
            summary_writer.add_summary(summary, global_step)

        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)



# In[ ]: # Session


with tf.Graph().as_default() as g:
    # Get images and labels for CIFAR-10.
    images, labels = inputs('/Users/Downloads/CNN, DCGANコード/cnn_test.tfrecords',128)

    logits = cnn(images)

    # Calculate predictions.
    top_k_op = tf.nn.in_top_k(logits, labels, 1)

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(0.9999)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter('/Users/hagiharatatsuya/Downloads/eval_dir', g)
    while True:
        eval_once(saver,summary_writer, top_k_op, summary_op)
        if False:
            break
        time.sleep(60*5)# how often eval

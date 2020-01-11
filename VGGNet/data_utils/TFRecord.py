import tensorflow as tf
import numpy as np
import pickle
import os
import sys
import _pickle as cPickle

def unpickle(file):
    fp = open(file, 'rb')
    if sys.version_info.major == 2:
        data = pickle.load(fp)
    elif sys.version_info.major == 3:
        data = pickle.load(fp, encoding='latin-1')
    fp.close()

    return data

# for one-hot

def dense_to_one_hot(labels_dense, num_classes):
    for_cnn3=np.array(labels_dense)
    num_classes=10
    num_labels = for_cnn3.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + for_cnn3.ravel()-1] = 1
    return labels_one_hot


def load_image(image_name, label_name):
    train_image = unpickle(image_name)
    train_label = unpickle(label_name)
    lists = []
    for i, v in zip(tr_im, tr_la):
        listed.append([v,i])
    return lists


def save_to_tfrecord_file(inpus_list, SAVE_DIR):
    writer = tf.python_io.TFRecordWriter(os.path,join(SAVE_DIR, 'sample.tfrecords'))
    for label, img in inpus_list:
        record = tf.train.Example(features=tf.train.Features(feature={
              "label": tf.train.Feature(
                  int64_list=tf.train.Int64List(value=[label])),
              "image": tf.train.Feature(
                  bytes_list=tf.train.BytesList(value=[img.tostring()]))
          }))

        writer.write(record.SerializeToString())
    writer.close()

def save():
    images_name = "dataset_images.pickle"
    label_name = "dataset_labels.pickle"
    input_lists = load_image(image_name, label_name)
    SAVE_DIR = '/Users/Downloads'
    save_to_tfrecord_file(inpus_lists, SAVE_DIR)



def plot_tfrecord(tfrecord_file_name):
    file_name_queue = tf.train.string_input_producer([tfrecord_file_name])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(file_name_queue)
    features = tf.parse_single_example(
          serialized_example,
          features={
              "label": tf.FixedLenFeature([], tf.int64),
              "image": tf.FixedLenFeature([], tf.string)
          })
    label = tf.cast(features["label"], tf.int32)
    imgin = tf.reshape(tf.decode_raw(features["image"], tf.uint8),
                               tf.stack([32, 32, 3]))

    images, labels = tf.train.batch([imgin, label], batch_size=128, num_threads=2, capacity=1000 + 3 * 128)

    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            print(sess.run(images))
        finally:
            coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    save()

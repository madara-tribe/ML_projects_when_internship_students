import pickle
import numpy as np
import tensorflow as tf
import scipy.misc
import time
import tensorflow.contrib.slim as slim

def batch_norm(x, name, train=True):
    with tf.variable_scope(name):
        return tf.contrib.layers.batch_norm(x,
                      decay=0.9,
                      updates_collections=None,
                      epsilon=1e-5,
                      scale=True,
                      is_training=train,
                      scope=name)

def conv2d(input_, output_dim, name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [5, 5, input_.get_shape()[-1], output_dim], initializer=tf.truncated_normal_initializer(stddev=0.02))
        conv = tf.nn.conv2d(input_, w, strides=[1, 2, 2, 1], padding='SAME')
        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
    return tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

def lrelu(x, name="lrelu"):
    return tf.maximum(x, 0.2*x)

def linear(input_, output_size,scope=None, with_w=False):
    shape = input_.get_shape().as_list()
    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32, tf.random_normal_initializer(stddev=0.02))
        bias = tf.get_variable("bias", [output_size], initializer=tf.constant_initializer(0.0))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias

def deconv2d(input_, output_shape, name="deconv2d", with_w=False):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [5, 5, output_shape[-1], input_.get_shape()[-1]], initializer=tf.random_normal_initializer(stddev=0.02))
        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape, strides=[1, 2, 2, 1])
        b = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, b), deconv.get_shape())
        if with_w:
            return deconv, w, b
        else:
            return deconv

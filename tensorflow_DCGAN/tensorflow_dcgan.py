import numpy as np
import tensorflow as tf
import os
import tensorflow.contrib.slim as slim
from dcgan_functions import *
from help_func import *

# discriminator, generator and sampler
# TO DO : To reduce amount of calculation, you change g_fc in generator as well as sampler

def discriminator(image, reuse=False):
    batch_size=64
    d_fc = 64
    with tf.variable_scope("discriminator", reuse=reuse) as scope:
        if reuse:
            scope.reuse_variables()
        h0 = lrelu(conv2d(image, d_fc, name='d_h0_conv'))
        h1 = lrelu(batch_norm(conv2d(h0, d_fc*2, name='d_h1_conv'),'d_bn1'))
        h2 = lrelu(batch_norm(conv2d(h1, d_fc*4, name='d_h2_conv'),'d_bn2'))
        h3 = lrelu(batch_norm(conv2d(h2, d_fc*8, name='d_h3_conv'),'d_bn3'))  # shape=(batch_size, 64, 64, 3)　
        h4 = linear(tf.reshape(h3, [batch_size, -1]),1,'d_h4_lin')
        return tf.nn.sigmoid(h4), h4


    # shape=(batch_size, 64, 64, 3)
def generator(z_):
    batch_size=64
    g_fc = 64
    with tf.variable_scope("generator") as scope:
        z, h0_w, h0_b = linear(z_, g_fc*4*4*8, 'g_h0_lin',with_w=True)
        h0 = tf.nn.relu(batch_norm(tf.reshape(z, [-1, 4, 4, g_fc*8]), 'g_bn0'))
        h1, h1_w, h1_b = deconv2d(h0, [batch_size, 8, 8, g_fc*4], name='g_h1', with_w=True)
        h1 = tf.nn.relu(batch_norm(h1, 'g_bn1'))
        h2, h2_w, h2_b = deconv2d(h1, [batch_size, 16, 16, g_fc*2], name='g_h2', with_w=True)
        h2 = tf.nn.relu(batch_norm(h2, 'g_bn2'))
        h3, h3_w, h3_b = deconv2d(h2, [batch_size, 32, 32, g_fc*1], name='g_h3', with_w=True)
        h3 = tf.nn.relu(batch_norm(h3, 'g_bn3'))
        h4, h4_w, h4_b = deconv2d(h3, [batch_size, 64, 64, 3], name='g_h4', with_w=True)
        return tf.nn.tanh(h4)  # shape=(batch_size, 64, 64, 3)



def sampler(z_):# shape=(batch_size, 64, 64, 3)　
    batch_size=64
    g_fc = 64
    with tf.variable_scope("generator") as scope:
        scope.reuse_variables()
        z= linear(z_, g_fc*4*4*8,'g_h0_lin')
        h0 = tf.nn.relu(batch_norm(tf.reshape(z, [-1, 4, 4, g_fc*8]),'g_bn0',train=False))
        h1 = deconv2d(h0, [batch_size, 8, 8, g_fc*4], name='g_h1')
        h1 = tf.nn.relu(batch_norm(h1,'g_bn1',train=False))
        h2 = deconv2d(h1, [batch_size, 16, 16, g_fc*2], name='g_h2')
        h2 = tf.nn.relu(batch_norm(h2,'g_bn2',train=False))
        h3 = deconv2d(h2, [batch_size, 32, 32, g_fc*1], name='g_h3')
        h3 = tf.nn.relu(batch_norm(h3,'g_bn3',train=False))
        h4 = deconv2d(h3, [batch_size, 64, 64, 3], name='g_h4')
        return tf.nn.tanh(h4)  # shape=(batch_size, 64, 64, 3)

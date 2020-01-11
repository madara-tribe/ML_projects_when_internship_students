import sys
import pickle
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import scipy.misc
import time
import tensorflow.contrib.slim as slim
from dcgan_functions import *
from help_func import *


def saple_train():
    with tf.Session(config=run_config) as sess:
        sess.run(tf.global_variables_initializer())
        sample_images =X_image[0:64]

        counter=1
        epochs=50
        start_time=time.time()
        show_variables()

        for epoch in range(epochs):
            batch_idxs= min (len(X_image), np.inf) // 64
            for idx in range (0, batch_idxs):
                batch_images = X_image[idx*64:(idx+1)*64]

                sess.run(d_optim, feed_dict = {z: batch_z, image: batch_images})
                sess.run(g_optim, feed_dict = {z: batch_z})

                # Run g_optim twice to realize loss value
                sess.run(g_optim, feed_dict = {z: batch_z})
                errD_fake = d_loss_fake.eval({z: batch_z })
                errD_real = d_loss_real.eval({image: batch_images})
                errG = g_loss.eval({z: batch_z})
                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time:%4.4f, d_loss: %.8f, g_loss: %.8f" % (epoch, idx, batch_idxs,
                                                                                             time.time()-start_time, errD_fake+errD_real, errG))

                # show sample image while trainig
                if np.mod(counter, 30)==1:
                    samples, d_loss_sample, g_loss_sample = sess.run([sampler, d_loss, g_loss],
                                                   feed_dict={z: sample_z, image: sample_images})

                    print("[Sample] d_loss:%.8f, g_loss:%.8f" % (d_loss_sample, g_loss_sample))

                    save_images(samples, (8, 8), '/home/dcgan_dir/train_{:02d}_{:04d}.png'.format(epoch, idx))
                    vnari=show_images(samples, (8,8))
                    plt.imshow(vnari)
                    plt.show()
                # save sess to directory
                if np.mod(counter, 100)==1:
                    saver.save(sess, "/home/dcgan_dir/decgan")



# In[ ]: # Restoring　session from directory and move again
def dcgan_train(saver):
    saver=tf.train.Saver()
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth=True

    with tf.Session(config=run_config) as sess:
        ckpt = tf.train.get_checkpoint_state('/Users/Downloads/dcgan_dir')
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print('No checkpoint file found')
            return

        sample_images =X_image[0:64]
        counter=1
        epochs=50
        start_time=time.time()
        show_variables()

        for epoch in range(epochs):
            batch_idxs= min (len(X_image), np.inf) // 64
            for idx in range (0, batch_idxs):
                batch_images = X_image[idx*64:(idx+1)*64]

                sess.run(d_optim, feed_dict = {z: batch_z, image: batch_images})
                sess.run(g_optim, feed_dict = {z: batch_z})

                # Run g_optim twice to realize loss value
                sess.run(g_optim, feed_dict = {z: batch_z})
                errD_fake = d_loss_fake.eval({z: batch_z })
                errD_real = d_loss_real.eval({image: batch_images})
                errG = g_loss.eval({z: batch_z})
                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time:%4.4f, d_loss: %.8f, g_loss: %.8f" % (epoch, idx, batch_idxs,
                                                                                         time.time()-start_time, errD_fake+errD_real, errG))

                # show sample image while trainig
                if np.mod(counter, 30)==1:
                    samples, d_loss_sample, g_loss_sample = sess.run([sampler, d_loss, g_loss],
                                               feed_dict={z: sample_z, image: sample_images})

                    print("[Sample] d_loss:%.8f, g_loss:%.8f" % (d_loss_sample, g_loss_sample))
                    save_images(samples, (8, 8), '/home/dcgan_dir/train_{:02d}_{:04d}.png'.format(epoch, idx))
                    vnari=show_images(samples, (8,8))
                    plt.imshow(vnari)
                    plt.show()
                # save sess to directory
                if np.mod(counter, 100)==1:
                    saver.save(sess, "/home/cnn_dir/decgan")

def train():
    test_image = unpickle("dataset_images.pickle")
    X_image=np.array(test_image)/127.5 - 1.
    print(X_image.shape)

    # hyper params
    z = tf.placeholder(tf.float32, [None, 100])
    image = tf.placeholder(tf.float32, [64, 64, 64, 3])
    sample_z = np.random.uniform(-1, 1, size=(64, 100))
    batch_z = np.random.uniform(-1, 1, [64, 100])

    G = generator(z)  # G(z)
    D, D_logits = discriminator(image, reuse=False) # D(x)
    sampler = sampler(z)
    D_, D_logits_ = discriminator(G, reuse=True)   # D(G(z))

    # loss function and optim
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits_, labels=tf.zeros_like(D_)))
    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits_, labels=tf.ones_like(D_)))
    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits, labels=tf.ones_like(D)))

    d_loss = d_loss_real + d_loss_fake

    d_vars = [var for var in tf.trainable_variables() if 'd_' in var.name]
    g_vars = [var for var in tf.trainable_variables() if 'g_' in var.name]

    g_optim = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(g_loss, var_list=g_vars)
    d_optim = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(d_loss, var_list=d_vars)
    saver=tf.train.Saver()
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth=True
    saver = tf.train.Saver()
    dcgan_train(saver)

if __name__ == '__main__':
    train()

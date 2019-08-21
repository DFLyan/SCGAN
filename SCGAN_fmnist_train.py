#! /usr/bin/python
# -*- coding: utf8 -*-

import os, time, pickle, random, time
from datetime import datetime
import numpy as np
from time import localtime, strftime
import logging, scipy

import tensorflow as tf
import tensorlayer as tl
import math as ma
from model import *
from utils import *
from config import config, log_config

batch_size = config.TRAIN.batch_size
lr_init_scgan = config.TRAIN.lr_init_scgan
lr_scgan = config.TRAIN.lr_scgan
lr_decay = config.TRAIN.lr_decay
beta1 = config.TRAIN.beta1
n_epoch_init = config.TRAIN.n_epoch_init
n_epoch_scgan = config.TRAIN.n_epoch_scgan
decay_every_scgan = config.TRAIN.decay_every_scgan
ni = int(4)
ni_ = int(batch_size//4)

block_size = config.TRAIN.block_size_mnist
MR = config.TRAIN.MR
imagesize = block_size * block_size
size_y = 50
tv_weight = config.TRAIN.tv_weight


def read_all_imgs(img_list, path='', n_threads=32):
    """ Returns all images in array by given path and name of each image file. """
    imgs = []
    for idx in range(0, len(img_list), n_threads):
        b_imgs_list = img_list[idx: idx + n_threads]
        b_imgs = tl.prepro.threading_data(b_imgs_list, fn=get_imgs_fn, path=path)
        imgs.extend(b_imgs)
        print('read %d from %s' % (len(imgs), path))
    return imgs

def _tensor_size(tensor):
    from operator import mul
    return reduce(mul, (d.value for d in tensor.get_shape()), 1)

def train():
    save_dir_ginit = ("samples/fmnist_init_%d" % size_y).format(tl.global_flag['mode'])
    save_dir_scgan = ("samples/fmnist_g_%d" % size_y).format(tl.global_flag['mode'])
    tl.files.exists_or_mkdir(save_dir_ginit)
    tl.files.exists_or_mkdir(save_dir_scgan)
    checkpoint_dir = "checkpoint/fmnist_%d" % size_y
    tl.files.exists_or_mkdir(checkpoint_dir)

    train_hr_img = load_fmnist(split='train')

    t_target_image = tf.placeholder('float32', [batch_size, block_size, block_size, 1], name='t_target_image')
    y_image = tf.placeholder('float32', [batch_size, size_y], name='y_image')

    if not os.path.isfile("Gaussian28_%d.npy" % size_y) :
        A = np.random.normal(loc=0, scale=(1/size_y), size=[imagesize, int(size_y)])
        A = A.astype(np.float32)
        np.save("Gaussian28_%d.npy" % size_y,A)
    else:
        A = np.load("Gaussian28_%d.npy" % size_y, encoding='latin1')

    x_hat = tf.reshape(t_target_image, [batch_size, imagesize])
    y_meas = tf.matmul(x_hat, A)

    net_scgan = scgan_fm(y_image, is_train=True, reuse=False)
    net_d, logits_real = scgan_d(t_target_image, is_train=True, reuse=False)
    _, logits_fake = scgan_d(net_scgan.outputs, is_train=True, reuse=True)
    net_d.print_params(False)
    net_scgan.print_params(False)


    scgan_test = scgan_fm(y_image, is_train=False, reuse=True)

    d_loss1 = tl.cost.mean_squared_error(logits_real, tf.ones_like(logits_real), is_mean=True)
    d_loss2 = tl.cost.mean_squared_error(logits_fake, tf.zeros_like(logits_fake), is_mean=True)
    d_loss = d_loss1 + d_loss2
    tf.summary.scalar('d_loss', d_loss)


    img_p = tf.reshape(net_scgan.outputs, [batch_size, imagesize])
    y_meas_p = tf.matmul(img_p, A)

    y_meas_1 = tf.reshape(y_meas, [batch_size, -1])

    g_gan_loss = 1e-3 * tl.cost.mean_squared_error(logits_fake,tf.ones_like(logits_fake), is_mean=True)
    mse_loss = tl.cost.mean_squared_error(net_scgan.outputs, t_target_image, is_mean=True)
    meas_loss = tl.cost.mean_squared_error(y_meas_1,y_meas_p,is_mean=True)
    g_loss = mse_loss + g_gan_loss + meas_loss

    tf.summary.scalar('g_gan_loss', g_gan_loss)
    tf.summary.scalar('meas_loss', meas_loss)
    tf.summary.scalar('mse_loss', mse_loss)
    tf.summary.scalar('g_loss', g_loss)


    psnr = tf.constant(10,dtype=tf.float32) * log10(tf.constant(4, dtype=tf.float32) / (mse_loss))
    tf.summary.scalar('psnr', psnr)

    scgan_vars = tl.layers.get_variables_with_name('scgan', True, True)
    d_vars = tl.layers.get_variables_with_name('scgan_d', True, True)

    with tf.variable_scope('learning_rate_init'):
            lr_v_init = tf.Variable(lr_init_scgan, trainable=False)

    with tf.variable_scope('learning_rate_scgan'):
            lr_v_ = tf.Variable(lr_scgan, trainable=False)

    scgan_optim_init = tf.train.AdamOptimizer(lr_v_init).minimize(mse_loss, var_list=scgan_vars)
    scgan_optim = tf.train.RMSPropOptimizer(lr_v_).minimize(g_loss, var_list=scgan_vars)
    d_optim = tf.train.RMSPropOptimizer(lr_v_).minimize(d_loss, var_list=d_vars)

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))

    merged = tf.summary.merge_all()
    writer_init = tf.summary.FileWriter("logs/fmnist/%d/init" % size_y, tf.get_default_graph())
    writer_final = tf.summary.FileWriter("logs/fmnist/%d/final" % size_y, tf.get_default_graph())

    tl.layers.initialize_global_variables(sess)
    if tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir+'/g_fmnist_g.npz'.format(tl.global_flag['mode']), network=net_scgan) is False:
        tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/g_fmnist_init.npz'.format(tl.global_flag['mode']),network=net_scgan)
    tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir+'/d_fmnist.npz'.format(tl.global_flag['mode']), network=net_d)


###============================= TRAINING ===============================###
    sample_imgs = load_fmnist(split='val')
    sample_imgs = tl.prepro.threading_data(sample_imgs,fn=norm)
    sample_imgs__ = np.reshape(sample_imgs.astype(np.float32),[batch_size, block_size, block_size])
    sample_imgs_ = np.reshape(sample_imgs__,[batch_size, imagesize])

    sample_imgs_ = sess.run(tf.matmul(sample_imgs_, A))

    print('sample HR sub-image:',sample_imgs.shape, sample_imgs.min(),sample_imgs.max())
    tl.vis.save_images(sample_imgs__, [ni, ni_], save_dir_scgan + '/_train_sample.jpg')
    tl.vis.save_images(sample_imgs__, [ni, ni_], save_dir_ginit + '/_train_sample.jpg')

    sess.run(tf.assign(lr_v_,lr_init_scgan))
    print(" ** fixed learning rate: %f (for init f1)" % lr_init_scgan)

    for epoch in range(0,n_epoch_init+1):

        epoch_time = time.time()
        total_mse_loss, n_iter_scgan = 0, 0

        if epoch == 0:
            global sum_init
            sum_init = 0
        else:
            pass

        for idx in range(0,int(train_hr_img.shape[0] // batch_size)):
            step_time = time.time()
            b_imgs = train_hr_img[idx * batch_size: (idx + 1) * batch_size]
            b_imgs = tl.prepro.threading_data(b_imgs, fn=norm)
            b_imgs = np.reshape(b_imgs, [batch_size, block_size, block_size, 1])
            y_meas_ = sess.run(y_meas, feed_dict={t_target_image: b_imgs})
            errF, _, summary= sess.run([mse_loss, scgan_optim_init , merged],
                                                          {y_image:y_meas_, t_target_image:b_imgs})
            if n_iter_scgan % 50 == 0:
                print("Epoch [%2d/%2d] %4d time: %4.4fs,mse_loss: %.8f" % (
                epoch, n_epoch_init, n_iter_scgan, time.time() - step_time, errF))


            if sum_init % 10 == 0:
                writer_init.add_summary(summary,sum_init)

            total_mse_loss += errF
            n_iter_scgan += 1
            sum_init +=1

        log = "[*] Epoch: [%2d/%2d] time: %4.4fs, mse_loss: %.8f" % (
        epoch, n_epoch_init, time.time() - epoch_time,  total_mse_loss / n_iter_scgan)
        print(log)



        ## quick evaluation on train set
        if (epoch != 0) and (epoch % 1 == 0):
            out = sess.run(scgan_test.outputs,
                           {y_image: sample_imgs_})
            print("[*] save images")
            out = np.reshape(out, [batch_size, block_size, block_size])
            tl.vis.save_images(out, [ni, ni_], save_dir_ginit + '/train_%d.png' % epoch)


        ## save model
        if (epoch != 0) and (epoch % 1 == 0):
            tl.files.save_npz(net_scgan.all_params, name=checkpoint_dir + '/g_fmnist_init.npz'.format(tl.global_flag['mode']),sess=sess)



    for epoch in range(0,n_epoch_scgan+1):
        if epoch !=0 and (epoch % decay_every_scgan == 0):
            new_lr_decay = lr_decay ** (epoch // decay_every_scgan)
            sess.run(tf.assign(lr_v_, lr_scgan * new_lr_decay))
            log = " ** new learning rate: %f (for GAN)" % (lr_scgan * new_lr_decay)
            print(log)
        elif epoch == 0:
            sess.run(tf.assign(lr_v_, lr_scgan))
            log = " ** init lr: %f  decay_every_init: %d, lr_decay: %f (for GAN)" % (lr_scgan, decay_every_scgan, lr_decay)
            print(log)

        epoch_time = time.time()
        total_d_loss,total_dcgan_loss, n_iter_dcgan = 0, 0, 0

        if epoch == 0:
            global sum
            sum = 0
        else:
            pass

        for idx in range(0, int(train_hr_img.shape[0] // batch_size)):
            step_time = time.time()
            b_imgs = train_hr_img[idx * batch_size: (idx + 1) * batch_size]
            b_imgs = tl.prepro.threading_data(b_imgs, fn=norm)
            b_imgs = np.reshape(b_imgs, [batch_size, block_size, block_size, 1])
            y_meas_ = sess.run(y_meas, feed_dict={t_target_image: b_imgs})

            errD, _, = sess.run([d_loss,d_optim], {y_image: y_meas_, t_target_image: b_imgs} )

            errG, errF, errM, __, ___, summary = sess.run(
                [g_loss, mse_loss, meas_loss, scgan_optim, psnr, merged], {y_image: y_meas_, t_target_image: b_imgs})
            if n_iter_dcgan % 50 == 0:
                print(
                    "Epoch [%2d/%2d] %4d time: %4.4fs, d_loss: %.8f, g_loss: %.8f(mse_loss: %.4f,meas_loss：%.8f)" % (
                        epoch, n_epoch_scgan, n_iter_dcgan, time.time() - step_time, errD, errG, errF, errM))

            if sum % 10 == 0:
                writer_final.add_summary(summary, sum)

            total_dcgan_loss += errG
            total_d_loss += errD
            n_iter_dcgan += 1
            sum += 1

        log = "[*] Epoch: [%2d/%2d] time: %4.4fs, d_loss: %.8f, g_loss: %.8f" % (
        epoch, n_epoch_scgan, time.time() - epoch_time, total_d_loss/n_iter_dcgan, total_dcgan_loss / n_iter_dcgan)
        print(log)



        ## quick evaluation on train set
        if (epoch != 0) and (epoch % 1 == 0):
            out = sess.run(scgan_test.outputs,
                           {y_image: sample_imgs_})  # ; print('gen sub-image:', out.shape, out.min(), out.max())
            print("[*] save images")
            out = np.reshape(out, [batch_size, block_size, block_size])
            tl.vis.save_images(out, [ni, ni_], save_dir_scgan + '/train_%d.png' % epoch)

        ## save model
        if (epoch != 0) and (epoch % 5 == 0):
            tl.files.save_npz(net_scgan.all_params, name=checkpoint_dir + '/g_fmnist_g.npz'.format(tl.global_flag['mode']), sess=sess)
            tl.files.save_npz(net_d.all_params, name=checkpoint_dir+'/d_fmnist.npz'.format(tl.global_flag['mode']), sess=sess)




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='gray', help='gray, evaluate')

    args = parser.parse_args()

    tl.global_flag['mode'] = args.mode

    if tl.global_flag['mode'] == 'gray':
        train()
    else:
        raise Exception("Unknow --mode")

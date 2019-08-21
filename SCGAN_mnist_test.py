#! /usr/bin/python
# -*- coding: utf8 -*-

import os, time, random, time
from datetime import datetime
import numpy as np
from time import localtime, strftime
import logging, scipy
from skimage.measure import compare_ssim as ssim_c
import tensorflow as tf
import tensorlayer as tl
import math as ma
from model import *
from utils import *
from config import config, log_config
from skimage.measure import compare_psnr, compare_mse


def read_all_imgs(img_list, path='', n_threads=32):
    """ Returns all images in array by given path and name of each image file. """
    imgs = []
    for idx in range(0, len(img_list), n_threads):
        b_imgs_list = img_list[idx : idx + n_threads]
        b_imgs = tl.prepro.threading_data(b_imgs_list, fn=get_imgs_fn, path=path)
        imgs.extend(b_imgs)
        print('read %d from %s' % (len(imgs), path))
    return imgs


def save_image(image, path):
    """Save an image as a png file."""
    min_val = image.min()
    if min_val < 0:
        image = image + min_val

    scipy.misc.imsave(path, image)
    print('[#] Image saved {}.'.format(path))


def evaluate():
    block_size = config.TRAIN.block_size_mnist
    imagesize = block_size * block_size
    size_y = 400
    test_num = 1000

    ## create folders to save result images
    save_dir = ("samples/mnist/%d" % size_y).format(tl.global_flag['mode'])
    tl.files.exists_or_mkdir(save_dir)
    checkpoint_dir = "checkpoint/mnist_%d" % size_y

    ###====================== PRE-LOAD DATA ===========================###
    test_hr_imgs = load_mnist(split='test')
    test_hr_imgs = test_hr_imgs[0:test_num]

    ###========================== DEFINE MODEL ============================###
    t_target_image = tf.placeholder('float32', [1, block_size, block_size, 1], name='t_target_image')
    y_image = tf.placeholder('float32', [1, size_y], name='y_image')

    A = np.load("Gaussian28_%d.npy" % size_y, encoding='latin1')

    x_hat = tf.reshape(t_target_image, [1, imagesize])
    y_meas = tf.matmul(x_hat, A)

    net_g = scgan(y_image, is_train=False, reuse=False)


    ###========================== RESTORE G =============================###
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    tl.layers.initialize_global_variables(sess)
    tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir+'/g_mnist_g.npz', network=net_g)

    ###======================= EVALUATION =============================###
    global sum, sum_s, sum_t
    sum = 0
    sum_s = 0
    sum_t = 0
    sum_m = 0

    for imid in range(0, test_hr_imgs.shape[0]):
        b_imgs_ = tl.prepro.threading_data(test_hr_imgs[imid:imid+1], fn=norm_0)
        b_imgs__ = np.reshape(b_imgs_, [1, block_size, block_size, 1])
        y_meas_ = sess.run(y_meas, feed_dict={t_target_image: b_imgs__})
        start_time = time.time()
        out = sess.run(net_g.outputs, feed_dict={y_image: y_meas_})
        print("took: %4.4fs" % (time.time() - start_time))
        sum_t += (time.time() - start_time)

        psnr = compare_psnr(b_imgs_.astype(np.float32), out)
        print("PSNR:%.8f" % psnr)
        mse = compare_mse(b_imgs_.astype(np.float32), out)
        imgs = np.reshape(b_imgs__, [block_size, block_size])
        out_ = np.reshape(out, [block_size, block_size])
        ssim = ssim_c(X=imgs.astype(np.float32), Y=out_, multichannel=False)
        print("SSIM:%.8f" % ssim)
        sum += psnr
        sum_m += mse
        sum_s += ssim

        print("[*] save images")
        out = np.reshape(out, [block_size, block_size])
        save_image(out, save_dir+'/test_gen%d.jpg' % imid)
        b_imgs_hr = np.reshape(b_imgs_, [block_size, block_size])
        save_image(b_imgs_hr, save_dir+'/test_hr%d.jpg' % imid)

    print("TIME_SUM:%.8f" % sum_t)
    print("Num of image:%d" % len(test_hr_imgs))
    psnr_a = sum / len(test_hr_imgs)
    print("PSNR_AVERAGE:%.8f" % psnr_a)
    mse_a = sum_m/ len(test_hr_imgs)
    print("MSE_AVERAGE:%.8f" % mse_a)
    ssim_a = sum_s / len(test_hr_imgs)
    print("SSIM_AVERAGE:%.8f" % ssim_a)
    time_a = sum_t / len(test_hr_imgs)
    print("TIME_AVERAGE:%.8f" % time_a)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='evaluate_mnist', help='evaluate_mnist')

    args = parser.parse_args()

    tl.global_flag['mode'] = args.mode

    if tl.global_flag['mode'] == 'evaluate_mnist':
        evaluate()
    else:
        raise Exception("Unknow --mode")
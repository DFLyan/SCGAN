#! /usr/bin/python
# -*- coding: utf8 -*-

import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
from config import config, log_config
# from tensorflow.python.ops import variable_scope as vs
# from tensorflow.python.ops import math_ops, init_ops, array_ops, nn
# from tensorflow.python.util import nest
# from tensorflow.contrib.rnn.python.ops import core_rnn_cell

# https://github.com/david-gpu/srez/blob/master/srez_model.py

batch_size = config.TRAIN.batch_size


###full image of celebA###
def scgan(y, is_train=False, reuse=False):
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None # tf.constant_initializer(value=0.0)
    g_init = tf.random_normal_initializer(1., 0.02)
    with tf.variable_scope("scgan", reuse=reuse) as vs:
        tl.layers.set_name_reuse(reuse)
        n = InputLayer(y)
        n = tl.layers.DenseLayer(n, n_units=49500, act=tf.identity, W_init=w_init, b_init=b_init, name='dense0')
        n = tl.layers.ReshapeLayer(n, shape=[-1, 11, 9, 500], name='reshape1')
        n = Conv2d(n, 500, (3, 3), (1, 1), act=tf.nn.selu, padding='SAME', W_init=w_init, name='n512s1/0')

        n = SubpixelConv2d(n, scale=5, n_out_channel=None, act=tf.nn.selu, name='pixelshufflerx2/1')
        n = BatchNormLayer(n, act=tf.nn.selu, is_train=is_train, gamma_init=g_init, name='n512s1/b1')
        n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='n512s1/1')
        n = Conv2d(n, 256, (3, 3), (1, 1), act=tf.nn.selu, padding='SAME', W_init=w_init, name='n512s1/2')

        n = SubpixelConv2d(n, scale=2, n_out_channel=None, act=tf.nn.selu, name='pixelshufflerx2/2')
        n = BatchNormLayer(n, act=tf.nn.selu, is_train=is_train, gamma_init=g_init, name='n512s1/b2')
        n = Conv2d(n, 128, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='n512s1/3')
        n = Conv2d(n, 128, (3, 3), (1, 1), act=tf.nn.selu, padding='SAME', W_init=w_init, name='n512s1/4')

        n = SubpixelConv2d(n, scale=2, n_out_channel=None, act=tf.nn.selu, name='pixelshufflerx2/3')
        n = BatchNormLayer(n, act=tf.nn.selu, is_train=is_train, gamma_init=g_init, name='n512s1/b3')
        n = Conv2d(n, 32, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='n512s1/5')

        n = Conv2d(n, 3, (3, 3), (1, 1), act=tf.nn.tanh, padding='SAME', W_init=w_init, name='out')

        return n


## MNIST ###
def scgan_m(y, is_train=False, reuse=False):
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None # tf.constant_initializer(value=0.0)
    g_init = tf.random_normal_initializer(1., 0.02)
    with tf.variable_scope("scgan", reuse=reuse) as vs:
        tl.layers.set_name_reuse(reuse)
        n = InputLayer(y)
        n = tl.layers.DenseLayer(n,n_units=12544, act=tf.identity, W_init=w_init, b_init=b_init, name='dense0')
        n = tl.layers.ReshapeLayer(n, shape=[-1, 7, 7, 256],name='reshape1')

        n = SubpixelConv2d(n, scale=2, n_out_channel=None, act=tf.nn.selu, name='pixelshufflerx2/2')
        n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='n512s1/2')
        n = BatchNormLayer(n, act=tf.nn.selu, is_train=is_train, gamma_init=g_init, name='n512s1/b2')
        # n = DropoutLayer(n, keep=0.5, is_train=is_train, is_fix=True, name='dropout1')

        n = SubpixelConv2d(n, scale=2, n_out_channel=None, act=tf.nn.selu, name='pixelshufflerx2/4')
        n = Conv2d(n, 32, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='n64s1/3')
        n = BatchNormLayer(n, act=tf.nn.selu, is_train=is_train, gamma_init=g_init, name='n512s1/b3')#之前的模型测试这条删除
        n = DropoutLayer(n, keep=0.6, is_train=is_train, is_fix=True, name='dropout2')

        n = Conv2d(n, 1, (1, 1), (1, 1), act=tf.nn.sigmoid, padding='SAME', W_init=w_init, name='out')

        return n


## (f)mnist ###
def scgan_fm(y, is_train=False, reuse=False):
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None # tf.constant_initializer(value=0.0)
    g_init = tf.random_normal_initializer(1., 0.02)
    with tf.variable_scope("scgan", reuse=reuse) as vs:
        tl.layers.set_name_reuse(reuse)
        n = InputLayer(y)
        n = tl.layers.DenseLayer(n,n_units=12544, act=tf.identity, W_init=w_init, b_init=b_init, name='dense0')
        n = tl.layers.ReshapeLayer(n, shape=[-1, 7, 7, 256],name='reshape1')

        n = Conv2d(n, 512, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='n512s1/1')
        n = BatchNormLayer(n, act=tf.nn.selu, is_train=is_train, gamma_init=g_init, name='n512s1/b1')
        n = SubpixelConv2d(n, scale=2, n_out_channel=None, act=tf.nn.selu, name='pixelshufflerx2/2')

        n = Conv2d(n, 512, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='n512s1/2')
        n = BatchNormLayer(n, act=tf.nn.selu, is_train=is_train, gamma_init=g_init, name='n512s1/b2')
        n = SubpixelConv2d(n, scale=2, n_out_channel=None, act=tf.nn.selu, name='pixelshufflerx2/3')

        n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='n64s1/3')
        n = BatchNormLayer(n, act=tf.nn.selu, is_train=is_train, gamma_init=g_init, name='n64s1/b4')

        n = Conv2d(n, 1, (1, 1), (1, 1), act=tf.nn.tanh, padding='SAME', W_init=w_init, name='out')

        return n



def scgan_d(t_image, is_train=False, reuse=False):
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None
    g_init = tf.random_normal_initializer(1., 0.02)
    lrelu = lambda x : tl.act.lrelu(x, 0.2)
    with tf.variable_scope("scgan_d", reuse=reuse) as vs:
        tl.layers.set_name_reuse(reuse)
        n = InputLayer(t_image, name='in')

        n = Conv2d(n, 64, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init, name='n64s1/c')

        n = Conv2d(n, 64, (3, 3), (2, 2), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='n64s2/c')
        n = BatchNormLayer(n, act=lrelu, is_train=is_train, gamma_init=g_init, name='n64s2/b')

        n = Conv2d(n, 128, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='n128s1/c')
        n = BatchNormLayer(n, act=lrelu, is_train=is_train, gamma_init=g_init, name='n128s1/b')

        n = Conv2d(n, 128, (3, 3), (2, 2), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='n128s2/c')
        n = BatchNormLayer(n, act=lrelu, is_train=is_train, gamma_init=g_init, name='n128s2/b')

        n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='n256s1/c')
        n = BatchNormLayer(n, act=lrelu, is_train=is_train, gamma_init=g_init, name='n256s1/b')

        n = Conv2d(n, 256, (3, 3), (2, 2), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='n256s2/c')
        n = BatchNormLayer(n, act=lrelu, is_train=is_train, gamma_init=g_init, name='n256s2/b')

        n = Conv2d(n, 512, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='n512s1/c')
        n = BatchNormLayer(n, act=lrelu, is_train=is_train, gamma_init=g_init, name='n512s1/b')

        n = Conv2d(n, 512, (3, 3), (2, 2), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='n512s2/c')
        n = BatchNormLayer(n, act=lrelu, is_train=is_train, gamma_init=g_init, name='n512s2/b')


        n = FlattenLayer(n, name='f')
        n = DenseLayer(n, n_units=1024, act=None, name='d1024')

        logits = n.outputs

        return n, logits


import os
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.prepro import *
import random
import scipy
import numpy as np
from functools import reduce

DIMX = 256
DIMY = 256


def get_imgs_fn(file_name, path):
    """ Input an image path and name, return an image array """
    # return scipy.misc.imread(path + file_name).astype(np.float)
    return scipy.misc.imread(path + file_name, mode='RGB')


def get_gray_imgs_fn(file_name, path):
    """ Input an image path and name, return an image array """
    # return scipy.misc.imread(path + file_name).astype(np.float)
    return scipy.misc.imread(path + file_name, mode='L')


def save_image(image, path):
    """Save an image as a png file."""
    min_val = image.min()
    if min_val < 0:
        image = image + min_val

    scipy.misc.imsave(path, image)
    print('[#] Image saved {}.'.format(path))


def crop_sub_imgs_fn(x, is_random=True):
    x = crop(x, wrg=64, hrg=64, is_random=is_random)
    x = x / (255. / 2.)
    x = x - 1.
    return x


def pad_image_zero2(x):
    x1 = x[:, 0:1, :]
    x2 = x[:, 177:178, :]
    x = np.concatenate((x1, x), axis=1)
    x = np.concatenate((x, x2), axis=1)
    x3 = x[0:1, :, :]
    x4 = x[217:218, :, :]
    x = np.concatenate((x3, x), axis=0)
    x = np.concatenate((x, x4), axis=0)
    x = x / (255. / 2.)
    x = x - 1.
    return x


def norm(x):
    x = x / (255. / 2.)
    x = x - 1.
    # x = x / 255
    return x


def norm_0(x):
    # x = x / (255. / 2.)
    # x = x - 1.
    x = x / 255
    return x


def inv_norm(x):
    x = x + 1
    x = x * (255. / 2.)
    # x = x * 255
    return x


def inv_norm_0(x):
    # x = x + 1
    # x = x * (255. / 2.)
    x = x * 255
    return x


def augm(x):
    size = x.shape
    x = tl.prepro.flip_axis(x, axis=0, is_random=True)
    x = tl.prepro.flip_axis(x, axis=1, is_random=True)
    x = np.reshape(x, (size[0], size[1], 1))
    rg = random.sample([0, 90, 180, 270], 1)
    rg = rg[0]
    x = tl.prepro.rotation(x, rg=rg, is_random=False)
    return x


def psnr_c(img1,img2):
    diff = np.abs(img1-img2)
    rmse = np.sqrt(np.mean(np.square(diff)))
    psnr = 20*np.log10(1/rmse)
    return psnr


def log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10,dtype=numerator.dtype))
    return numerator / denominator


def _tensor_size(tensor):
    from operator import mul
    return reduce(mul, (d.value for d in tensor.get_shape()), 1)


def new_act(x, name='new_act'):
    return tf.clip_by_value(x, 0, 1, name=name)


def load_mnist(split='train'):

    data_dir = 'data/mnist/'

    fd = open(os.path.join(data_dir, 'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    train_images = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float)

    fd = open(os.path.join(data_dir, 'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    train_labels = loaded[8:].reshape((60000)).astype(np.float)

    fd = open(os.path.join(data_dir, 't10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    test_images = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)

    fd = open(os.path.join(data_dir, 't10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    test_labels = loaded[8:].reshape((10000)).astype(np.float)

    train_labels = np.asarray(train_labels)
    test_labels = np.asarray(test_labels)
    if split == 'train':
        images = train_images[:50000]
        labels = train_labels[:50000]
    elif split == 'val':
        images = train_images[50000:50064]
        labels = train_labels[50000:50064]
    elif split == 'test':
        images = test_images
        labels = test_labels

    # ids = range(len(labels))

    return images


def load_fmnist(split='train', lazy=False):
    """Implements the load function.

    Args:
        split: Dataset split, can be [train|dev|test], default: train.
        lazy: Not used for F-MNIST.

    Returns:
         Images of np.ndarray, Int array of labels, and int array of ids.

    Raises:
        ValueError: If split is not one of [train|val|test].
    """

    data_dir = 'data/f-mnist/'

    fd = open(os.path.join(data_dir, 'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    train_images = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float)

    fd = open(os.path.join(data_dir, 'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    train_labels = loaded[8:].reshape((60000)).astype(np.float)

    fd = open(os.path.join(data_dir, 't10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    test_images = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)

    fd = open(os.path.join(data_dir, 't10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    test_labels = loaded[8:].reshape((10000)).astype(np.float)

    train_labels = np.asarray(train_labels)
    test_labels = np.asarray(test_labels)
    if split == 'train':
        images = train_images[:50000]
        labels = train_labels[:50000]
    elif split == 'val':
        images = train_images[50000:50064]
        labels = train_labels[50000:50064]
    elif split == 'test':
        images = test_images
        labels = test_labels
    else:
        raise ValueError('[!] Invalid split {}.'.format(split))

    # ids = range(len(labels))
    return images
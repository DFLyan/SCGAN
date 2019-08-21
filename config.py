from easydict import EasyDict as edict
import json
import math as ma

config = edict()
config.TRAIN = edict()
config.VALID = edict()
config.TEST = edict()

config.TRAIN.block_size_mnist = 28
config.TRAIN.tv_weight = 0.0002

config.TRAIN.batch_size = 8

config.TRAIN.lr_init_scgan = 0.001
config.TRAIN.lr_scgan = 0.0002

config.TRAIN.beta1 = 0.9

config.TRAIN.n_epoch_init = 10
config.TRAIN.n_epoch_scgan = 30

config.TRAIN.lr_decay = 0.8

config.TRAIN.decay_every_scgan = int(40)

## train set location
config.TRAIN.hr_img_path = 'data/celebA_TVT/train/'

## valid set location
config.VALID.hr_img_path = 'data/celebA_TVT/valid/'

## test
config.TEST.hr_img_path = 'data/celebA_TVT/test/'

def log_config(filename, cfg):
    with open(filename, 'w') as f:
        f.write("================================================\n")
        f.write(json.dumps(cfg, indent=4))
        f.write("\n================================================\n")

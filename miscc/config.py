from __future__ import division
from __future__ import print_function

import os.path as osp
import numpy as np
from easydict import EasyDict as edict


__C = edict()
cfg = __C

# Dataset name: flowers, birds
__C.DATASET_NAME = 'birds'
__C.CONFIG_NAME = ''
__C.DATA_DIR = ''
__C.SAVE_DIR = ''
__C.GPU_ID = '0'
__C.CUDA = True

__C.WORKERS = 6

__C.TREE = edict()
__C.TREE.BRANCH_NUM = 3
__C.TREE.BASE_SIZE = 64
__C.SUPER_CATEGORIES = 20
__C.FINE_GRAINED_CATEGORIES = 200
__C.TEST_CHILD_CLASS = 0
__C.TEST_PARENT_CLASS = 0
__C.TEST_BACKGROUND_CLASS = 0
__C.TIED_CODES = True
__C.TEST_DEPTH = 4

# Test options
__C.TEST = edict()

# Training options
__C.TRAIN = edict()
__C.TRAIN.BATCH_SIZE = 64
__C.TRAIN.BG_LOSS_WT_LCL = 10
__C.TRAIN.BG_LOSS_WT_GLB = 10
__C.TRAIN.BG_CLASSI_WT = 10
__C.TRAIN.DATASET_SIZE = 10000
__C.TRAIN.VIS_COUNT = 64
__C.TRAIN.MAX_EPOCH = 600
__C.TRAIN.HARDNEG_MAX_ITER = 1500
__C.TRAIN.SNAPSHOT_INTERVAL = 2000
__C.TRAIN.SAVEMODEL_INTERVAL = 8000
__C.TRAIN.SNAPSHOT_INTERVAL_HARDNEG = 500
__C.TRAIN.DISCRIMINATOR_LR = 2e-4
__C.TRAIN.GENERATOR_LR = 2e-4
__C.TRAIN.FLAG = True
__C.TRAIN.NET_G = ''
__C.TRAIN.NET_D = ''

__C.TRAIN.PRE_PAD = True
__C.TRAIN.RAND_PAD = True
__C.TRAIN.PAD = [30, 26, 21, 10]
__C.TRAIN.PAD_ALT = [1, 1, 1, 1]
__C.TRAIN.START_DEPTH = 0
__C.TRAIN.END_DEPTH = 3
__C.TRAIN.BATCHSIZE_PER_DEPTH = [64, 32, 16, 4]
__C.TRAIN.BLEND_EPOCHS_PER_DEPTH = [0, 60, 80, 100]
__C.TRAIN.STABLE_EPOCHS_PER_DEPTH = [2, 80, 100, 120]


# Modal options
__C.GAN = edict()
__C.GAN.DF_DIM = 64
__C.GAN.GF_DIM = 64
__C.GAN.Z_DIM = 100
__C.GAN.NETWORK_TYPE = 'default'
__C.GAN.R_NUM = 2




def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if k not in b:
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(b[k]),
                                                               type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)

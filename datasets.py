from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import sys

import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
import PIL
import os
import os.path
import pickle
import random
import numpy as np
import pandas as pd
from miscc.config import cfg

import torch.utils.data as data
from PIL import Image
import os
import os.path
import six
import string
import sys
import torch
from copy import deepcopy
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG',
                  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def get_mask(imshape, bbox):
    """
    mask: 0's for foreground
    """
    c, r = imshape
    x1, y1, w, h = bbox
    x2 = x1 + w - 1
    y2 = y1 + h - 1
    mk = np.ones((r, c))
    mk[y1: y2 + 1, x1: x2 + 1] = 0
    return Image.fromarray(mk)


def pad_mask(mask, pad):
    """
    - mask (PIL.Image)
    """
    c, r = mask.size
    ys, xs = np.where(np.array(mask) == 0)
    if len(xs) == 0:
        bbox = [0, 0, 0, 0]
        return get_mask(mask.size, bbox)

    if cfg.TRAIN.RAND_PAD and random.random() > 0.8:
        x1 = min(xs)
    else:
        x1 = min(xs) - pad
        x1 = max(0, x1)
    if cfg.TRAIN.RAND_PAD and random.random() > 0.8:
        y1 = min(ys)
    else:
        y1 = min(ys) - pad
        y1 = max(0, y1)
    if cfg.TRAIN.RAND_PAD and random.random() > 0.8:
        x2 = max(xs)
    else:
        x2 = max(xs) + pad
        x2 = min(c-1, x2)
    if cfg.TRAIN.RAND_PAD and random.random() > 0.8:
        y2 = max(ys)
    else:
        y2 = max(ys) + pad
        y2 = min(r-1, y2)

    bbox = [x1, y1, x2-x1+1, y2-y1+1]
    return get_mask(mask.size, bbox)


def get_imgs(img_path, imsize, bbox=None,
             transform=None, normalize=None):
    # imsize = 32 * (2 ** cur_depth)
    pre_pad = 1
    post_pad = 1
    img = Image.open(img_path).convert('RGB')
    mask = get_mask(img.size, bbox)
    if cfg.TRAIN.PRE_PAD:
        mask = pad_mask(mask, pre_pad)

    width, height = img.size
    if bbox is not None:
        r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
        center_x = int((2 * bbox[0] + bbox[2]) / 2)
        center_y = int((2 * bbox[1] + bbox[3]) / 2)
        y1 = np.maximum(0, center_y - r)
        y2 = np.minimum(height, center_y + r)
        x1 = np.maximum(0, center_x - r)
        x2 = np.minimum(width, center_x + r)
        cimg = img.crop([x1, y1, x2, y2])
        fimg = deepcopy(img)
        fimg_arr = np.array(fimg)
        fimg = Image.fromarray(fimg_arr)

    if transform is not None:
        cimg = transform(cimg)

    retf = []
    retc = []
    re_cimg = transforms.Scale(imsize)(cimg)

    retc = normalize(re_cimg)

    resize = transforms.Resize(int(imsize * 76 / 64))
    re_fimg = resize(fimg)
    re_mk = resize(mask)

    i, j, h, w = transforms.RandomCrop.get_params(
        re_fimg, output_size=(imsize, imsize))
    re_fimg = TF.crop(re_fimg, i, j, h, w)
    re_mk = TF.crop(re_mk, i, j, h, w)

    if random.random() > 0.5:
        re_fimg = TF.hflip(re_fimg)
        re_mk = TF.hflip(re_mk)

    retf = normalize(re_fimg)

    if not cfg.TRAIN.PRE_PAD:
        re_mk = pad_mask(re_mk, post_pad)

    retmk = torch.tensor(np.array(re_mk)).view(1, imsize, imsize)
    return retf, retc, retmk


def get_aux_mask(img_path, imsize, bbox=None,
                 transform=None, normalize=None):
    pre_pad = 1
    post_pad = 1
    img = Image.open(img_path).convert('RGB')
    mask = get_mask(img.size, bbox)
    if cfg.TRAIN.PRE_PAD:
        mask = pad_mask(mask, pre_pad)

    width, height = img.size
    if bbox is not None:
        r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
        center_x = int((2 * bbox[0] + bbox[2]) / 2)
        center_y = int((2 * bbox[1] + bbox[3]) / 2)
        y1 = np.maximum(0, center_y - r)
        y2 = np.minimum(height, center_y + r)
        x1 = np.maximum(0, center_x - r)
        x2 = np.minimum(width, center_x + r)
        fimg = deepcopy(img)
        fimg_arr = np.array(fimg)
        fimg = Image.fromarray(fimg_arr)

    resize = transforms.Resize(int(imsize * 76 / 64))
    re_fimg = resize(fimg)
    re_mk = resize(mask)

    i, j, h, w = transforms.RandomCrop.get_params(
        re_fimg, output_size=(imsize, imsize))
    re_mk = TF.crop(re_mk, i, j, h, w)

    if random.random() > 0.5:
        re_mk = TF.hflip(re_mk)

    if not cfg.TRAIN.PRE_PAD:
        re_mk = pad_mask(re_mk, post_pad)

    retmk = torch.tensor(np.array(re_mk)).view(1, imsize, imsize)
    return retmk


class Dataset(data.Dataset):
    def __init__(self, data_dir, imsize, transform=None):

        self.transform = transform
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.imsize = imsize
        # self.imsize = 32 * (2 ** cur_depth)

        self.data = []
        self.data_dir = data_dir
        self.bbox = self.load_bbox()
        self.filenames = self.load_filenames(data_dir)
        if cfg.TRAIN.FLAG:
            self.iterator = self.prepair_training_pairs
        else:
            self.iterator = self.prepair_test_pairs

    # only used in background stage
    def load_bbox(self):
        # Returns a dictionary with image filename as 'key' and its bounding box coordinates as 'value'

        data_dir = self.data_dir
        bbox_path = os.path.join(data_dir, 'bounding_boxes.txt')
        df_bounding_boxes = pd.read_csv(bbox_path,
                                        delim_whitespace=True,
                                        header=None).astype(int)
        filepath = os.path.join(data_dir, 'images.txt')
        df_filenames = \
            pd.read_csv(filepath, delim_whitespace=True, header=None)
        filenames = df_filenames[1].tolist()
        print('Total filenames: ', len(filenames), filenames[0])
        filename_bbox = {img_file[:-4]: [] for img_file in filenames}
        numImgs = len(filenames)
        for i in range(0, numImgs):
            bbox = df_bounding_boxes.iloc[i][1:].tolist()
            key = filenames[i][:-4]
            filename_bbox[key] = bbox
        return filename_bbox

    def load_filenames(self, data_dir):
        filepath = os.path.join(data_dir, 'images.txt')
        df_filenames = \
            pd.read_csv(filepath, delim_whitespace=True, header=None)
        filenames = df_filenames[1].tolist()
        filenames = [fname[:-4] for fname in filenames]
        print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
        return filenames

    def prepair_training_pairs(self, index):
        key = self.filenames[index]
        if self.bbox is not None:
            bbox = self.bbox[key]
        else:
            bbox = None
        data_dir = self.data_dir
        img_name = '%s/images/%s.jpg' % (data_dir, key)
        fimgs, cimgs, masks = get_imgs(img_name, self.imsize,
                                       bbox, self.transform, normalize=self.norm)

        # Randomly generating child code during training
        rand_class = random.sample(range(cfg.FINE_GRAINED_CATEGORIES), 1)
        c_code = torch.zeros([cfg.FINE_GRAINED_CATEGORIES, ])
        c_code[rand_class] = 1

        # load exiliary bbox
        mk_id = random.randint(0, len(self.filenames)-1)
        key = self.filenames[mk_id]
        if self.bbox is not None:
            bbox = self.bbox[key]
        else:
            bbox = None
        data_dir = self.data_dir
        img_name = '%s/images/%s.jpg' % (data_dir, key)
        ex_masks = get_aux_mask(img_name, self.imsize,
                                bbox, self.transform, normalize=self.norm)

        return fimgs, cimgs, c_code, key, masks, ex_masks

    def prepair_test_pairs(self, index):
        key = self.filenames[index]
        if self.bbox is not None:
            bbox = self.bbox[key]
        else:
            bbox = None
        data_dir = self.data_dir
        c_code = self.c_code[index, :, :]
        img_name = '%s/images/%s.jpg' % (data_dir, key)
        _, imgs, _ = get_imgs(img_name, self.imsize,
                              bbox, self.transform, normalize=self.norm)

        return imgs, c_code, key

    def __getitem__(self, index):
        return self.iterator(index)

    def __len__(self):
        return len(self.filenames)

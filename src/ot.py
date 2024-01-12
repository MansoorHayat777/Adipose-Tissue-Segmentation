# -*- coding: utf-8 -*-

import os

import numpy as np
import scipy.io as sio
import torch
from PIL import Image
from torch.utils import data

num_classes = 3
ignore_label = 255
	
root = '/home/pan_member/fat/ot'



#调色板
palette = [0, 0, 0, 255, 0, 0, 0, 255, 0]

zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask


def make_dataset(mode):
    assert mode in ['train', 'val', 'test']
    items = []
    if mode == 'train':
        img_path = os.path.join(root, 'train', 'source')
        mask_path = os.path.join(root, 'train', 'label')
        for it in list(range(21,160)):
            item = (os.path.join(img_path,  "IM"+str(it)+"_source"+".png"), os.path.join(mask_path, "IM" + str(it) + "_label_nocoler" + ".png"))
            items.append(item)
    elif mode == 'val':
        img_path = os.path.join(root, 'val', 'source')
        mask_path = os.path.join(root, 'val', 'label')
        for it in list(range(0, 20)):
            item = (os.path.join(img_path, "IM" + str(it) + "_source" + ".png"),
                    os.path.join(mask_path, "IM" + str(it) + "_label_nocoler" + ".png"))
            items.append(item)
    return items


class OT(data.Dataset):
    def __init__(self, mode, joint_transform=None, sliding_crop=None, transform=None, target_transform=None):
        self.imgs = make_dataset(mode)
        if len(self.imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set')
        self.mode = mode
        self.joint_transform = joint_transform
        self.sliding_crop = sliding_crop
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        if self.mode == 'test':
            img_path, img_name = self.imgs[index]
            img = Image.open(os.path.join(img_path, img_name + '.jpg')).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            return img_name, img

        img_path, mask_path = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)

        if self.joint_transform is not None:
            img, mask = self.joint_transform(img, mask)

        if self.sliding_crop is not None:
            img_slices, mask_slices, slices_info = self.sliding_crop(img, mask)
            if self.transform is not None:
                img_slices = [self.transform(e) for e in img_slices]
            if self.target_transform is not None:
                mask_slices = [self.target_transform(e) for e in mask_slices]
            img, mask = torch.stack(img_slices, 0), torch.stack(mask_slices, 0)
            return img, mask, torch.LongTensor(slices_info)
        else:
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                mask = self.target_transform(mask)
            return img, mask

    def __len__(self):
        return len(self.imgs)

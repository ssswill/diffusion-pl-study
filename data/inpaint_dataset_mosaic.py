import random
import os.path as op
import numpy as np
import pickle as pkl
from pathlib2 import Path
from torchvision import transforms
import torch

import torch.utils.data as data
from . import common
from .util.mask import (bbox2mask, brush_stroke_mask, get_irregular_mask, random_bbox, random_cropping_bbox)


class InpaintDatasetMosaic(data.Dataset):
    def __init__(self, data_root, mask_mode='center', data_len=-1, image_size=[256, 256], loader=common.pil_loader, train=False):
        imgs = common.make_dataset(data_root)
        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs
        self.tfs = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
            # 1.0-255 -> 0-1。 2. -均值/标准差。(均值和标准差，根据所有数据计算得到。每张图片就不需要单独计算。)
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.loader = loader
        self.mask_mode = mask_mode
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        img = self.tfs(self.loader(path))
        mask = self.get_mask()
        x_input = img * (1. - mask) + mask * torch.randn_like(img)
        mask_img = img * (1. - mask) + mask

        ret['y_gt'] = img
        ret['x_input'] = x_input
        ret['mask_image'] = mask_img
        ret['mask'] = mask
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_mask(self):
        if self.mask_mode == 'bbox':
            mask = bbox2mask(self.image_size, random_bbox())
        elif self.mask_mode == 'center':
            h, w = self.image_size
            mask = bbox2mask(self.image_size, (h // 4, w // 4, h // 2, w // 2))
        elif self.mask_mode == 'irregular':
            mask = get_irregular_mask(self.image_size)
        elif self.mask_mode == 'free_form':
            mask = brush_stroke_mask(self.image_size)
        elif self.mask_mode == 'hybrid':
            regular_mask = bbox2mask(self.image_size, random_bbox())
            irregular_mask = brush_stroke_mask(self.image_size, )
            mask = regular_mask | irregular_mask
        elif self.mask_mode == 'file':
            pass
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask).permute(2, 0, 1)
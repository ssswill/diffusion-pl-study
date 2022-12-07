import os
import random
import os.path as op
import numpy as np
import pickle as pkl

from PIL import Image
from pathlib2 import Path
from torchvision import transforms
import torch

import torch.utils.data as data
from . import common
from .util.mask import (bbox2mask, brush_stroke_mask, get_irregular_mask, random_bbox, random_cropping_bbox)


class InpaintDatasetSr(data.Dataset):
    def __init__(self, data_root, data_len=-1, image_size=256, train=False):
        self.data_root = data_root
        flist = common.make_dataset(data_root)
        if data_len > 0:
            self.flist = flist[:int(data_len)]
        else:
            self.flist = flist
        self.tfs = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.tf_down = transforms.Compose([
            transforms.Resize((image_size//4, image_size//4)),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        file_name = str(self.flist[index])
        img = common.pil_loader(file_name)
        hr = self.tfs(img)
        lr = self.tf_down(img)

        ret['y_gt'] = hr
        ret['x_input'] = lr
        # ret['mask'] = None  不可以写None
        ret['path'] = file_name
        return ret

    def __len__(self):
        return len(self.flist)


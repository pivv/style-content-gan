import sys
import os

from typing import List, Callable, Union, Any, TypeVar, Tuple, Dict
from torch import Tensor

import math

import numpy as np
import pandas as pd

import cv2

import yaml

import torch
from torch.utils.data import Dataset, DataLoader

from scgan.util.io import read_image


SEG_CLASS_NUM = 15


class BeautyganDataset(Dataset):
    def __init__(self, root: str = '', dirname: str = 'beautygan',
                 image_size: int = 256, seg_size: int = 64) -> None:
        self._image_size = image_size
        self._seg_size = seg_size
        self._non_makeup_dir: str = os.path.join(root, f"{dirname}/images/non-makeup/")
        self._makeup_dir: str = os.path.join(root, f"{dirname}/images/makeup/")
        self._non_makeup_seg_dir: str = os.path.join(root, f"{dirname}/segs/non-makeup/")
        self._makeup_seg_dir: str = os.path.join(root, f"{dirname}/segs/makeup/")
        self._non_makeup_names: List[str] = os.listdir(self._non_makeup_dir)
        self._makeup_names: List[str] = os.listdir(self._makeup_dir)

    def __getitem__(self, index):
        non_makeup_name: str = np.random.choice(self._non_makeup_names)
        makeup_name: str = np.random.choice(self._makeup_names)
        non_makeup_image: np.ndarray = cv2.imread(os.path.join(self._non_makeup_dir, non_makeup_name))
        makeup_image: np.ndarray = cv2.imread(os.path.join(self._makeup_dir, makeup_name))
        non_makeup_seg: np.ndarray = cv2.imread(os.path.join(self._non_makeup_seg_dir, non_makeup_name), cv2.IMREAD_UNCHANGED)
        makeup_seg: np.ndarray = cv2.imread(os.path.join(self._makeup_seg_dir, makeup_name), cv2.IMREAD_UNCHANGED)

        non_makeup_image = cv2.resize(non_makeup_image, (self._image_size, self._image_size),
                                      interpolation=cv2.INTER_CUBIC)
        makeup_image = cv2.resize(makeup_image, (self._image_size, self._image_size),
                                  interpolation=cv2.INTER_CUBIC)
        non_makeup_seg = cv2.resize(non_makeup_seg, (self._seg_size, self._seg_size),
                                    interpolation=cv2.INTER_NEAREST)
        makeup_seg = cv2.resize(makeup_seg, (self._seg_size, self._seg_size),
                                interpolation=cv2.INTER_NEAREST)

        non_makeup_image = cv2.cvtColor(non_makeup_image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
        makeup_image = cv2.cvtColor(makeup_image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
        #non_makeup_seg = (non_makeup_seg[:, :, np.newaxis] == np.arange(SEG_CLASS_NUM).reshape((1, 1, SEG_CLASS_NUM))
        #                  ).astype(np.float32)
        #makeup_seg = (makeup_seg[:, :, np.newaxis] == np.arange(SEG_CLASS_NUM).reshape((1, 1, SEG_CLASS_NUM))
        #              ).astype(np.float32)

        x1 = torch.FloatTensor(non_makeup_image.transpose((2, 0, 1)))
        #x1 = torch.FloatTensor(np.concatenate([non_makeup_image, non_makeup_seg], axis=-1).transpose((2, 0, 1)))
        x2 = torch.FloatTensor(makeup_image.transpose((2, 0, 1)))
        #x2 = torch.FloatTensor(np.concatenate([makeup_image, makeup_seg], axis=-1).transpose((2, 0, 1)))
        seg1 = torch.LongTensor(non_makeup_seg)
        seg2 = torch.LongTensor(makeup_seg)
        return {'x1': x1, 'x2': x2, 'seg1': seg1, 'seg2': seg2}

    def __len__(self):
        return max(len(self._non_makeup_names), len(self._makeup_names))

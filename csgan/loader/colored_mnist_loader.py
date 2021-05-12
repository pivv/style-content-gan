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


class ColoredMnistDataset(Dataset):
    def __init__(self, data: dict = None, root: str = '',
                 dirname: str = 'colored_mnist', image_size: int = 28,
                 train: bool = True) -> None:
        self._image_size = image_size
        if data is None:
            assert root
            data = dict(np.load(os.path.join(root, dirname, f"{'train' if train else 'test'}.npz")))
        data['gray_image'] = np.stack([np.clip(cv2.resize(img, (self._image_size, self._image_size),
                                                          interpolation=cv2.INTER_CUBIC), 0, 1) for
                                       img in data['gray_image']], axis=0)
        data['colored_image'] = np.stack([np.clip(cv2.resize(img, (self._image_size, self._image_size),
                                                             interpolation=cv2.INTER_CUBIC), 0, 1) for
                                          img in data['colored_image']], axis=0)
        self._data = data
        self._colored_indices = np.arange(len(self))
        np.random.shuffle(self._colored_indices)

    @property
    def data(self):
        return self._data

    def __getitem__(self, index: int) -> dict:
        return {'x1': torch.FloatTensor(self._data['gray_image'][index].transpose((2, 0, 1))),
                'x2': torch.FloatTensor(self._data['colored_image'][self._colored_indices[index]].transpose((2, 0, 1))),
                'y1': torch.LongTensor([self._data['label'][index]])[0],
                'y2': torch.LongTensor([self._data['label'][self._colored_indices[index]]])[0],
                'c1': torch.FloatTensor([0., 0., 0.]),
                'c2': (torch.FloatTensor(list(self._data['color'][self._colored_indices[index]])) / 255. if
                       'color' in self._data else
                       torch.FloatTensor(list(self._data['white_color'][self._colored_indices[index]])) / 255.)}

    def __len__(self) -> int:
        return len(self._data['label'])

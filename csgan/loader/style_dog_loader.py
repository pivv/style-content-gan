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

from csgan.util.io import read_image


class StylingDogDataset(Dataset):
    def __init__(self, root: str = '/data', dirname: str = 'stylingDog',
                    list_txt: str = 'list_dark.txt', image_size: int = 256) -> None:
        self._image_size = image_size
        self._dark_dog_dir: str = os.path.join(root, f"{dirname}/train/")
        self._style_dog_dir: str = self._dark_dog_dir
#        self._dark_dog_names: List[str] = np.loadtxt(os.path.join(root, f'{dirname}/{list_txt}'))
        with open(os.path.join(root, f'{dirname}/{list_txt}'), 'r') as f:
            self._dark_dog_names: List[str] = f.read().splitlines()
        
        all_image_names: str = os.listdir(self._dark_dog_dir)
        self._style_dog_names: List[str] = [elem for elem in all_image_names if elem not in self._dark_dog_names]
        print(f"[DEBUG] dark_dog_names: {len(self._dark_dog_names)}")
        print(f"[DEBUG] style_dog_names: {len(self._style_dog_names)}")

    def __getitem__(self, index):
        dark_dog_name: str = np.random.choice(self._dark_dog_names)
        style_dog_name: str = np.random.choice(self._style_dog_names)
        dark_dog_image: np.ndarray = cv2.imread(os.path.join(self._dark_dog_dir, dark_dog_name))
        style_dog_image: np.ndarray = cv2.imread(os.path.join(self._style_dog_dir, style_dog_name))

        dark_dog_image = cv2.resize(dark_dog_image, (self._image_size, self._image_size),
                                      interpolation=cv2.INTER_CUBIC)
        style_dog_image = cv2.resize(style_dog_image, (self._image_size, self._image_size),
                                  interpolation=cv2.INTER_CUBIC)

        dark_dog_image = cv2.cvtColor(dark_dog_image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
        style_dog_image = cv2.cvtColor(style_dog_image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.

        x1 = torch.FloatTensor(dark_dog_image.transpose((2, 0, 1)))
        x2 = torch.FloatTensor(style_dog_image.transpose((2, 0, 1)))
        
        return {'x1': x1, 'x2': x2}

    def __len__(self):
        return max(len(self._dark_dog_names), len(self._style_dog_names))

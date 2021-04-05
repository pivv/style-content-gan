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
                    list_txt_dark: str = 'list_dark.txt', list_txt_bright: str = "list_bright.txt", 
                    image_size: int = 256) -> None:
        self._image_size = image_size
        self._dark_dog_dir: str = os.path.join(root, f"{dirname}/train/")
        self._bright_dog_dir: str = self._dark_dog_dir
#        self._dark_dog_names: List[str] = np.loadtxt(os.path.join(root, f'{dirname}/{list_txt}'))
        with open(os.path.join(root, f'{dirname}/{list_txt_dark}'), 'r') as f:
            self._dark_dog_names: List[str] = f.read().splitlines()
        
        with open(os.path.join(root, f'{dirname}/{list_txt_bright}'), 'r') as f:
            self._bright_dog_names: List[str] = f.read().splitlines()
        
        print(f"[DEBUG] dark_dog_names: {len(self._dark_dog_names)}")
        print(f"[DEBUG] bright_dog_names: {len(self._bright_dog_names)}")

    def __getitem__(self, index):
        dark_dog_name: str = np.random.choice(self._dark_dog_names)
        bright_dog_name: str = np.random.choice(self._bright_dog_names)
        dark_dog_image: np.ndarray = cv2.imread(os.path.join(self._dark_dog_dir, dark_dog_name))
        bright_dog_image: np.ndarray = cv2.imread(os.path.join(self._bright_dog_dir, bright_dog_name))

        dark_dog_image = cv2.resize(dark_dog_image, (self._image_size, self._image_size),
                                      interpolation=cv2.INTER_CUBIC)
        bright_dog_image = cv2.resize(bright_dog_image, (self._image_size, self._image_size),
                                  interpolation=cv2.INTER_CUBIC)

        dark_dog_image = cv2.cvtColor(dark_dog_image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
        bright_dog_image = cv2.cvtColor(bright_dog_image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.

        x1 = torch.FloatTensor(dark_dog_image.transpose((2, 0, 1)))
        x2 = torch.FloatTensor(bright_dog_image.transpose((2, 0, 1)))
        
        return {'x1': x1, 'x2': x2}

    def __len__(self):
        return max(len(self._dark_dog_names), len(self._bright_dog_names))

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
    def __init__(self, root: str = '/data', dirname: str = 'afhq',
                    list_txt_white: str = 'white_dog.csv', image_size: int = 256) -> None:
        self._image_size = image_size
        self._white_dog_dir: str = os.path.join(root, f"{dirname}/train/dog")
        self._style_dog_dir: str = self._white_dog_dir
        
        with open(os.path.join(root, f'{dirname}/{list_txt_white}'), 'r') as f:
            self._white_dog_names: List[str] = f.read().splitlines()
        
        all_image_names: str = os.listdir(self._white_dog_dir)
        self._style_dog_names: List[str] = [elem for elem in all_image_names if elem not in self._white_dog_names]
        print(f"[DEBUG] white_dog_names: {len(self._white_dog_names)}")
        print(f"[DEBUG] style_dog_names: {len(self._style_dog_names)}")

    def __getitem__(self, index):
        white_dog_name: str = np.random.choice(self._white_dog_names)
        style_dog_name: str = np.random.choice(self._style_dog_names)
        white_dog_image: np.ndarray = cv2.imread(os.path.join(self._white_dog_dir, white_dog_name))
        style_dog_image: np.ndarray = cv2.imread(os.path.join(self._style_dog_dir, style_dog_name))
        
        try:    
            white_dog_image = cv2.resize(white_dog_image, (self._image_size, self._image_size),
                                          interpolation=cv2.INTER_CUBIC)
            style_dog_image = cv2.resize(style_dog_image, (self._image_size, self._image_size),
                                      interpolation=cv2.INTER_CUBIC)
        except: 
            print(f"[DEBUG] {os.path.join(self._white_dog_dir, white_dog_name)}")
            print(f"[DEBUG] {os.path.join(self._style_dog_dir, style_dog_name)}")

        white_dog_image = cv2.cvtColor(white_dog_image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
        style_dog_image = cv2.cvtColor(style_dog_image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.


        x1 = torch.FloatTensor(white_dog_image.transpose((2, 0, 1)))
        x2 = torch.FloatTensor(style_dog_image.transpose((2, 0, 1)))
        
        return {'x1': x1, 'x2': x2}

    def __len__(self):
        return max(len(self._white_dog_names), len(self._style_dog_names))

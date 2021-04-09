import sys
import os

from typing import List, Callable, Union, Any, TypeVar, Tuple, Dict
from torch import Tensor

import numpy as np
import pandas as pd


def color_gray_image(gray_img: np.ndarray,
                     color_white: Tuple[int, int, int] = (255, 255, 255),
                     color_black: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    assert(len(color_white) == len(color_black) == 3)
    if len(gray_img.shape) == 3:
        colored_img: np.ndarray = gray_img.copy()
    else:
        assert(len(gray_img.shape) == 2)
        colored_img = gray_img[:, :, np.newaxis].repeat(3, axis=-1)
    if np.issubdtype(gray_img.dtype, np.integer):  # 0 ~ 255 image
        colored_img /= 255.
    for icolor in range(2, -1, -1):
        colored_img[:, :, icolor] = (colored_img[:, :, 0] * (float(color_white[icolor]) / 255.) +
                                     (1. - colored_img[:, :, 0]) * (float(color_black[icolor]) / 255.))
    if np.issubdtype(gray_img.dtype, np.integer):  # 0 ~ 255 image
        colored_img = np.round(colored_img * 255.).astype(int)
    return colored_img

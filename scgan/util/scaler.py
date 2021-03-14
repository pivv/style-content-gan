import sys
import os

from typing import List, Callable, Union, Any, TypeVar, Tuple, Dict
from torch import Tensor

import math

import numpy as np
import pandas as pd


class Scaler(object):
    def __init__(self, scale: float = 0., bias: float = 0.) -> None:
        self.scale: float = scale
        self.bias: float = bias

    def scaling(self, input: Tensor) -> Tensor:
        return (input - self.bias) * self.scale

    def unscaling(self, input: Tensor) -> Tensor:
        return input / self.scale + self.bias

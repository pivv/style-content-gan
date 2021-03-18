import sys
import os

from typing import List, Callable, Union, Any, TypeVar, Tuple, Dict
from torch import Tensor

import numpy as np
import pandas as pd


def acquire_batches(n: int, bs: int, shuffle: bool = True, remove_last: bool = True) -> List[np.ndarray]:
    indices = np.arange(n)
    if shuffle:
        np.random.shuffle(indices)
    if remove_last:
        return [indices[i*bs:(i+1)*bs] for i in range(n//bs)]
    else:
        return [indices[i*bs:min((i+1)*bs, n)] for i in range((n-1)//bs + 1)]

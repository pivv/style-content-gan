import sys
import os

from typing import List, Callable, Union, Any, TypeVar, Tuple, Dict
from torch import Tensor

from abc import abstractmethod

import time

from collections import defaultdict

import numpy as np

import cv2

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader

from tensorboardX import SummaryWriter

from .base_model import BaseModel


class ScganModel(BaseModel):
    def __init__(self, device, encoder: nn.Module, decoder: nn.Module) -> None:
        super().__init__(device)
        self._encoder = encoder
        self._decoder = decoder
        self.to(self._device)

    def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        self.eval()
        test_batch_size = 512
        outputs = []
        with torch.no_grad():
            for ibatch in range((len(x) - 1) // test_batch_size + 1):
                x_batch = torch.FloatTensor(x[test_batch_size*ibatch:test_batch_size*(ibatch+1)]).to(self.device)
                outputs.append(torch.argmax(self.forward(x_batch), dim=1).detach().cpu().numpy())
        outputs = np.concatenate(outputs, axis=0)
        outputs = (outputs + 1).astype(int)
        assert(len(outputs) == len(x))
        assert(len(outputs.shape) == 1)
        return outputs

    def _update_optimizers(self, loss_dict: Dict[str, Tensor], params: Dict[str, Any],
                           global_step: int = 0) -> None:
        for optimizer, loss_str in self._optimizers:
            if loss_str in loss_dict:
                optimizer.zero_grad()
                loss_dict[loss_str].backward()
                if 'clip_size' in params:
                    clip_grad_norm_(self.parameters(), params['clip_size'])
                optimizer.step()

    def set_optimizers(self, params: Dict[str, Any]) -> None:
        self._optimizers = []
        self._schedulers = []
        optimizer = torch.optim.Adam(self.parameters(), params['learning_rate'])
        self._optimizers.append((optimizer, 'loss'))
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer,
                                                     gamma=params['scheduler_gamma'])
        self._schedulers.append(scheduler)

    @abstractmethod
    def forward(self, batch: Dict[str, Tensor], **kwargs) -> Dict[str, Tensor]:
        pass

    @abstractmethod
    def loss_function(self, batch: Dict[str, Tensor], output: Dict[str, Tensor], **kwargs) -> Dict[str, Tensor]:
        pass

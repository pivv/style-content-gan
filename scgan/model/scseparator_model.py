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
from scgan.util.scaler import Scaler


class SCSeparatorModel(BaseModel):
    def __init__(self, device, encoder: nn.Module, decoder: nn.Module, style_fc: nn.Module,
                 style_discriminator: nn.Module, scaler: Scaler = None) -> None:
        super().__init__(device)
        self._encoder = encoder
        self._decoder = decoder
        self._style_fc = style_fc
        self._style_discriminator = style_discriminator
        if scaler is None:
            scaler = Scaler(1., 0.)
        self._scaler = scaler
        self.to(self._device)

    def _update_optimizers(self, loss_dict: Dict[str, Tensor], params: Dict[str, Any],
                           global_step: int = 0) -> None:
        for optimizer, loss_str in self._optimizers:
            if loss_str in loss_dict:
                optimizer.zero_grad()
                loss_dict[loss_str].backward()
                if 'clip_size' in params:
                    clip_grad_norm_(self.parameters(), params['clip_size'])
                optimizer.step()

    def _update_schedulers(self, params: Dict[str, Any], global_step: int = 0) -> None:
        if global_step % params['scheduler_interval'] == 0:
            for scheduler in self._schedulers:
                scheduler.step()

    def _set_optimizers(self, params: Dict[str, Any]) -> None:
        self._optimizers = []
        self._schedulers = []
        optimizer = torch.optim.Adam(self.parameters(), params['learning_rate'])
        self._optimizers.append((optimizer, 'loss'))
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer,
                                                     gamma=params['scheduler_gamma'])
        self._schedulers.append(scheduler)

    def _predict(self, batch: Dict[str, Tensor], **kwargs) -> Dict[str, Tensor]:
        if 'x2' not in batch.keys():  # Style-content separate
            xp: Tensor = self._scaler.scaling(batch['x']) if 'x' in batch.keys() else self._scaler.scaling(batch['x1'])
            z, s, c = self._style_content_separate(xp)
            output: Dict[str, Tensor] = {'z': z, 's': s, 'c': c} if 'x' in batch.keys() else {'z1': z, 's1': s, 'c1': c}
        else:
            assert('x1' in batch.keys() and 'x2' in batch.keys())  # Style change
            xp1: Tensor = self._scaler.scaling(batch['x1'])
            xp2: Tensor = self._scaler.scaling(batch['x2'])
            z1, s1, c1 = self._style_content_separate(xp1)
            z2, s2, c2 = self._style_content_separate(xp2)
            x12: Tensor = self._scaler.unscaling(self._decoder(c1 + s2))
            x21: Tensor = self._scaler.unscaling(self._decoder(c2 + s1))
            output: Dict[str, Tensor] = {'z1': z1, 'z2': z2, 's1': s1, 's2': s2, 'c1': c1, 'c2': c2,
                                         'x12': x12, 'x21': x21}
        return output

    def _style_content_separate(self, xp: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        z: Tensor = self._encoder(xp)  # Latent
        s: Tensor = self._style_fc(z)  # Style
        assert(s.size() == z.size())
        c: Tensor = z - s  # Content
        return z, s, c

    def forward(self, batch: Dict[str, Tensor], params: Dict[str, Any],
                global_step: int = 0, **kwargs) -> Dict[str, Tensor]:
        xp1: Tensor = self._scaler.scaling(batch['x1'])
        xp2: Tensor = self._scaler.scaling(batch['x2'])
        z1, s1, c1 = self._style_content_separate(xp1)
        z2, s2, c2 = self._style_content_separate(xp2)

        # Bias Loss
        b1: Tensor = self._style_discriminator(c1)  # Bias 1
        b2: Tensor = self._style_discriminator(c1)  # Bias 2

        # Identity Loss
        #xp1_idt: Tensor = self._decoder(z1)
        xp1_idt: Tensor = self._decoder(c1)  # Instead using latent, using content only.
        xp2_idt: Tensor = self._decoder(z2)

        output: Dict[str, Tensor] = {'xp1': xp1, 'xp2': xp2,
                                     'z1': z1, 'z2': z2, 's1': s1, 's2': s2, 'c1': c1, 'c2': c2,
                                     'b1': b1, 'b2': b2, 'xp1_idt': xp1_idt, 'xp2_idt': xp2_idt}
        return output

    def loss_function(self, batch: Dict[str, Tensor], output: Dict[str, Tensor], params: Dict[str, Any],
                      global_step: int = 0, **kwargs) -> Dict[str, Tensor]:
        # 0. Parameters
        lambda_idt, lambda_bias_ent, params['identity_lambda']

        # 1. Identity Loss
        xp1: Tensor = output['xp1']
        xp2: Tensor = output['xp2']
        xp1_idt: Tensor = output['xp1_idt']
        xp2_idt: Tensor = output['xp2_idt']
        assert(xp1.size() == xp1_idt.size() and xp2.size() == xp2_idt.size())

        # 2. Bias-Entropy Loss

        # 3. Bias-Adversarial Loss

        # 4.

        loss_dict: Dict[str, Tensor] = {'loss': loss}
        return loss_dict

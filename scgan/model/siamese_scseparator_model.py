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

import matplotlib.pyplot as plt

from .base_model import BaseModel
from scgan.util.scaler import Scaler
from scgan.deep.loss import SiameseLoss, BinaryEntropyWithLogitsLoss
from scgan.deep.layer import View
from scgan.deep.grad import grad_reverse
from scgan.deep.resnet import weights_init, simple_resnet, simple_bottleneck_resnet


class SCSeparatorModel(BaseModel):
    def __init__(self, device, encoder: nn.Module, decoder: nn.Module, style_w: nn.Module,
                 bias_discriminator: nn.Module, scaler: Scaler = None) -> None:
        super().__init__(device)
        self._encoder = encoder
        self._decoder = decoder
        self._style_w = style_w
        self._bias_discriminator = bias_discriminator
        if scaler is None:
            scaler = Scaler(1., 0.)
        self._scaler = scaler

        self._identity_criterion = nn.MSELoss()
        self._entropy_criterion = BinaryEntropyWithLogitsLoss()
        self._style_criterion1 = nn.MSELoss()
        self._style_criterion2 = SiameseLoss()
        self._bias_criterion = nn.BCEWithLogitsLoss()
        self._weight_criterion = nn.MSELoss()

        self.to(self._device)

    def _update_optimizers(self, loss_dict: Dict[str, Tensor], params: Dict[str, Any],
                           global_step: int = 0) -> None:
        optimizer_enc, optimizer_dec, optimizer_w, optimizer_bias = self._optimizers

        loss_idt: Tensor = loss_dict['loss_idt']
        loss_entropy: Tensor = loss_dict['loss_idt']
        loss_style: Tensor = loss_dict['loss_style']
        loss_bias: Tensor = loss_dict['loss_bias']
        loss_weight: Tensor = loss_dict['loss_weight']

        optimizer_enc.zero_grad()
        optimizer_dec.zero_grad()
        optimizer_w.zero_grad()
        optimizer_bias.zero_grad()
        #(loss_style + loss_weight).backward(retain_graph=True)
        optimizer_enc.zero_grad()
        #loss_idt.backward(retain_graph=True)
        #(loss_idt + loss_entropy).backward(retain_graph=True)
        if global_step < 100 or global_step % 5 != 0:
            (loss_idt + loss_weight + loss_bias).backward(retain_graph=True)
        else:
            (loss_idt + loss_style + loss_weight + loss_bias).backward(retain_graph=True)
        #(loss_idt + loss_style + loss_weight).backward(retain_graph=True)
        #(loss_idt + loss_entropy + loss_style + loss_weight).backward(retain_graph=True)
        #optimizer_bias.zero_grad()
        #loss_bias.backward()
        if 'clip_size' in params:
            clip_grad_norm_(self._encoder.parameters(), params['clip_size'])
            clip_grad_norm_(self._decoder.parameters(), params['clip_size'])
            clip_grad_norm_(self._bias_discriminator.parameters(), params['clip_size'])
            clip_grad_norm_(self._style_w.parameters(), params['clip_size'])
        optimizer_enc.step()
        optimizer_dec.step()
        optimizer_bias.step()
        optimizer_w.step()

    def _update_schedulers(self, params: Dict[str, Any], global_step: int = 0) -> None:
        if global_step % params['scheduler_interval'] == 0:
            for scheduler in self._schedulers:
                scheduler.step()

        if global_step == 1 or global_step % params['scheduler_interval'] == 0:
            lr = self._schedulers[0].get_last_lr()
            print("")
            print(f"Learning with learning rate: {lr[0]: .8f}.")
            print("")

    def _set_optimizers(self, params: Dict[str, Any]) -> None:
        optimizer_enc = optim.Adam(self._encoder.parameters(), params['learning_rate'])
        optimizer_dec = optim.Adam(self._decoder.parameters(), params['learning_rate'])
        optimizer_w = optim.Adam(self._style_w.parameters(), params['learning_rate'])
        optimizer_bias = optim.Adam(self._bias_discriminator.parameters(), params['learning_rate'])

        scheduler_enc = optim.lr_scheduler.ExponentialLR(optimizer_enc, gamma=params['scheduler_gamma'])
        scheduler_dec = optim.lr_scheduler.ExponentialLR(optimizer_dec, gamma=params['scheduler_gamma'])
        scheduler_w = optim.lr_scheduler.ExponentialLR(optimizer_w, gamma=params['scheduler_gamma'])
        scheduler_bias = optim.lr_scheduler.ExponentialLR(optimizer_bias, gamma=params['scheduler_gamma'])

        self._optimizers = [optimizer_enc, optimizer_dec, optimizer_w, optimizer_bias]
        self._schedulers = [scheduler_enc, scheduler_dec, scheduler_w, scheduler_bias]

    def _predict(self, batch: Dict[str, Tensor], **kwargs) -> Dict[str, Tensor]:
        if 'x' in batch.keys():  # Style-content separate
            xp: Tensor = self._scaler.scaling(batch['x'])
            z, s, c = self._style_content_separate(xp)
            output: Dict[str, Tensor] = {'z': z, 's': s, 'c': c}
        else:
            assert('x1' in batch.keys() and 'x2' in batch.keys())  # Style change
            xp1: Tensor = self._scaler.scaling(batch['x1'])
            xp2: Tensor = self._scaler.scaling(batch['x2'])
            z1, s1, c1 = self._style_content_separate(xp1)
            z2, s2, c2 = self._style_content_separate(xp2)
            x1_idt: Tensor = self._scaler.unscaling(self._decoder(c1))
            x2_idt: Tensor = self._scaler.unscaling(self._decoder(c2 + s2))
            x12: Tensor = self._scaler.unscaling(self._decoder(c1 + s2))
            x21: Tensor = self._scaler.unscaling(self._decoder(c2))
            output: Dict[str, Tensor] = {'z1': z1, 'z2': z2, 's1': s1, 's2': s2, 'c1': c1, 'c2': c2,
                                         'x1_idt': x1_idt, 'x2_idt': x2_idt, 'x12': x12, 'x21': x21}
        return output

    def _style_content_separate(self, xp: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        z: Tensor = self._encoder(xp)  # Latent
        s: Tensor = self._style_w(z)  # Style
        assert(s.size() == z.size())
        c: Tensor = z - s  # Content
        return z, s, c

    def _post_processing(self, batch: Dict[str, Tensor], params: Dict[str, Any],
                         global_step: int = 0) -> None:
        if global_step == 1 or global_step % params['sampling_interval'] == 0:
            output: Dict[str, Tensor] = self._predict(batch)
            result_dir = os.path.join(params['run_dir'], 'result')

            fig = plt.figure(figsize=(15, 20))
            for index in range(8):
                ax = fig.add_subplot(8, 6, 6*index+1)
                ax.imshow(batch['x1'][index].detach().cpu().numpy().transpose(1, 2, 0))
                ax = fig.add_subplot(8, 6, 6*index+2)
                ax.imshow(batch['x2'][index].detach().cpu().numpy().transpose(1, 2, 0))
                ax = fig.add_subplot(8, 6, 6*index+3)
                ax.imshow(output['x1_idt'][index].detach().cpu().numpy().transpose(1, 2, 0))
                ax = fig.add_subplot(8, 6, 6*index+4)
                ax.imshow(output['x2_idt'][index].detach().cpu().numpy().transpose(1, 2, 0))
                ax = fig.add_subplot(8, 6, 6*index+5)
                ax.imshow(output['x12'][index].detach().cpu().numpy().transpose(1, 2, 0))
                ax = fig.add_subplot(8, 6, 6*index+6)
                ax.imshow(output['x21'][index].detach().cpu().numpy().transpose(1, 2, 0))

            plt.savefig(os.path.join(result_dir, f"sampling_{global_step}.png"), dpi=200, bbox_inches='tight')
            plt.close('all')

    def forward(self, batch: Dict[str, Tensor], params: Dict[str, Any],
                global_step: int = 0, **kwargs) -> Dict[str, Tensor]:
        xp1: Tensor = self._scaler.scaling(batch['x1'])
        xp2: Tensor = self._scaler.scaling(batch['x2'])
        z1, s1, c1 = self._style_content_separate(xp1)
        z2, s2, c2 = self._style_content_separate(xp2)

        # Bias Loss
        b1: Tensor = self._bias_discriminator(grad_reverse(c1))  # Bias 1
        b2: Tensor = self._bias_discriminator(grad_reverse(c2))  # Bias 2

        # Identity Loss
        #xp1_idt: Tensor = self._decoder(z1)
        xp1_idt: Tensor = self._decoder(c1)  # Instead using latent, using content only.
        xp2_idt: Tensor = self._decoder(z2)

        # Weight Cycle Loss
        z12: Tensor = c1 + s2
        s2_idt: Tensor = self._style_w(z12)
        c1_idt: Tensor = z12 - s2_idt
        z21: Tensor = c2 + s1
        s1_idt: Tensor = self._style_w(z21)
        c2_idt: Tensor = z21 - s1_idt

        output: Dict[str, Tensor] = {'xp1': xp1, 'xp2': xp2,
                                     'z1': z1, 'z2': z2, 's1': s1, 's2': s2, 'c1': c1, 'c2': c2,
                                     'b1': b1, 'b2': b2, 'xp1_idt': xp1_idt, 'xp2_idt': xp2_idt,
                                     'c1_idt': c1_idt, 'c2_idt': c2_idt, 's1_idt': s1_idt, 's2_idt': s2_idt}
        return output

    def loss_function(self, batch: Dict[str, Tensor], output: Dict[str, Tensor], params: Dict[str, Any],
                      global_step: int = 0, **kwargs) -> Dict[str, Tensor]:
        # 0. Parameters
        lambda_idt: float = params['lambda_idt']
        lambda_entropy: float = params['lambda_entropy']
        lambda_style: float = params['lambda_style']
        margin_style: float = params['margin_style']
        lambda_bias: float = params['lambda_bias']
        lambda_weight: float = params['lambda_weight']

        # 1. Identity Loss
        xp1: Tensor = output['xp1']
        xp2: Tensor = output['xp2']
        xp1_idt: Tensor = output['xp1_idt']
        xp2_idt: Tensor = output['xp2_idt']
        assert(xp1.size() == xp1_idt.size() and xp2.size() == xp2_idt.size())
        loss_idt: Tensor = lambda_idt * (self._identity_criterion(xp1_idt, xp1) +
                                         self._identity_criterion(xp2_idt, xp2)) / 2.

        # 2. Entropy Loss
        b1: Tensor = output['b1']
        b2: Tensor = output['b2']
        loss_entropy: Tensor = lambda_entropy * (self._entropy_criterion(b1) +
                                                 self._entropy_criterion(b2)) / 2.

        # 3. Style Loss
        s1: Tensor = output['s1']
        s2: Tensor = output['s2']
        # Siamese-like Loss
        loss_style: Tensor = lambda_style * (self._style_criterion1(s1, torch.zeros_like(s1)) +
                                             self._style_criterion2(s2, margin=margin_style)) / 2.
        norm_s1: Tensor = torch.sqrt((s1 * s1).flatten(start_dim=1).mean(dim=1)).mean()
        norm_s2: Tensor = torch.sqrt((s2 * s2).flatten(start_dim=1).mean(dim=1)).mean()

        # 4. Bias Loss
        loss_bias: Tensor = lambda_bias * (self._bias_criterion(b1, torch.zeros_like(b1)) +
                                           self._bias_criterion(b2, torch.ones_like(b2))) / 2.
        correct_b1: Tensor = b1 < 0
        correct_b2: Tensor = b2 >= 0
        accuracy_bias: Tensor = (correct_b1.sum() + correct_b2.sum()) / float(len(b1) + len(b2))

        # 5. Weight Cycle Loss
        c1: Tensor = output['c1']
        c2: Tensor = output['c2']
        c1_idt: Tensor = output['c1_idt']
        c2_idt: Tensor = output['c2_idt']
        s1_idt: Tensor = output['s1_idt']
        s2_idt: Tensor = output['s2_idt']
        loss_weight: Tensor = lambda_weight * (self._weight_criterion(c1_idt, c1) +
                                               self._weight_criterion(c2_idt, c2) +
                                               self._weight_criterion(s1_idt, s1) +
                                               self._weight_criterion(s2_idt, s2)) / 4.

        #loss: Tensor = loss_idt + loss_entropy + loss_style + loss_bias + loss_weight
        loss: Tensor = loss_idt + loss_style + loss_bias + loss_weight

        loss_dict: Dict[str, Tensor] = {'loss': loss,
                                        'loss_idt': loss_idt, 'loss_entropy': loss_entropy,
                                        'loss_style': loss_style, 'loss_bias': loss_bias,
                                        'loss_weight': loss_weight, 'accuracy_bias': accuracy_bias,
                                        'norm_s1': norm_s1, 'norm_s2': norm_s2}
        return loss_dict


class SCSeparatorResnetModel(SCSeparatorModel):
    def __init__(self, device) -> None:
        dimension = 2
        in_channels = 3
        latent_dim = 1024
        num_blocks = [4]
        planes = [64, 64]

        encoder: nn.Module = nn.Sequential(
            nn.Conv2d(in_channels, planes[0], kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(planes[0]), nn.LeakyReLU(0.01), nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            simple_resnet(dimension, num_blocks, planes,
                          transpose=False, norm='BatchNorm', activation='LeakyReLU', pool=False),
            nn.Flatten(start_dim=1), nn.Linear(planes[-1]*7*7, latent_dim))
        decoder: nn.Module = nn.Sequential(
            nn.Linear(latent_dim, planes[-1]*7*7), View((-1, planes[-1], 7, 7)),
            simple_resnet(dimension, num_blocks, planes,
                          transpose=True, norm='BatchNorm', activation='LeakyReLU', pool=False),
            nn.ConvTranspose2d(planes[0], planes[0], kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(planes[0]), nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(planes[0], in_channels, kernel_size=5, stride=1, padding=2, output_padding=0),
            nn.Tanh())
        style_w: nn.Module = nn.Linear(latent_dim, latent_dim, bias=False)
        bias_discriminator: nn.Module = nn.Sequential(
            nn.Linear(latent_dim, 256, bias=False), nn.BatchNorm1d(256), nn.LeakyReLU(0.01),
            nn.Linear(256, 64, bias=False), nn.BatchNorm1d(64), nn.LeakyReLU(0.01),
            nn.Linear(64, 16, bias=False), nn.BatchNorm1d(16), nn.LeakyReLU(0.01),
            nn.Linear(16, 1), nn.Flatten())
        scaler: Scaler = Scaler(2., 0.5)

        super().__init__(device, encoder, decoder, style_w, bias_discriminator, scaler)
        self.apply(weights_init)

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
from csgan.util.scaler import Scaler
from csgan.deep.loss import SiameseLoss, BinaryEntropyWithLogitsLoss
from csgan.deep.layer import View, Permute
from csgan.deep.grad import grad_scale, grad_reverse
from csgan.deep.resnet import simple_resnet, simple_bottleneck_resnet
from csgan.deep.initialize import weights_init_resnet
from csgan.deep.norm import spectral_norm


class CSDoubleEncoderModel(BaseModel):
    def __init__(self, device, content_encoder: nn.Module, style_encoder: nn.Module, decoder: nn.Module,
                 content_disc: nn.Module, style_disc: nn.Module, scaler: Scaler = None) -> None:
        super().__init__(device)
        self._content_encoder = content_encoder
        self._style_encoder = style_encoder
        self._decoder = decoder
        self._content_disc = content_disc
        self._style_disc = style_disc
        if scaler is None:
            scaler = Scaler(1., 0.)
        self._scaler = scaler

        self._identity_criterion = nn.L1Loss()
        self._cycle_criterion = nn.L1Loss()
        self._content_criterion = nn.MSELoss()
        self._style_criterion = nn.MSELoss()
        #self._siamese_criterion = SiameseLoss()

        self.to(self._device)

    def _update_optimizers(self, loss_dict: Dict[str, Tensor], params: Dict[str, Any],
                           global_step: int = 0) -> None:
        pass

    def _update_schedulers(self, params: Dict[str, Any], global_step: int = 0) -> None:
        pass

    def _set_optimizers(self, params: Dict[str, Any]) -> None:
        lr: float = params['learning_rate']
        beta1: float = params['beta1']
        beta2: float = params['beta2']
        weight_decay: float = params['weight_decay']
        gamma: float = params['scheduler_gamma']
        self._optimizers = [optim.Adam(module.parameters(), lr, (beta1, beta2), weight_decay=weight_decay) for
                            module in [self._content_encoder, self._style_encoder, self._decoder,
                                       self._content_disc, self._style_disc]]
        self._schedulers = [optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma) for
                            optimizer in self._optimizers]

    @abstractmethod
    def _cs_to_latent(self, c: Tensor, s: Tensor = None) -> Tensor:
        pass

    def _predict(self, batch: Dict[str, Tensor], **kwargs) -> Dict[str, Tensor]:
        if 'x' in batch.keys():  # Style-content separate
            xp: Tensor = self._scaler.scaling(batch['x'])
            c: Tensor = self._content_encoder(xp)
            s: Tensor = self._style_encoder(xp)
            z: Tensor = self._cs_to_latent(c, s)
            output: Dict[str, Tensor] = {'z': z, 's': s, 'c': c}
        else:
            assert('x1' in batch.keys() and 'x2' in batch.keys())  # Style change
            xp1: Tensor = self._scaler.scaling(batch['x1'])
            xp2: Tensor = self._scaler.scaling(batch['x2'])
            c1: Tensor = self._content_encoder(xp1)
            s1: Tensor = self._style_encoder(xp1)
            c2: Tensor = self._content_encoder(xp2)
            s2: Tensor = self._style_encoder(xp2)
            x1_idt: Tensor = self._scaler.unscaling(self._cs_to_latent(c1, s1))
            x1_idt2: Tensor = self._scaler.unscaling(self._cs_to_latent(c1))
            x2_idt: Tensor = self._scaler.unscaling(self._cs_to_latent(c2, s2))
            x12: Tensor = self._scaler.unscaling(self._cs_to_latent(c1, s2))
            x21: Tensor = self._scaler.unscaling(self._cs_to_latent(c2, s1))
            x21_2: Tensor = self._scaler.unscaling(self._cs_to_latent(c2))
            output: Dict[str, Tensor] = {'s1': s1, 's2': s2, 'c1': c1, 'c2': c2,
                                         'x1_idt': x1_idt, 'x1_idt2': x1_idt2,
                                         'x2_idt': x2_idt, 'x12': x12, 'x21': x21, 'x21_2': x21_2}
        return output

    def _post_processing(self, batch: Dict[str, Tensor], params: Dict[str, Any],
                         global_step: int = 0) -> None:
        if global_step == 1 or global_step % params['sampling_interval'] == 0:
            output: Dict[str, Tensor] = self._predict(batch)
            result_dir = os.path.join(params['run_dir'], 'results')

            n: int = 6
            fig = plt.figure(figsize=(20, (20 * n) // 8))
            for index in range(n):
                ax = fig.add_subplot(n, 8, 8*index+1)
                ax.imshow(batch['x1'][index, :3].detach().cpu().numpy().transpose(1, 2, 0))
                ax = fig.add_subplot(n, 8, 8*index+2)
                ax.imshow(batch['x2'][index, :3].detach().cpu().numpy().transpose(1, 2, 0))
                ax = fig.add_subplot(n, 8, 8*index+3)
                ax.imshow(output['x1_idt'][index].detach().cpu().numpy().transpose(1, 2, 0))
                ax = fig.add_subplot(n, 8, 8*index+4)
                ax.imshow(output['x1_idt2'][index].detach().cpu().numpy().transpose(1, 2, 0))
                ax = fig.add_subplot(n, 8, 8*index+5)
                ax.imshow(output['x2_idt'][index].detach().cpu().numpy().transpose(1, 2, 0))
                ax = fig.add_subplot(n, 8, 8*index+6)
                ax.imshow(output['x12'][index].detach().cpu().numpy().transpose(1, 2, 0))
                ax = fig.add_subplot(n, 8, 8*index+7)
                ax.imshow(output['x21'][index].detach().cpu().numpy().transpose(1, 2, 0))
                ax = fig.add_subplot(n, 8, 8*index+8)
                ax.imshow(output['x21_2'][index].detach().cpu().numpy().transpose(1, 2, 0))

            plt.savefig(os.path.join(result_dir, f"sampling_{global_step}.png"), dpi=200, bbox_inches='tight')
            plt.close('all')

    def forward(self, batch: Dict[str, Tensor], params: Dict[str, Any],
                global_step: int = 0, **kwargs) -> Dict[str, Tensor]:
        output: Dict[str, Tensor] = {}
        return output

    def loss_function(self, batch: Dict[str, Tensor], output: Dict[str, Tensor], params: Dict[str, Any],
                      global_step: int = 0, **kwargs) -> Dict[str, Tensor]:
        # 0. Parameters
        lambda_idt: float = params['lambda_identity']
        lambda_cycle: float = params['lambda_cycle']
        lambda_content: float = params['lambda_content']
        lambda_style: float = params['lambda_style']
        lambda_siamese: float = params['lambda_siamese']

        gamma_content = params['gamma_content']
        gamma_style = params['gamma_style']

        xp1: Tensor = self._scaler.scaling(batch['x1'])
        xp2: Tensor = self._scaler.scaling(batch['x2'])

        # 1. Content Disc Loss

        with torch.no_grad():
            c1_detach: Tensor = self._content_encoder(xp1).detach()
            c2_detach: Tensor = self._content_encoder(xp2).detach()

        b1_content: Tensor = self._content_disc(c1_detach)
        b2_content: Tensor = self._content_disc(c2_detach)

        loss_content: Tensor = torch.FloatTensor([0.])[0].to(self._device)
        if lambda_content > 0:
            loss_content: Tensor = lambda_content * (b1_content.mean() - b2_content.mean()) / 2.
            #loss_content: Tensor = lambda_content * (torch.sigmoid(b1_content).mean() - torch.sigmoid(b2_content).mean()) / 2.
            #loss_content: Tensor = lambda_content * (
            #        self._content_criterion(b1_content, torch.zeros_like(b1_content)) +
            #        self._content_criterion(b2_content, torch.ones_like(b2_content))) / 2.
        correct1: Tensor = b1_content < 0
        correct2: Tensor = b2_content >= 0
        accuracy_content: Tensor = (correct1.sum() + correct2.sum()) / float(len(b1_content) + len(b2_content))

        # 2. Style Disc Loss

        b1_style: Tensor = self._style_disc(grad_scale(s1, gamma=gamma_style))
        b2_style: Tensor = self._style_disc(grad_scale(s2, gamma=gamma_style))

        loss_style: Tensor = torch.FloatTensor([0.])[0].to(self._device)
        if lambda_style > 0:
            loss_style: Tensor = lambda_style * (b1_style.mean() - b2_style.mean()) / 2.
            #loss_style: Tensor = lambda_style * (torch.sigmoid(b1_style).mean() - torch.sigmoid(b2_style).mean()) / 2.
            #loss_style: Tensor = lambda_style * (
            #        self._style_criterion(b1_style, torch.zeros_like(b1_style)) +
            #        self._style_criterion(b2_style, torch.ones_like(b2_style))) / 2.
        correct1: Tensor = b1_style < 0
        correct2: Tensor = b2_style >= 0
        accuracy_style: Tensor = (correct1.sum() + correct2.sum()) / float(len(b1_style) + len(b2_style))

        self._content_disc.requires_grad_(False)
        self._style_disc.requires_grad_(False)

        # 1. Identity Loss

        c1: Tensor = self._content_encoder(xp1)
        s1: Tensor = self._style_encoder(xp1)
        c2: Tensor = self._content_encoder(xp2)
        s2: Tensor = self._style_encoder(xp2)

        xp1_idt: Tensor = self._decoder(self._cs_to_latent(c1))
        xp1_idt2: Tensor = self._decoder(self._cs_to_latent(c1, s1))
        xp2_idt: Tensor = self._decoder(self._cs_to_latent(c2, s2))

        loss_idt: Tensor = torch.FloatTensor([0.])[0].to(self._device)
        if lambda_idt > 0:
            #loss_idt: Tensor = lambda_idt * (
            #        self._identity_criterion(xp1_idt, xp1[:, :3]) + self._identity_criterion(xp2_idt, xp2[:, :3])) / 2.
            loss_idt: Tensor = lambda_idt * (
                    (self._identity_criterion(xp1_idt, xp1[:, :3]) +
                     self._identity_criterion(xp1_idt2, xp1[:, :3])) / 2. +
                    self._identity_criterion(xp2_idt, xp2[:, :3])) / 2.

        # 2. Cycle Loss

        xp12: Tensor = self._decoder((c1 + s2).detach())
        #xp21: Tensor = self._decoder((c2 + s1).detach())
        xp21: Tensor = self._decoder(c2.detach())
        s12, c12 = self._style_content_separate(self._encoder(xp12))
        s21, c21 = self._style_content_separate(self._encoder(xp21))
        #xp1_cycle: Tensor = self._decoder(c12 + s1)
        xp1_cycle: Tensor = self._decoder(c12)
        xp2_cycle: Tensor = self._decoder(c21 + s2)

        if lambda_cycle == 0:
            loss_cycle: Tensor = torch.FloatTensor([0.])[0].to(self._device)
        else:
            loss_cycle: Tensor = lambda_cycle * (
                    self._cycle_criterion(xp1_cycle, xp1[:, :3]) + self._cycle_criterion(xp2_cycle, xp2[:, :3])) / 2.

        # 5. Siamese Loss

        s1: Tensor = output['s1']
        s2: Tensor = output['s2']
        if lambda_siamese == 0:
            loss_siamese: Tensor = torch.FloatTensor([0.])[0].to(self._device)
        else:
            #loss_siamese: Tensor = lambda_siamese * (s1 * s1).mean()
            loss_siamese: Tensor = lambda_siamese * ((s1 * s1).mean() + self._siamese_criterion(s2, margin=1.)) / 2.
        norm_s1: Tensor = torch.sqrt((s1 * s1).flatten(start_dim=1).mean(dim=1)).mean()
        norm_s2: Tensor = torch.sqrt((s2 * s2).flatten(start_dim=1).mean(dim=1)).mean()

        self._content_disc.requires_grad_(True)
        self._style_disc.requires_grad_(True)

        loss: Tensor = loss_idt + loss_cycle + loss_content + loss_style + loss_siamese

        loss_dict: Dict[str, Tensor] = {'loss': loss,
                                        'loss_identity': loss_idt, 'loss_cycle': loss_cycle,
                                        'loss_content': loss_content, 'accuracy_content': accuracy_content,
                                        'loss_style': loss_style, 'accuracy_style': accuracy_style,
                                        'loss_siamese': loss_siamese,
                                        'norm_s1': norm_s1, 'norm_s2': norm_s2}
        return loss_dict


class CSDoubleEconderMnistModel(CSDoubleEncoderModel):
    def __init__(self, device) -> None:
        dimension = 2
        in_channels = 3
        content_dim = 512
        style_dim = 64
        num_blocks = [4]
        planes = [64, 64]

        content_encoder: nn.Module = nn.Sequential(
            nn.Conv2d(in_channels, planes[0], kernel_size=5, stride=1, padding=2, bias=False),
            nn.InstanceNorm2d(planes[0], affine=True), nn.LeakyReLU(0.01), nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            simple_resnet(dimension, num_blocks, planes,
                          transpose=False, norm='InstanceNorm', activation='LeakyReLU', pool=False),
            nn.Flatten(start_dim=1), nn.Linear(planes[-1]*7*7, content_dim))
        style_encoder: nn.Module = nn.Sequential(
            nn.Conv2d(in_channels, planes[0], kernel_size=5, stride=1, padding=2, bias=False),
            nn.InstanceNorm2d(planes[0], affine=True), nn.LeakyReLU(0.01), nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            simple_resnet(dimension, num_blocks, planes,
                          transpose=False, norm='InstanceNorm', activation='LeakyReLU', pool=False),
            nn.Flatten(start_dim=1), nn.Linear(planes[-1]*7*7, style_dim))
        decoder: nn.Module = nn.Sequential(
            nn.Linear(content_dim + style_dim, planes[-1]*7*7), View((-1, planes[-1], 7, 7)),
            simple_resnet(dimension, num_blocks, planes,
                          transpose=True, norm='InstanceNorm', activation='LeakyReLU', pool=False),
            nn.ConvTranspose2d(planes[0], planes[0], kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.InstanceNorm2d(planes[0], affine=True), nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(planes[0], in_channels, kernel_size=5, stride=1, padding=2, output_padding=0),
            nn.Tanh())
        content_disc: nn.Module = nn.Sequential(
            nn.Linear(content_dim, 256, bias=False), nn.BatchNorm1d(256), nn.LeakyReLU(0.01),
            nn.Linear(256, 64, bias=False), nn.BatchNorm1d(64), nn.LeakyReLU(0.01),
            nn.Linear(64, 1, bias=True))
        style_disc: nn.Module = nn.Sequential(
            #nn.Dropout(p=0.5, inplace=False),
            nn.Linear(style_dim, 256, bias=False), nn.BatchNorm1d(256), nn.LeakyReLU(0.01),
            nn.Linear(256, 64, bias=False), nn.BatchNorm1d(64), nn.LeakyReLU(0.01),
            nn.Linear(64, 1, bias=True))
        scaler: Scaler = Scaler(2., 0.5)

        super().__init__(device, content_encoder, style_encoder, decoder, content_disc, style_disc, scaler)

        self._content_dim = content_dim
        self._style_dim = style_dim
        self.apply(weights_init_resnet)

    def _cs_to_latent(self, c: Tensor, s: Tensor = None) -> Tensor:
        #z: Tensor = c + s if s is not None else c  # Addition
        z: Tensor = (torch.cat([c, s], dim=1) if s is not None else
                     torch.cat([c, torch.zeros((len(c), self._style_dim)).to(self._device)]))  # Channel combination
        return z

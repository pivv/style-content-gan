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
from scgan.deep.layer import View, Permute
from scgan.deep.grad import grad_scale, grad_reverse
from scgan.deep.resnet import weights_init, simple_resnet, simple_bottleneck_resnet
from scgan.deep.norm import spectral_norm


class SCSeparatorModel(BaseModel):
    def __init__(self, device, encoder: nn.Module, decoder: nn.Module, style_w: nn.Module,
                 content_disc: nn.Module, style_disc: nn.Module,
                 scaler: Scaler = None) -> None:
        super().__init__(device)
        self._encoder = encoder
        self._decoder = decoder
        self._style_w = style_w
        self._content_disc = content_disc
        self._style_disc = style_disc
        if scaler is None:
            scaler = Scaler(1., 0.)
        self._scaler = scaler

        self._identity_criterion = nn.MSELoss()
        self._latent_identity_criterion = nn.MSELoss()
        self._weight_cycle_criterion = nn.MSELoss()
        self._content_criterion = nn.BCEWithLogitsLoss()
        self._style_criterion = nn.BCEWithLogitsLoss()
        #self._siamese_criterion = SiameseLoss()

        self.to(self._device)

    def _update_optimizers(self, loss_dict: Dict[str, Tensor], params: Dict[str, Any],
                           global_step: int = 0) -> None:
        #(optimizer_enc, optimizer_dec, optimizer_w, optimizer_content, optimizer_style) = self._optimizers

        loss: Tensor = loss_dict['loss']
        #loss_idt: Tensor = loss_dict['loss_idt']
        #loss_weight_cycle: Tensor = loss_dict['loss_weight_cycle']
        #loss_content: Tensor = loss_dict['loss_content']
        #loss_style: Tensor = loss_dict['loss_style']
        #loss_siamese: Tensor = loss_dict['loss_siamese']

        for optimizer in self._optimizers:
            optimizer.zero_grad()

        loss.backward()
        #(loss_idt + loss_weight_cycle + loss_content + loss_style + loss_siamese).backward(retain_graph=True)
        if 'clip_size' in params:
            clip_grad_norm_(self.parameters(), params['clip_size'])

        for optimizer in self._optimizers:
            optimizer.step()

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
        weight_decay: float = params['weight_decay']
        optimizer_enc = optim.Adam(self._encoder.parameters(), params['learning_rate'], weight_decay=weight_decay)
        optimizer_dec = optim.Adam(self._decoder.parameters(), params['learning_rate'], weight_decay=weight_decay)
        optimizer_w = optim.Adam(self._style_w.parameters(), params['learning_rate'], weight_decay=weight_decay)
        optimizer_content = optim.Adam(self._content_disc.parameters(), params['learning_rate'], weight_decay=weight_decay)
        optimizer_style = optim.Adam(self._style_disc.parameters(), params['learning_rate'], weight_decay=weight_decay)

        self._optimizers = [optimizer_enc, optimizer_dec, optimizer_w, optimizer_content, optimizer_style]
        self._schedulers = [optim.lr_scheduler.ExponentialLR(optimizer, gamma=params['scheduler_gamma']) for
                            optimizer in self._optimizers]

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
            x1_idt: Tensor = self._scaler.unscaling(self._decoder(c1 + s1))
            x1_idt2: Tensor = self._scaler.unscaling(self._decoder(c1))
            x2_idt: Tensor = self._scaler.unscaling(self._decoder(c2 + s2))
            x12: Tensor = self._scaler.unscaling(self._decoder(c1 + s2))
            x21: Tensor = self._scaler.unscaling(self._decoder(c2 + s1))
            x21_2: Tensor = self._scaler.unscaling(self._decoder(c2))
            output: Dict[str, Tensor] = {'z1': z1, 'z2': z2, 's1': s1, 's2': s2, 'c1': c1, 'c2': c2,
                                         'x1_idt': x1_idt, 'x1_idt2': x1_idt2,
                                         'x2_idt': x2_idt, 'x12': x12, 'x21': x21, 'x21_2': x21_2}
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
            result_dir = os.path.join(params['run_dir'], 'results')

            fig = plt.figure(figsize=(20, 20))
            for index in range(8):
                ax = fig.add_subplot(8, 8, 8*index+1)
                ax.imshow(batch['x1'][index, :3].detach().cpu().numpy().transpose(1, 2, 0))
                ax = fig.add_subplot(8, 8, 8*index+2)
                ax.imshow(batch['x2'][index, :3].detach().cpu().numpy().transpose(1, 2, 0))
                ax = fig.add_subplot(8, 8, 8*index+3)
                ax.imshow(output['x1_idt'][index].detach().cpu().numpy().transpose(1, 2, 0))
                ax = fig.add_subplot(8, 8, 8*index+4)
                ax.imshow(output['x1_idt2'][index].detach().cpu().numpy().transpose(1, 2, 0))
                ax = fig.add_subplot(8, 8, 8*index+5)
                ax.imshow(output['x2_idt'][index].detach().cpu().numpy().transpose(1, 2, 0))
                ax = fig.add_subplot(8, 8, 8*index+6)
                ax.imshow(output['x12'][index].detach().cpu().numpy().transpose(1, 2, 0))
                ax = fig.add_subplot(8, 8, 8*index+7)
                ax.imshow(output['x21'][index].detach().cpu().numpy().transpose(1, 2, 0))
                ax = fig.add_subplot(8, 8, 8*index+8)
                ax.imshow(output['x21_2'][index].detach().cpu().numpy().transpose(1, 2, 0))

            plt.savefig(os.path.join(result_dir, f"sampling_{global_step}.png"), dpi=200, bbox_inches='tight')
            plt.close('all')

    def forward(self, batch: Dict[str, Tensor], params: Dict[str, Any],
                global_step: int = 0, **kwargs) -> Dict[str, Tensor]:
        gamma_content = params['gamma_content']
        gamma_style = params['gamma_style']

        xp1: Tensor = self._scaler.scaling(batch['x1'])
        xp2: Tensor = self._scaler.scaling(batch['x2'])
        z1, s1, c1 = self._style_content_separate(xp1)
        z2, s2, c2 = self._style_content_separate(xp2)

        # Identity Loss
        xp1_idt: Tensor = self._decoder(z1)
        #xp1_idt2: Tensor = self._decoder(c1)  # Instead using latent, using content only.
        xp2_idt: Tensor = self._decoder(z2)

        # Latent Identity Loss
        z1_detach: Tensor = z1.detach()
        z2_detach: Tensor = z2.detach()
        z1_idt: Tensor = self._encoder(self._decoder(z1_detach))  # Latent
        z2_idt: Tensor = self._encoder(self._decoder(z2_detach))  # Latent

        # Weight Cycle Loss
        s1_detach: Tensor = self._style_w(z1_detach)
        c1_detach: Tensor = z1_detach - s1_detach
        s2_detach: Tensor = self._style_w(z2_detach)
        c2_detach: Tensor = z2_detach - s2_detach
        z12: Tensor = (c1_detach + s2_detach)
        #z12: Tensor = (c1 + s2)
        s2_idt: Tensor = self._style_w(z12)
        c1_idt: Tensor = z12 - s2_idt
        z21: Tensor = (c2_detach + s1_detach)
        #z21: Tensor = (c2 + s1)
        s1_idt: Tensor = self._style_w(z21)
        c2_idt: Tensor = z21 - s1_idt

        # Content Disc Loss
        b1_content: Tensor = self._content_disc(c1.detach())
        b2_content: Tensor = self._content_disc(grad_reverse(c2, gamma=gamma_content))

        # Style Disc Loss
        b1_style: Tensor = self._style_disc(grad_scale(s1, gamma=gamma_style))
        b2_style: Tensor = self._style_disc(grad_scale(s2, gamma=gamma_style))

        output: Dict[str, Tensor] = {'xp1': xp1, 'xp2': xp2,
                                     'z1': z1, 'z2': z2, 's1': s1, 's2': s2, 'c1': c1, 'c2': c2,
                                     'xp1_idt': xp1_idt, 'xp2_idt': xp2_idt, #'xp1_idt2': xp1_idt2,
                                     'z1_detach': z1_detach, 'z2_detach': z2_detach, 'z1_idt': z1_idt, 'z2_idt': z2_idt,
                                     'c1_detach': c1_detach, 'c2_detach': c2_detach,
                                     's1_detach': s1_detach, 's2_detach': s2_detach,
                                     'c1_idt': c1_idt, 'c2_idt': c2_idt, 's1_idt': s1_idt, 's2_idt': s2_idt,
                                     'b1_content': b1_content, 'b2_content': b2_content,
                                     'b1_style': b1_style, 'b2_style': b2_style}
        return output

    def loss_function(self, batch: Dict[str, Tensor], output: Dict[str, Tensor], params: Dict[str, Any],
                      global_step: int = 0, **kwargs) -> Dict[str, Tensor]:
        # 0. Parameters
        lambda_idt: float = params['lambda_identity']
        lambda_latent_idt: float = params['lambda_latent_identity']
        lambda_weight_cycle: float = params['lambda_weight_cycle']
        lambda_content: float = params['lambda_content']
        lambda_style: float = params['lambda_style']
        lambda_siamese: float = params['lambda_siamese']

        # 1. Identity Loss
        xp1: Tensor = output['xp1']
        xp2: Tensor = output['xp2']
        xp1_idt: Tensor = output['xp1_idt']
        #xp1_idt2: Tensor = output['xp1_idt2']
        xp2_idt: Tensor = output['xp2_idt']
        assert(xp1[:, :3].size() == xp1_idt.size() and xp2[:, :3].size() == xp2_idt.size())
        loss_idt: Tensor = lambda_idt * (
                self._identity_criterion(xp1_idt, xp1[:, :3]) + self._identity_criterion(xp2_idt, xp2[:, :3])) / 2.
        #loss_idt: Tensor = lambda_idt * (
        #        (self._identity_criterion(xp1_idt, xp1[:, :3]) + self._identity_criterion(xp1_idt2, xp1[:, :3])) / 2. +
        #        self._identity_criterion(xp2_idt, xp2[:, :3])) / 2.

        # 2. Latent Identity Loss
        z1_detach: Tensor = output['z1_detach']
        z2_detach: Tensor = output['z2_detach']
        z1_idt: Tensor = output['z1_idt']
        z2_idt: Tensor = output['z2_idt']
        loss_latent_idt: Tensor = lambda_latent_idt * (
                self._latent_identity_criterion(z1_idt, z1_detach) +
                self._latent_identity_criterion(z2_idt, z2_detach)) / 2.

        # 2. Weight Cycle Loss
        c1_detach: Tensor = output['c1_detach']
        c2_detach: Tensor = output['c2_detach']
        s1_detach: Tensor = output['s1_detach']
        s2_detach: Tensor = output['s2_detach']
        c1_idt: Tensor = output['c1_idt']
        c2_idt: Tensor = output['c2_idt']
        s1_idt: Tensor = output['s1_idt']
        s2_idt: Tensor = output['s2_idt']
        loss_weight_cycle: Tensor = lambda_weight_cycle * (
                self._weight_cycle_criterion(c1_idt, c1_detach) + self._weight_cycle_criterion(c2_idt, c2_detach) +
                self._weight_cycle_criterion(s1_idt, s1_detach) + self._weight_cycle_criterion(s2_idt, s2_detach)) / 4.

        # 3. Content Disc Loss
        b1_content: Tensor = output['b1_content']
        b2_content: Tensor = output['b2_content']
        loss_content: Tensor = lambda_content * (
                self._content_criterion(b1_content, torch.zeros_like(b1_content)) +
                self._content_criterion(b2_content, torch.ones_like(b2_content))) / 2.
        correct1: Tensor = b1_content < 0
        correct2: Tensor = b2_content >= 0
        accuracy_content: Tensor = (correct1.sum() + correct2.sum()) / float(len(b1_content) + len(b2_content))

        # 4. Style Disc Loss
        b1_style: Tensor = output['b1_style']
        b2_style: Tensor = output['b2_style']
        loss_style: Tensor = lambda_style * (
                self._style_criterion(b1_style, torch.zeros_like(b1_style)) +
                self._style_criterion(b2_style, torch.ones_like(b2_style))) / 2.
        correct1: Tensor = b1_style < 0
        correct2: Tensor = b2_style >= 0
        accuracy_style: Tensor = (correct1.sum() + correct2.sum()) / float(len(b1_style) + len(b2_style))

        # 5. Siamese Loss
        s1: Tensor = output['s1']
        s2: Tensor = output['s2']
        loss_siamese: Tensor = lambda_siamese * (s1 * s1).mean()
        #loss_siamese: Tensor = lambda_siamese * ((s1 * s1).mean() + self._siamese_criterion(s2, margin=0.5)) / 2.
        norm_s1: Tensor = torch.sqrt((s1 * s1).flatten(start_dim=1).mean(dim=1)).mean()
        norm_s2: Tensor = torch.sqrt((s2 * s2).flatten(start_dim=1).mean(dim=1)).mean()

        loss: Tensor = loss_idt + loss_latent_idt + loss_weight_cycle + loss_content + loss_style + loss_siamese

        loss_dict: Dict[str, Tensor] = {'loss': loss,
                                        'loss_identity': loss_idt, 'loss_latent_identity': loss_latent_idt,
                                        'loss_weight_cycle': loss_weight_cycle,
                                        'loss_content': loss_content, 'accuracy_content': accuracy_content,
                                        'loss_style': loss_style, 'accuracy_style': accuracy_style,
                                        'loss_siamese': loss_siamese,
                                        'norm_s1': norm_s1, 'norm_s2': norm_s2}
        return loss_dict


class SCSeparatorMnistModel(SCSeparatorModel):
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
        content_disc: nn.Module = nn.Sequential(
            nn.Linear(latent_dim, 256, bias=False), nn.BatchNorm1d(256), nn.LeakyReLU(0.01),
            nn.Linear(256, 64, bias=False), nn.BatchNorm1d(64), nn.LeakyReLU(0.01),
            nn.Linear(64, 1))
        style_disc: nn.Module = nn.Sequential(
            #nn.Dropout(p=0.5, inplace=False),
            nn.Linear(latent_dim, 256, bias=False), nn.BatchNorm1d(256), nn.LeakyReLU(0.01),
            nn.Linear(256, 64, bias=False), nn.BatchNorm1d(64), nn.LeakyReLU(0.01),
            nn.Linear(64, 1))
        scaler: Scaler = Scaler(2., 0.5)

        super().__init__(device, encoder, decoder, style_w, content_disc, style_disc, scaler)
        self.apply(weights_init)


class SCSeparatorBeautyganModel(SCSeparatorModel):
    def __init__(self, device) -> None:
        dimension = 2
        in_channels = 3
        out_channels = 3
        latent_dim = 256
        num_blocks = [4, 4]
        planes = [64, 128, 256]

        encoder: nn.Module = nn.Sequential(
            nn.Conv2d(in_channels, planes[0], kernel_size=7, stride=1, padding=3, bias=False),
            nn.InstanceNorm2d(planes[0], affine=True), nn.LeakyReLU(0.01), nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            simple_resnet(dimension, num_blocks, planes,
                          transpose=False, norm='InstanceNorm', activation='LeakyReLU', pool=False),
            Permute((0, 2, 3, 1)), nn.Linear(planes[-1], latent_dim))
        decoder: nn.Module = nn.Sequential(
            nn.Linear(latent_dim, planes[-1]), Permute((0, 3, 1, 2)),
            simple_resnet(dimension, num_blocks, planes,
                          transpose=True, norm='InstanceNorm', activation='LeakyReLU', pool=False),
            nn.ConvTranspose2d(planes[0], planes[0], kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.InstanceNorm2d(planes[0], affine=True), nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(planes[0], out_channels, kernel_size=7, stride=1, padding=3, output_padding=0),
            nn.Tanh())
        style_w: nn.Module = nn.Linear(latent_dim, latent_dim, bias=False)
        content_disc: nn.Module = nn.Sequential(
            Permute((0, 3, 1, 2)),
            spectral_norm(nn.Conv2d(latent_dim, 64, kernel_size=4, stride=2, padding=1, bias=True)), nn.LeakyReLU(0.01),
            spectral_norm(nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1, bias=True)), nn.LeakyReLU(0.01),
            spectral_norm(nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1, bias=True)), nn.LeakyReLU(0.01),
            nn.Flatten(), nn.Linear(4*4*64, 1))
        style_disc: nn.Module = nn.Sequential(
            Permute((0, 3, 1, 2)),
            spectral_norm(nn.Conv2d(latent_dim, 64, kernel_size=4, stride=2, padding=1, bias=True)), nn.LeakyReLU(0.01),
            spectral_norm(nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1, bias=True)), nn.LeakyReLU(0.01),
            spectral_norm(nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1, bias=True)), nn.LeakyReLU(0.01),
            nn.Flatten(), nn.Linear(4*4*64, 1))
        scaler: Scaler = Scaler(2., 0.5)

        super().__init__(device, encoder, decoder, style_w, content_disc, style_disc, scaler)

        self._source_disc: nn.Module = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1, bias=False)), nn.LeakyReLU(0.01),
            spectral_norm(nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1, bias=False)), nn.LeakyReLU(0.01),
            spectral_norm(nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False)), nn.LeakyReLU(0.01),
            spectral_norm(nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1, bias=False)), nn.LeakyReLU(0.01),
            spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False)), nn.LeakyReLU(0.01),
            spectral_norm(nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1, bias=False)), nn.LeakyReLU(0.01),
            nn.Flatten(), nn.Linear(4*4*128, 1))
        self._reference_disc: nn.Module = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1, bias=False)), nn.LeakyReLU(0.01),
            spectral_norm(nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1, bias=False)), nn.LeakyReLU(0.01),
            spectral_norm(nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False)), nn.LeakyReLU(0.01),
            spectral_norm(nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1, bias=False)), nn.LeakyReLU(0.01),
            spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False)), nn.LeakyReLU(0.01),
            spectral_norm(nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1, bias=False)), nn.LeakyReLU(0.01),
            nn.Flatten(), nn.Linear(4*4*128, 1))

        self._content_seg_disc: nn.Module = nn.Sequential(
            Permute((0, 3, 1, 2)),
            spectral_norm(nn.Conv2d(latent_dim, 128, kernel_size=3, stride=1, padding=1, bias=True)), nn.LeakyReLU(0.01),
            spectral_norm(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True)), nn.LeakyReLU(0.01),
            spectral_norm(nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=True)), nn.LeakyReLU(0.01),
            spectral_norm(nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True)),
            nn.LeakyReLU(0.01), nn.Conv2d(64, 15, kernel_size=7, stride=1, padding=3))
        self._style_seg_disc: nn.Module = nn.Sequential(
            Permute((0, 3, 1, 2)),
            spectral_norm(nn.Conv2d(latent_dim, 128, kernel_size=3, stride=1, padding=1, bias=True)), nn.LeakyReLU(0.01),
            spectral_norm(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True)), nn.LeakyReLU(0.01),
            spectral_norm(nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=True)), nn.LeakyReLU(0.01),
            spectral_norm(nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True)),
            nn.LeakyReLU(0.01), nn.Conv2d(64, 15, kernel_size=7, stride=1, padding=3))

        self._source_criterion = nn.BCEWithLogitsLoss()
        self._reference_criterion = nn.BCEWithLogitsLoss()
        self._content_seg_criterion = nn.CrossEntropyLoss()
        self._style_seg_criterion = nn.CrossEntropyLoss()

        self.to(self._device)
        self.apply(weights_init)

    def _update_optimizers(self, loss_dict: Dict[str, Tensor], params: Dict[str, Any],
                           global_step: int = 0) -> None:
        super()._update_optimizers(loss_dict, params, global_step)

    def _update_schedulers(self, params: Dict[str, Any], global_step: int = 0) -> None:
        super()._update_schedulers(params, global_step)

    def _set_optimizers(self, params: Dict[str, Any]) -> None:
        super()._set_optimizers(params)
        weight_decay: float = params['weight_decay']
        optimizer_source = optim.Adam(self._source_disc.parameters(), params['learning_rate'], weight_decay=weight_decay)
        optimizer_reference = optim.Adam(self._reference_disc.parameters(), params['learning_rate'], weight_decay=weight_decay)
        optimizer_content_seg = optim.Adam(self._content_seg_disc.parameters(), params['learning_rate'], weight_decay=weight_decay)
        optimizer_style_seg = optim.Adam(self._style_seg_disc.parameters(), params['learning_rate'], weight_decay=weight_decay)

        self._optimizers += [optimizer_source, optimizer_reference, optimizer_content_seg, optimizer_style_seg]
        self._schedulers += [optim.lr_scheduler.ExponentialLR(optimizer, gamma=params['scheduler_gamma']) for
                             optimizer in [optimizer_source, optimizer_reference,
                                           optimizer_content_seg, optimizer_style_seg]]

    def forward(self, batch: Dict[str, Tensor], params: Dict[str, Any],
                global_step: int = 0, **kwargs) -> Dict[str, Tensor]:
        gamma_source = params['gamma_source']
        gamma_reference = params['gamma_reference']
        gamma_content_seg = params['gamma_content_seg']
        gamma_style_seg = params['gamma_style_seg']

        output: Dict[str, Tensor] = super().forward(batch, params, global_step)

        c1: Tensor = output['c1']
        c2: Tensor = output['c2']
        s1: Tensor = output['s1']
        s2: Tensor = output['s2']

        # Source Disc Loss
        xp1: Tensor = output['xp1']
        xp21: Tensor = self._decoder(c2 + s1)
        xp20: Tensor = self._decoder(c2)
        b1_source: Tensor = self._source_disc(xp1.detach())
        b2_source: Tensor = self._source_disc(grad_reverse(xp21, gamma=gamma_source))
        b2_source2: Tensor = self._source_disc(grad_reverse(xp20, gamma=gamma_source))

        # Reference Disc Loss
        xp2: Tensor = output['xp2']
        xp12: Tensor = self._decoder(c1 + s2)
        b1_reference: Tensor = self._source_disc(xp2.detach())
        b2_reference: Tensor = self._source_disc(grad_reverse(xp12, gamma=gamma_reference))

        # Content Segmentation Disc Loss
        b1_content_seg: Tensor = self._content_seg_disc(grad_scale(c1, gamma=gamma_content_seg))
        b2_content_seg: Tensor = self._content_seg_disc(grad_scale(c2, gamma=gamma_content_seg))

        # Style Segmentation Disc Loss
        b1_style_seg: Tensor = self._style_seg_disc(grad_reverse(s1, gamma=gamma_style_seg))
        b2_style_seg: Tensor = self._style_seg_disc(grad_reverse(s2, gamma=gamma_style_seg))

        output.update({'b1_source': b1_source, 'b2_source': b2_source, 'b2_source2': b2_source2,
                       'b1_reference': b1_reference, 'b2_reference': b2_reference,
                       'b1_content_seg': b1_content_seg, 'b2_content_seg': b2_content_seg,
                       'b1_style_seg': b1_style_seg, 'b2_style_seg': b2_style_seg})
        return output

    def loss_function(self, batch: Dict[str, Tensor], output: Dict[str, Tensor], params: Dict[str, Any],
                      global_step: int = 0, **kwargs) -> Dict[str, Tensor]:
        loss_dict: Dict[str, Tensor] = super().loss_function(batch, output, params, global_step)

        # 0. Parameters
        lambda_source: float = params['lambda_source']
        lambda_reference: float = params['lambda_reference']
        lambda_content_seg: float = params['lambda_content_seg']
        lambda_style_seg: float = params['lambda_style_seg']

        # 1. Source Disc Loss
        b1_source: Tensor = output['b1_source']
        b2_source: Tensor = output['b2_source']
        b2_source2: Tensor = output['b2_source2']
        loss_source: Tensor = lambda_source * (
                self._source_criterion(b1_source, torch.zeros_like(b1_source)) +
                (self._source_criterion(b2_source, torch.ones_like(b2_source)) +
                 self._source_criterion(b2_source2, torch.ones_like(b2_source2))) / 2.) / 2.
        correct1: Tensor = b1_source < 0
        correct2: Tensor = b2_source >= 0
        correct22: Tensor = b2_source2 >= 0
        accuracy_source: Tensor = (correct1.sum() + (correct2.sum() + correct22.sum()) / 2.
                                   ) / float(len(b1_source) + len(b2_source))

        # 2. Reference Disc Loss
        b1_reference: Tensor = output['b1_reference']
        b2_reference: Tensor = output['b2_reference']
        loss_reference: Tensor = lambda_reference * (
                self._reference_criterion(b1_reference, torch.zeros_like(b1_reference)) +
                self._reference_criterion(b2_reference, torch.ones_like(b2_reference))) / 2.
        correct1: Tensor = b1_reference < 0
        correct2: Tensor = b2_reference >= 0
        accuracy_reference: Tensor = (correct1.sum() + correct2.sum()) / float(len(b1_reference) + len(b2_reference))

        # 3. Content Seg Disc Loss
        seg1: Tensor = batch['seg1']
        seg2: Tensor = batch['seg2']
        b1_content_seg: Tensor = output['b1_content_seg']
        b2_content_seg: Tensor = output['b2_content_seg']
        loss_content_seg: Tensor = lambda_content_seg * (
                self._content_seg_criterion(b1_content_seg, seg1) +
                self._content_seg_criterion(b2_content_seg, seg2)) / 2.
        correct1: Tensor = b1_content_seg.argmax(dim=1) == seg1
        correct2: Tensor = b2_content_seg.argmax(dim=1) == seg2
        accuracy_content_seg: Tensor = (correct1.sum() + correct2.sum()) / float(torch.numel(correct1) +
                                                                                 torch.numel(correct2))

        # 4. Style Seg Disc Loss
        seg1: Tensor = batch['seg1']
        seg2: Tensor = batch['seg2']
        b1_style_seg: Tensor = output['b1_style_seg']
        b2_style_seg: Tensor = output['b2_style_seg']
        loss_style_seg: Tensor = lambda_style_seg * (
                self._style_seg_criterion(b1_style_seg, seg1) +
                self._style_seg_criterion(b2_style_seg, seg2)) / 2.
        correct1: Tensor = b1_style_seg.argmax(dim=1) == seg1
        correct2: Tensor = b2_style_seg.argmax(dim=1) == seg2
        accuracy_style_seg: Tensor = (correct1.sum() + correct2.sum()) / float(torch.numel(correct1) +
                                                                               torch.numel(correct2))

        loss: Tensor = (loss_dict['loss'] +
                        loss_source + loss_reference + loss_content_seg + loss_style_seg)

        loss_dict.update({'loss': loss,
                          'loss_source': loss_source, 'accuracy_source': accuracy_source,
                          'loss_reference': loss_reference, 'accuracy_reference': accuracy_reference,
                          'loss_content_seg': loss_content_seg, 'accuracy_content_seg': accuracy_content_seg,
                          'loss_style_seg': loss_style_seg, 'accuracy_style_seg': accuracy_style_seg})
        return loss_dict


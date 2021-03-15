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
from scgan.deep.grad import grad_scale, grad_reverse
from scgan.deep.resnet import weights_init, simple_resnet, simple_bottleneck_resnet


class SCSeparatorModel(BaseModel):
    def __init__(self, device, encoder: nn.Module, decoder: nn.Module, style_w: nn.Module,
                 content_disc: nn.Module, style_disc: nn.Module, cross_disc: nn.Module,
                 scaler: Scaler = None) -> None:
        super().__init__(device)
        self._encoder = encoder
        self._decoder = decoder
        self._style_w = style_w
        self._content_disc = content_disc
        self._style_disc = style_disc
        self._cross_disc = cross_disc
        if scaler is None:
            scaler = Scaler(1., 0.)
        self._scaler = scaler

        self._identity_criterion = nn.MSELoss()
        self._weight_criterion = nn.MSELoss()
        self._content_criterion = nn.BCEWithLogitsLoss()
        self._style_criterion = nn.BCEWithLogitsLoss()
        self._cross_criterion = nn.BCEWithLogitsLoss()
        self._entropy_criterion = BinaryEntropyWithLogitsLoss()
        self._siamese_criterion = SiameseLoss()

        self.to(self._device)

    def _update_optimizers(self, loss_dict: Dict[str, Tensor], params: Dict[str, Any],
                           global_step: int = 0) -> None:
        (optimizer_enc, optimizer_dec, optimizer_w,
         optimizer_content, optimizer_style, optimizer_cross) = self._optimizers

        loss_idt: Tensor = loss_dict['loss_idt']
        loss_cycle: Tensor = loss_dict['loss_cycle']
        loss_content: Tensor = loss_dict['loss_content']
        loss_content_entropy: Tensor = loss_dict['loss_content_entropy']
        loss_style: Tensor = loss_dict['loss_style']
        loss_cross: Tensor = loss_dict['loss_cross']
        loss_cross_entropy: Tensor = loss_dict['loss_cross_entropy']
        loss_siamese: Tensor = loss_dict['loss_siamese']

        optimizer_enc.zero_grad()
        optimizer_dec.zero_grad()
        optimizer_w.zero_grad()
        optimizer_content.zero_grad()
        optimizer_style.zero_grad()
        optimizer_cross.zero_grad()

        if global_step < 1000000 or global_step % 5 != 0:
            #loss_content.backward(retain_graph=True)
            #loss_cross.backward(retain_graph=True)
            #optimizer_enc.zero_grad()
            #optimizer_w.zero_grad()
            #loss_entropy.backward(retain_graph=True)
            #optimizer_content.zero_grad()
            #optimizer_cross.zero_grad()
            #(loss_idt + loss_cycle + loss_content).backward(retain_graph=True)
            (loss_idt + loss_cycle + loss_content + loss_style + loss_siamese).backward(retain_graph=True)
            #(loss_idt + loss_cycle + loss_content + loss_style + loss_cross + loss_siamese).backward(retain_graph=True)
            #(loss_idt + loss_cycle + loss_siamese + loss_entropy).backward(retain_graph=True)
            #(loss_idt + loss_cycle + loss_content + loss_cross + loss_siamese).backward(retain_graph=True)
        else:
            (loss_idt + loss_cycle + loss_content + loss_style).backward(retain_graph=True)
            #(loss_idt + loss_cycle + loss_content + loss_style + loss_cross).backward(retain_graph=True)
        #(loss_idt + loss_style + loss_cycle).backward(retain_graph=True)
        #(loss_idt + loss_entropy + loss_style + loss_cycle).backward(retain_graph=True)
        #optimizer_content.zero_grad()
        #loss_content.backward()
        if 'clip_size' in params:
            clip_grad_norm_(self._encoder.parameters(), params['clip_size'])
            clip_grad_norm_(self._decoder.parameters(), params['clip_size'])
            clip_grad_norm_(self._style_w.parameters(), params['clip_size'])
            clip_grad_norm_(self._content_disc.parameters(), params['clip_size'])
            clip_grad_norm_(self._style_disc.parameters(), params['clip_size'])
            clip_grad_norm_(self._cross_disc.parameters(), params['clip_size'])

        optimizer_enc.step()
        optimizer_dec.step()
        optimizer_w.step()
        optimizer_content.step()
        optimizer_style.step()
        optimizer_cross.step()

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
        optimizer_cross = optim.Adam(self._cross_disc.parameters(), params['learning_rate'], weight_decay=weight_decay)

        self._optimizers = [optimizer_enc, optimizer_dec, optimizer_w,
                            optimizer_content, optimizer_style, optimizer_cross]
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
            x2_idt: Tensor = self._scaler.unscaling(self._decoder(c2 + s2))
            x12: Tensor = self._scaler.unscaling(self._decoder(c1 + s2))
            x21: Tensor = self._scaler.unscaling(self._decoder(c2 + s1))
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
        gamma_content = params['gamma_content']
        gamma_style = params['gamma_style']
        gamma_cross = params['gamma_cross']

        xp1: Tensor = self._scaler.scaling(batch['x1'])
        xp2: Tensor = self._scaler.scaling(batch['x2'])
        z1, s1, c1 = self._style_content_separate(xp1)
        z2, s2, c2 = self._style_content_separate(xp2)

        # Identity Loss
        xp1_idt: Tensor = self._decoder(z1)
        xp1_idt2: Tensor = self._decoder(c1)  # Instead using latent, using content only.
        xp2_idt: Tensor = self._decoder(z2)

        # Weight Cycle Loss
        s1_detach: Tensor = self._style_w(z1.detach())
        c1_detach: Tensor = z1.detach() - s1_detach
        s2_detach: Tensor = self._style_w(z2.detach())
        c2_detach: Tensor = z2.detach() - s2_detach
        z12: Tensor = (c1_detach + s2_detach)
        #z12: Tensor = (c1 + s2)
        s2_idt: Tensor = self._style_w(z12)
        c1_idt: Tensor = z12 - s2_idt
        z21: Tensor = (c2_detach + s1_detach)
        #z21: Tensor = (c2 + s1)
        s1_idt: Tensor = self._style_w(z21)
        c2_idt: Tensor = z21 - s1_idt

        # Content Disc Loss
        b1_content: Tensor = self._content_disc(c1.detach())#self._content_disc(grad_reverse(c1, gamma=gamma_content))
        b2_content: Tensor = self._content_disc(grad_reverse(c2, gamma=gamma_content))

        # Style Disc Loss
        b1_style: Tensor = self._style_disc(grad_scale(s1, gamma=gamma_style))
        b2_style: Tensor = self._style_disc(grad_scale(s2, gamma=gamma_style))

        # Cross Disc Loss
        b11_cross: Tensor = self._cross_disc(torch.cat([c1, s1], dim=1).detach())#self._cross_disc(grad_reverse(torch.cat([c1, s1], dim=1), gamma=gamma_cross))
        b22_cross: Tensor = self._cross_disc(torch.cat([c2, s2], dim=1).detach())#self._cross_disc(grad_reverse(torch.cat([c2, s2], dim=1), gamma=gamma_cross))
        b12_cross: Tensor = self._cross_disc(grad_reverse(torch.cat([c1, s2], dim=1), gamma=gamma_cross))
        b21_cross: Tensor = self._cross_disc(grad_reverse(torch.cat([c2, s1], dim=1), gamma=gamma_cross))

        output: Dict[str, Tensor] = {'xp1': xp1, 'xp2': xp2,
                                     'z1': z1, 'z2': z2, 's1': s1, 's2': s2, 'c1': c1, 'c2': c2,
                                     'xp1_idt': xp1_idt,'xp1_idt2': xp1_idt2,  'xp2_idt': xp2_idt,
                                     'c1_detach': c1_detach, 'c2_detach': c2_detach,
                                     's1_detach': s1_detach, 's2_detach': s2_detach,
                                     'c1_idt': c1_idt, 'c2_idt': c2_idt, 's1_idt': s1_idt, 's2_idt': s2_idt,
                                     'b1_content': b1_content, 'b2_content': b2_content,
                                     'b1_style': b1_style, 'b2_style': b2_style,
                                     'b11_cross': b11_cross, 'b22_cross': b22_cross,
                                     'b12_cross': b12_cross, 'b21_cross': b21_cross}
        return output

    def loss_function(self, batch: Dict[str, Tensor], output: Dict[str, Tensor], params: Dict[str, Any],
                      global_step: int = 0, **kwargs) -> Dict[str, Tensor]:
        # 0. Parameters
        lambda_idt: float = params['lambda_idt']
        lambda_cycle: float = params['lambda_cycle']
        lambda_content: float = params['lambda_content']
        lambda_content_entropy: float = params['lambda_content_entropy']
        lambda_style: float = params['lambda_style']
        lambda_cross: float = params['lambda_cross']
        lambda_cross_entropy: float = params['lambda_cross_entropy']
        lambda_siamese: float = params['lambda_siamese']

        # 1. Identity Loss
        xp1: Tensor = output['xp1']
        xp2: Tensor = output['xp2']
        xp1_idt: Tensor = output['xp1_idt']
        xp1_idt2: Tensor = output['xp1_idt2']
        xp2_idt: Tensor = output['xp2_idt']
        assert(xp1.size() == xp1_idt.size() == xp1_idt2.size() and xp2.size() == xp2_idt.size())
        loss_idt: Tensor = lambda_idt * (
                self._identity_criterion(xp1_idt, xp1) +
                self._identity_criterion(xp2_idt, xp2)) / 2.
        #loss_idt: Tensor = lambda_idt * (
        #        self._identity_criterion(xp1_idt, xp1) + self._identity_criterion(xp1_idt2, xp1) +
        #        self._identity_criterion(xp2_idt, xp2)) / 3.

        # 2. Weight Cycle Loss
        c1: Tensor = output['c1']
        c2: Tensor = output['c2']
        s1: Tensor = output['s1']
        s2: Tensor = output['s2']
        c1_detach: Tensor = output['c1_detach']
        c2_detach: Tensor = output['c2_detach']
        s1_detach: Tensor = output['s1_detach']
        s2_detach: Tensor = output['s2_detach']
        c1_idt: Tensor = output['c1_idt']
        c2_idt: Tensor = output['c2_idt']
        s1_idt: Tensor = output['s1_idt']
        s2_idt: Tensor = output['s2_idt']
        loss_cycle: Tensor = lambda_cycle * (
                self._weight_criterion(c1_idt, c1_detach) + self._weight_criterion(c2_idt, c2_detach) +
                self._weight_criterion(s1_idt, s1_detach) + self._weight_criterion(s2_idt, s2_detach)) / 4.
        #loss_cycle: Tensor = lambda_cycle * (
        #        self._weight_criterion(c1_idt, c1) + self._weight_criterion(c2_idt, c2) +
        #        self._weight_criterion(s1_idt, s1) + self._weight_criterion(s2_idt, s2)) / 4.

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

        # 5. Cross Disc Loss
        b11_cross: Tensor = output['b11_cross']
        b22_cross: Tensor = output['b22_cross']
        b12_cross: Tensor = output['b12_cross']
        b21_cross: Tensor = output['b21_cross']
        loss_cross: Tensor = lambda_cross * (
                self._cross_criterion(b11_cross, torch.zeros_like(b11_cross)) +
                self._cross_criterion(b22_cross, torch.zeros_like(b22_cross)) +
                self._cross_criterion(b12_cross, torch.ones_like(b12_cross)) +
                self._cross_criterion(b21_cross, torch.ones_like(b21_cross))) / 4.
        correct11: Tensor = b11_cross < 0
        correct22: Tensor = b22_cross < 0
        correct12: Tensor = b12_cross >= 0
        correct21: Tensor = b21_cross >= 0
        accuracy_cross: Tensor = (correct11.sum() + correct22.sum() + correct12.sum() + correct21.sum()
                                  ) / float(len(b11_cross) + len(b22_cross) + len(b12_cross) + len(b21_cross))

        # 6. Entropy Loss

        loss_content_entropy: Tensor = lambda_content_entropy * (
                self._entropy_criterion(b1_content) + self._entropy_criterion(b2_content)) / 2.

        loss_cross_entropy: Tensor = lambda_cross_entropy * (
                self._entropy_criterion(b11_cross) + self._entropy_criterion(b22_cross) +
                self._entropy_criterion(b12_cross) + self._entropy_criterion(b21_cross)) / 4.

        # 7. Siamese Loss
        loss_siamese: Tensor = lambda_siamese * (s1 * s1).mean()
        #loss_siamese: Tensor = lambda_siamese * ((s1 * s1).mean() + self._siamese_criterion(s2, margin=0.5)) / 2.
        norm_s1: Tensor = torch.sqrt((s1 * s1).flatten(start_dim=1).mean(dim=1)).mean()
        norm_s2: Tensor = torch.sqrt((s2 * s2).flatten(start_dim=1).mean(dim=1)).mean()

        loss: Tensor = (loss_idt + loss_cycle + loss_content + loss_content_entropy + loss_style +
                        loss_cross + loss_cross_entropy + loss_siamese)

        loss_dict: Dict[str, Tensor] = {'loss': loss,
                                        'loss_idt': loss_idt, 'loss_cycle': loss_cycle,
                                        'loss_content': loss_content, 'accuracy_content': accuracy_content,
                                        'loss_content_entropy': loss_content_entropy,
                                        'loss_style': loss_style, 'accuracy_style': accuracy_style,
                                        'loss_cross': loss_cross, 'accuracy_cross': accuracy_cross,
                                        'loss_cross_entropy': loss_cross_entropy,
                                        'loss_siamese': loss_siamese,
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
        content_disc: nn.Module = nn.Sequential(
            nn.Linear(latent_dim, 256, bias=False), nn.BatchNorm1d(256), nn.LeakyReLU(0.01),
            nn.Linear(256, 64, bias=False), nn.BatchNorm1d(64), nn.LeakyReLU(0.01),
            nn.Linear(64, 1), nn.Flatten())
        style_disc: nn.Module = nn.Sequential(
            #nn.Dropout(p=0.5, inplace=False),
            nn.Linear(latent_dim, 256, bias=False), nn.BatchNorm1d(256), nn.LeakyReLU(0.01),
            nn.Linear(256, 64, bias=False), nn.BatchNorm1d(64), nn.LeakyReLU(0.01),
            nn.Linear(64, 1), nn.Flatten())
        cross_disc: nn.Module = nn.Sequential(
            nn.Linear(2*latent_dim, 256, bias=False), nn.BatchNorm1d(256), nn.LeakyReLU(0.01),
            nn.Linear(256, 64, bias=False), nn.BatchNorm1d(64), nn.LeakyReLU(0.01),
            nn.Linear(64, 1), nn.Flatten())
        scaler: Scaler = Scaler(2., 0.5)

        super().__init__(device, encoder, decoder, style_w, content_disc, style_disc, cross_disc, scaler)
        self.apply(weights_init)

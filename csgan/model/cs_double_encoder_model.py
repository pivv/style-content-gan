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
from csgan.deep.loss import SiameseLoss, L1SiameseLoss, BinaryEntropyWithLogitsLoss
from csgan.deep.layer import View, Permute
from csgan.deep.grad import grad_scale, grad_reverse
from csgan.deep.resnet import simple_resnet, simple_bottleneck_resnet
from csgan.deep.initialize import weights_init_resnet
from csgan.deep.norm import spectral_norm


class CSDoubleEncoderModel(BaseModel):
    def __init__(self, device, content_encoder: nn.Module, style_encoder: nn.Module, decoder: nn.Module,
                 content_disc: nn.Module, style_disc: nn.Module,
                 source_disc: nn.Module, reference_disc: nn.Module,
                 content_seg_disc: nn.Module, style_seg_disc: nn.Module,
                 scaler: Scaler = None) -> None:
        super().__init__(device)
        self._content_encoder: nn.Module = content_encoder
        self._style_encoder: nn.Module = style_encoder
        self._decoder: nn.Module = decoder

        self._content_disc: nn.Module = content_disc
        self._style_disc: nn.Module = style_disc

        self._source_disc: nn.Module = source_disc
        self._reference_disc: nn.Module = reference_disc
        self._content_seg_disc: nn.Module = content_seg_disc
        self._style_seg_disc: nn.Module = style_seg_disc

        if scaler is None:
            scaler = Scaler(1., 0.)
        self._scaler = scaler

        self._identity_criterion = nn.L1Loss()
        self._cycle_criterion = nn.L1Loss()
        self._content_criterion = nn.MSELoss()
        self._style_criterion = nn.MSELoss()
        self._siamese_criterion = SiameseLoss()
        #self._siamese_criterion = L1SiameseLoss()

        self._source_criterion = nn.MSELoss()
        self._reference_criterion = nn.MSELoss()
        self._content_seg_criterion = nn.MSELoss()
        self._style_seg_criterion = nn.MSELoss()

        self.to(self._device)

    def _update_optimizers(self, loss_dict: Dict[str, Tensor], params: Dict[str, Any],
                           global_step: int = 0) -> None:
        pass

    def _update_schedulers(self, params: Dict[str, Any], global_step: int = 0) -> None:
        if global_step % params['scheduler_interval'] == 0:
            for scheduler in self._schedulers:
                if scheduler is not None:
                    scheduler.step()

        if global_step == 1 or global_step % params['scheduler_interval'] == 0:
            assert(self._schedulers[0] is not None)
            lr = self._schedulers[0].get_last_lr()
            print("")
            print(f"Learning with learning rate: {lr[0]: .8f}.")
            print("")

    def _set_optimizers(self, params: Dict[str, Any]) -> None:
        lr: float = params['learning_rate']
        beta1: float = params['beta1']
        beta2: float = params['beta2']
        weight_decay: float = params['weight_decay']
        gamma: float = params['scheduler_gamma']
        self._optimizers = [optim.Adam(module.parameters(), lr, (beta1, beta2), weight_decay=weight_decay) if
                            module is not None else None for
                            module in [self._content_encoder, self._style_encoder, self._decoder,
                                       self._content_disc, self._style_disc,
                                       self._source_disc, self._reference_disc,
                                       self._content_seg_disc, self._style_seg_disc]]
        self._schedulers = [optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma) if
                            optimizer is not None else None for optimizer in self._optimizers]

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
            z1: Tensor = self._cs_to_latent(c1, s1)
            z2: Tensor = self._cs_to_latent(c2, s2)
            x1_idt: Tensor = self._scaler.unscaling(self._decoder(self._cs_to_latent(c1)))
            x2_idt: Tensor = self._scaler.unscaling(self._decoder(self._cs_to_latent(c2, s2)))
            xp12: Tensor = self._decoder(self._cs_to_latent(c1, s2))
            xp21: Tensor = self._decoder(self._cs_to_latent(c2))
            x12: Tensor = self._scaler.unscaling(xp12)
            x21: Tensor = self._scaler.unscaling(xp21)
            x1_cycle: Tensor = self._scaler.unscaling(self._decoder(self._cs_to_latent(self._content_encoder(xp12))))
            x2_cycle: Tensor = self._scaler.unscaling(self._decoder(self._cs_to_latent(self._content_encoder(xp21), s2)))
            output: Dict[str, Tensor] = {'z1': z1, 'z2': z2, 's1': s1, 's2': s2, 'c1': c1, 'c2': c2,
                                         'x1_idt': x1_idt, 'x2_idt': x2_idt, 'x12': x12, 'x21': x21,
                                         'x1_cycle': x1_cycle, 'x2_cycle': x2_cycle}
        return output

    def _post_processing(self, batch: Dict[str, Tensor], params: Dict[str, Any],
                         global_step: int = 0) -> None:
        if global_step == 1 or global_step % params['sampling_interval'] == 0:
            self.eval()
            with torch.no_grad():
                output: Dict[str, Tensor] = self._predict(batch)
            result_dir = os.path.join(params['run_dir'], 'results')

            n: int = len(batch['x1'])
            fig = plt.figure(figsize=(20, (20 * n) // 8))
            for index in range(n):
                ax = fig.add_subplot(n, 8, 8*index+1)
                ax.imshow(batch['x1'][index, :3].detach().cpu().numpy().transpose(1, 2, 0))
                ax = fig.add_subplot(n, 8, 8*index+2)
                ax.imshow(batch['x2'][index, :3].detach().cpu().numpy().transpose(1, 2, 0))
                ax = fig.add_subplot(n, 8, 8*index+3)
                ax.imshow(output['x1_idt'][index].detach().cpu().numpy().transpose(1, 2, 0))
                ax = fig.add_subplot(n, 8, 8*index+4)
                ax.imshow(output['x2_idt'][index].detach().cpu().numpy().transpose(1, 2, 0))
                ax = fig.add_subplot(n, 8, 8*index+5)
                ax.imshow(output['x12'][index].detach().cpu().numpy().transpose(1, 2, 0))
                ax = fig.add_subplot(n, 8, 8*index+6)
                ax.imshow(output['x21'][index].detach().cpu().numpy().transpose(1, 2, 0))
                ax = fig.add_subplot(n, 8, 8*index+7)
                ax.imshow(output['x1_cycle'][index].detach().cpu().numpy().transpose(1, 2, 0))
                ax = fig.add_subplot(n, 8, 8*index+8)
                ax.imshow(output['x2_cycle'][index].detach().cpu().numpy().transpose(1, 2, 0))

            plt.savefig(os.path.join(result_dir, f"sampling_{global_step}.png"), dpi=200, bbox_inches='tight')
            plt.close('all')

    def forward(self, batch: Dict[str, Tensor], params: Dict[str, Any],
                global_step: int = 0, **kwargs) -> Dict[str, Tensor]:
        output: Dict[str, Tensor] = {}
        return output

    def loss_function(self, batch: Dict[str, Tensor], output: Dict[str, Tensor], params: Dict[str, Any],
                      global_step: int = 0, **kwargs) -> Dict[str, Tensor]:
        (optimizer_content_encoder, optimizer_style_encoder, optimizer_decoder,
         optimizer_content_disc, optimizer_style_disc,
         optimizer_source_disc, optimizer_reference_disc,
         optimizer_content_seg_disc, optimizer_style_seg_disc) = self._optimizers

        # 0. Parameters
        lambda_idt: float = params['lambda_identity']
        lambda_cycle: float = params['lambda_cycle']
        lambda_content: float = params['lambda_content']
        lambda_style: float = params['lambda_style']
        lambda_siamese: float = params['lambda_siamese']

        lambda_source: float = params['lambda_source']
        lambda_reference: float = params['lambda_reference']
        lambda_content_seg: float = params['lambda_content_seg']
        lambda_style_seg: float = params['lambda_style_seg']

        gamma_content = params['gamma_content']
        gamma_style = params['gamma_style']

        gamma_source = params['gamma_source']
        gamma_reference = params['gamma_reference']
        gamma_content_seg = params['gamma_content_seg']
        gamma_style_seg = params['gamma_style_seg']

        xp1: Tensor = self._scaler.scaling(batch['x1']).detach()
        xp2: Tensor = self._scaler.scaling(batch['x2']).detach()

        # 1. Disc Loss

        with torch.no_grad():
            c1_detach: Tensor = self._content_encoder(xp1).detach()
            s1_detach: Tensor = self._style_encoder(xp1).detach()
            c2_detach: Tensor = self._content_encoder(xp2).detach()
            s2_detach: Tensor = self._style_encoder(xp2).detach()

        # 1-1. Content Disc Loss

        loss_content: Tensor = torch.FloatTensor([0.])[0].to(self._device)
        accuracy_content: Tensor = torch.FloatTensor([0.5])[0].to(self._device)
        if lambda_content > 0:
            b1_content: Tensor = self._content_disc(c1_detach)
            b2_content: Tensor = self._content_disc(c2_detach)

            loss_content: Tensor = lambda_content * (self._content_criterion(b1_content, torch.ones_like(b1_content)) +
                                                     self._content_criterion(b2_content, torch.zeros_like(b2_content))) / 2.
            correct1: Tensor = b1_content >= 0.5
            correct2: Tensor = b2_content < 0.5
            accuracy_content: Tensor = (correct1.sum() + correct2.sum()) / float(len(b1_content) + len(b2_content))

            optimizer_content_disc.zero_grad()
            loss_content.backward()
            optimizer_content_disc.step()

        # 1-2. Style Disc Loss

        loss_style: Tensor = torch.FloatTensor([0.])[0].to(self._device)
        accuracy_style: Tensor = torch.FloatTensor([0.5])[0].to(self._device)
        if lambda_style > 0:
            b1_style: Tensor = self._style_disc(s1_detach)
            b2_style: Tensor = self._style_disc(s2_detach)

            loss_style: Tensor = lambda_style * (self._style_criterion(b1_style, torch.ones_like(b1_style)) +
                                                 self._style_criterion(b2_style, torch.zeros_like(b2_style))) / 2.
            correct1: Tensor = b1_style >= 0.5
            correct2: Tensor = b2_style < 0.5
            accuracy_style: Tensor = (correct1.sum() + correct2.sum()) / float(len(b1_style) + len(b2_style))

            optimizer_style_disc.zero_grad()
            loss_style.backward()
            optimizer_style_disc.step()

        # 1-3. Source Disc Loss

        loss_source: Tensor = torch.FloatTensor([0.])[0].to(self._device)
        accuracy_source: Tensor = torch.FloatTensor([0.5])[0].to(self._device)
        if lambda_source > 0:
            with torch.no_grad():
                xp21_detach: Tensor = self._decoder(self._cs_to_latent(c2_detach)).detach()
            b1_source: Tensor = self._source_disc(xp1)
            b2_source: Tensor = self._source_disc(xp21_detach)

            loss_source: Tensor = lambda_source * (self._source_criterion(b1_source, torch.ones_like(b1_source)) +
                                                   self._source_criterion(b2_source, torch.zeros_like(b2_source))) / 2.
            correct1: Tensor = b1_source >= 0.5
            correct2: Tensor = b2_source < 0.5
            accuracy_source: Tensor = (correct1.sum() + correct2.sum()) / float(len(b1_source) + len(b2_source))

            optimizer_source_disc.zero_grad()
            loss_source.backward()
            optimizer_source_disc.step()

        # 1-4. Reference Disc Loss

        loss_reference: Tensor = torch.FloatTensor([0.])[0].to(self._device)
        accuracy_reference: Tensor = torch.FloatTensor([0.5])[0].to(self._device)
        if lambda_reference > 0:
            with torch.no_grad():
                xp12_detach: Tensor = self._decoder(self._cs_to_latent(c1_detach, s2_detach)).detach()
            b1_reference: Tensor = self._reference_disc(xp2)
            b2_reference: Tensor = self._reference_disc(xp12_detach)

            loss_reference: Tensor = lambda_reference * (self._reference_criterion(b1_reference, torch.ones_like(b1_reference)) +
                                                         self._reference_criterion(b2_reference, torch.zeros_like(b2_reference))) / 2.
            correct1: Tensor = b1_reference >= 0.5
            correct2: Tensor = b2_reference < 0.5
            accuracy_reference: Tensor = (correct1.sum() + correct2.sum()) / float(len(b1_reference) + len(b2_reference))

            optimizer_reference_disc.zero_grad()
            loss_reference.backward()
            optimizer_reference_disc.step()

        # 1-5. Content Seg Loss

        seg1: Tensor = batch['seg1'] if 'seg1' in batch else None
        seg2: Tensor = batch['seg2'] if 'seg2' in batch else None

        loss_content_seg: Tensor = torch.FloatTensor([0.])[0].to(self._device)
        accuracy_content_seg: Tensor = torch.FloatTensor([0.5])[0].to(self._device)
        if lambda_content_seg > 0:
            b1_content_seg: Tensor = F.softmax(self._content_seg_disc(c1_detach), dim=1).gather(dim=1, index=seg1.unsqueeze(1))
            b2_content_seg: Tensor = F.softmax(self._content_seg_disc(c2_detach), dim=1).gather(dim=1, index=seg2.unsqueeze(1))

            loss_content_seg: Tensor = lambda_content_seg * (
                    self._content_seg_criterion(b1_content_seg, torch.ones_like(b1_content_seg)) +
                    self._content_seg_criterion(b2_content_seg, torch.ones_like(b2_content_seg))) / 2.
            correct1: Tensor = b1_content_seg.argmax(dim=1) == seg1
            correct2: Tensor = b2_content_seg.argmax(dim=1) == seg2
            accuracy_content_seg: Tensor = (correct1.sum() + correct2.sum()) / float(torch.numel(correct1) +
                                                                                     torch.numel(correct2))

            optimizer_content_seg_disc.zero_grad()
            loss_content_seg.backward()
            optimizer_content_seg_disc.step()

        # 1-6. Style Seg Loss

        loss_style_seg: Tensor = torch.FloatTensor([0.])[0].to(self._device)
        accuracy_style_seg: Tensor = torch.FloatTensor([0.5])[0].to(self._device)
        if lambda_style_seg > 0:
            b1_style_seg: Tensor = F.softmax(self._style_seg_disc(s1_detach), dim=1).gather(dim=1, index=seg1.unsqueeze(1))
            b2_style_seg: Tensor = F.softmax(self._style_seg_disc(s2_detach), dim=1).gather(dim=1, index=seg2.unsqueeze(1))

            loss_style_seg: Tensor = lambda_style_seg * (
                    self._style_seg_criterion(b1_style_seg, torch.ones_like(b1_style_seg)) +
                    self._style_seg_criterion(b2_style_seg, torch.ones_like(b2_style_seg))) / 2.
            correct1: Tensor = b1_style_seg.argmax(dim=1) == seg1
            correct2: Tensor = b2_style_seg.argmax(dim=1) == seg2
            accuracy_style_seg: Tensor = (correct1.sum() + correct2.sum()) / float(torch.numel(correct1) +
                                                                                     torch.numel(correct2))

            optimizer_style_seg_disc.zero_grad()
            loss_style_seg.backward()
            optimizer_style_seg_disc.step()

        # 2. Encoder Loss

        for module in [self._content_disc, self._style_disc, self._source_disc, self._reference_disc,
                       self._content_seg_disc, self._style_seg_disc]:
            if module is not None:
                module.requires_grad_(False)

        c1: Tensor = self._content_encoder(xp1)
        s1: Tensor = self._style_encoder(xp1)
        c2: Tensor = self._content_encoder(xp2)
        s2: Tensor = self._style_encoder(xp2)

        # 2-1. Content Encoder Loss

        loss_content_encoder: Tensor = torch.FloatTensor([0.])[0].to(self._device)
        if lambda_content > 0:
            b2_content: Tensor = self._content_disc(c2)
            loss_content_encoder: Tensor = lambda_content * gamma_content * (
                self._content_criterion(b2_content, torch.ones_like(b2_content)))

        # 2-2. Style Encoder Loss

        loss_style_encoder: Tensor = torch.FloatTensor([0.])[0].to(self._device)
        if lambda_style > 0:
            b1_style: Tensor = self._style_disc(s1)
            b2_style: Tensor = self._style_disc(s2)
            loss_style_encoder: Tensor = lambda_style * gamma_style * (
                    self._style_criterion(b1_style, torch.ones_like(b1_style)) +
                    self._style_criterion(b2_style, torch.zeros_like(b2_style))) / 2.

        xp21: Tensor = self._decoder(self._cs_to_latent(c2))
        xp12: Tensor = self._decoder(self._cs_to_latent(c1, s2))

        # 2-3. Source Encoder Loss

        loss_source_encoder: Tensor = torch.FloatTensor([0.])[0].to(self._device)
        if lambda_source > 0:
            #xp21_detach: Tensor = self._decoder(self._cs_to_latent(c2).detach())
            #b2_source: Tensor = self._source_disc(xp21_detach)
            b2_source: Tensor = self._source_disc(xp21)

            loss_source_encoder: Tensor = lambda_source * gamma_source * (
                    self._source_criterion(b2_source, torch.ones_like(b2_source))) / 2.

        # 2-4. Reference Encoder Loss

        loss_reference_encoder: Tensor = torch.FloatTensor([0.])[0].to(self._device)
        if lambda_reference > 0:
            #xp12_detach: Tensor = self._decoder(self._cs_to_latent(c1, s2).detach())
            #b2_reference: Tensor = self._reference_disc(xp12_detach)
            b2_reference: Tensor = self._reference_disc(xp12)

            loss_reference_encoder: Tensor = lambda_reference * gamma_reference * (
                self._reference_criterion(b2_reference, torch.ones_like(b2_reference))) / 2.

        # 2-5. Content Seg Encoder Loss

        loss_content_seg_encoder: Tensor = torch.FloatTensor([0.])[0].to(self._device)
        if lambda_content_seg > 0:
            b1_content_seg: Tensor = F.softmax(self._content_seg_disc(c1), dim=1).gather(dim=1, index=seg1.unsqueeze(1))
            b2_content_seg: Tensor = F.softmax(self._content_seg_disc(c2), dim=1).gather(dim=1, index=seg2.unsqueeze(1))

            loss_content_seg_encoder: Tensor = lambda_content_seg * gamma_content_seg * (
                    self._content_seg_criterion(b1_content_seg, torch.ones_like(b1_content_seg)) +
                    self._content_seg_criterion(b2_content_seg, torch.ones_like(b2_content_seg))) / 2.

        # 2-6. Style Seg Encoder Loss

        loss_style_seg_encoder: Tensor = torch.FloatTensor([0.])[0].to(self._device)
        if lambda_style_seg > 0:
            b1_style_seg: Tensor = F.softmax(self._style_seg_disc(s1), dim=1).gather(dim=1, index=seg1.unsqueeze(1))
            b2_style_seg: Tensor = F.softmax(self._style_seg_disc(s2), dim=1).gather(dim=1, index=seg2.unsqueeze(1))

            loss_style_seg_encoder: Tensor = lambda_style_seg * gamma_style_seg * (
                    self._style_seg_criterion(b1_style_seg, torch.zeros_like(b1_style_seg)) +
                    self._style_seg_criterion(b2_style_seg, torch.zeros_like(b2_style_seg))) / 2.

        # 2-7. Identity Loss

        xp1_idt: Tensor = self._decoder(self._cs_to_latent(c1))
        xp1_idt2: Tensor = self._decoder(self._cs_to_latent(c1, s1))
        xp2_idt: Tensor = self._decoder(self._cs_to_latent(c2, s2))

        loss_idt: Tensor = torch.FloatTensor([0.])[0].to(self._device)
        if lambda_idt > 0:
            loss_idt: Tensor = lambda_idt * (
                    (self._identity_criterion(xp1_idt, xp1[:, :3]) +
                     self._identity_criterion(xp1_idt2, xp1[:, :3])) / 2. +
                    self._identity_criterion(xp2_idt, xp2[:, :3])) / 2.

        # 2-8. Cycle Loss

        loss_cycle: Tensor = torch.FloatTensor([0.])[0].to(self._device)
        if lambda_cycle > 0:
            xp1_cycle: Tensor = self._decoder(self._cs_to_latent(self._content_encoder(xp12)))
            xp2_cycle: Tensor = self._decoder(self._cs_to_latent(self._content_encoder(xp21), s2))

            loss_cycle: Tensor = lambda_cycle * (
                    self._cycle_criterion(xp1_cycle, xp1[:, :3]) +
                    self._cycle_criterion(xp2_cycle, xp2[:, :3])) / 2.

        # 2-9. Siamese Loss

        loss_siamese: Tensor = torch.FloatTensor([0.])[0].to(self._device)
        if lambda_siamese > 0:
            loss_siamese: Tensor = lambda_siamese * (s1 * s1).mean()
            #loss_siamese: Tensor = lambda_siamese * (s1.abs().mean() + self._siamese_criterion(s2, margin=1.)) / 2.
            #loss_siamese: Tensor = lambda_siamese * ((s1 * s1).mean() + self._siamese_criterion(s2, margin=1.)) / 2.
        norm_s1: Tensor = torch.sqrt((s1 * s1).flatten(start_dim=1).mean(dim=1)).mean()
        norm_s2: Tensor = torch.sqrt((s2 * s2).flatten(start_dim=1).mean(dim=1)).mean()

        loss: Tensor = (loss_content_encoder + loss_style_encoder +
                        loss_source_encoder + loss_reference_encoder +
                        loss_content_seg_encoder + loss_style_seg_encoder +
                        loss_idt + loss_cycle + loss_siamese)

        optimizer_content_encoder.zero_grad()
        optimizer_style_encoder.zero_grad()
        optimizer_decoder.zero_grad()
        loss.backward()
        optimizer_content_encoder.step()
        optimizer_style_encoder.step()
        optimizer_decoder.step()

        for module in [self._content_disc, self._style_disc, self._source_disc, self._reference_disc,
                       self._content_seg_disc, self._style_seg_disc]:
            if module is not None:
                module.requires_grad_(True)

        loss_dict: Dict[str, Tensor] = {'loss': loss,
                                        'loss_identity': loss_idt, 'loss_cycle': loss_cycle,
                                        'loss_content': loss_content, 'accuracy_content': accuracy_content,
                                        'loss_style': loss_style, 'accuracy_style': accuracy_style,

                                        'loss_source': loss_source, 'accuracy_source': accuracy_source,
                                        'loss_reference': loss_reference, 'accuracy_reference': accuracy_reference,
                                        'loss_content_seg': loss_content_seg, 'accuracy_content_seg': accuracy_content_seg,
                                        'loss_style_seg': loss_style_seg, 'accuracy_style_seg': accuracy_style_seg,

                                        'loss_siamese': loss_siamese,
                                        'norm_s1': norm_s1, 'norm_s2': norm_s2}
        return loss_dict


class CSDoubleEconderMnistModel(CSDoubleEncoderModel):
    def __init__(self, device) -> None:
        dimension = 2
        in_channels = 3
        content_dim = 512
        style_dim = 64
        latent_dim = content_dim + style_dim
        num_blocks = [4]
        planes = [64, 64]

        #content_encoder: nn.Module = nn.Sequential(
        #    nn.Conv2d(in_channels, planes[0], kernel_size=5, stride=1, padding=2, bias=False),
        #    nn.InstanceNorm2d(planes[0], affine=True), nn.ReLU(), nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        #    simple_resnet(dimension, num_blocks, planes,
        #                  transpose=False, norm='InstanceNorm', activation='ReLU', pool=False),
        #    nn.Flatten(start_dim=1), nn.Linear(planes[-1]*7*7, content_dim))
        #style_encoder: nn.Module = nn.Sequential(
        #    nn.Conv2d(in_channels, planes[0], kernel_size=5, stride=1, padding=2, bias=False),
        #    nn.InstanceNorm2d(planes[0], affine=True), nn.ReLU(), nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        #    simple_resnet(dimension, num_blocks, planes,
        #                  transpose=False, norm='InstanceNorm', activation='ReLU', pool=False),
        #    nn.Flatten(start_dim=1), nn.Linear(planes[-1]*7*7, style_dim))
        #decoder: nn.Module = nn.Sequential(
        #    nn.Linear(latent_dim, planes[-1]*7*7), View((-1, planes[-1], 7, 7)),
        #    simple_resnet(dimension, num_blocks, planes,
        #                  transpose=True, norm='InstanceNorm', activation='ReLU', pool=False),
        #    nn.ConvTranspose2d(planes[0], planes[0], kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
        #    nn.InstanceNorm2d(planes[0], affine=True), nn.ReLU(),
        #    nn.ConvTranspose2d(planes[0], in_channels, kernel_size=5, stride=1, padding=2, output_padding=0),
        #    nn.Tanh())
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
            nn.Linear(latent_dim, planes[-1]*7*7), View((-1, planes[-1], 7, 7)),
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
        source_disc: nn.Module = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False), nn.InstanceNorm2d(64, affine=True), nn.LeakyReLU(0.01),
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1, bias=False), nn.InstanceNorm2d(64, affine=True), nn.LeakyReLU(0.01),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False), nn.InstanceNorm2d(128, affine=True), nn.LeakyReLU(0.01),
            nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1, bias=False), nn.InstanceNorm2d(128, affine=True), nn.LeakyReLU(0.01),
            nn.Flatten(), nn.Linear(7*7*128, 1, bias=False))
        reference_disc: nn.Module = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False), nn.InstanceNorm2d(64, affine=True), nn.LeakyReLU(0.01),
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1, bias=False), nn.InstanceNorm2d(64, affine=True), nn.LeakyReLU(0.01),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False), nn.InstanceNorm2d(128, affine=True), nn.LeakyReLU(0.01),
            nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1, bias=False), nn.InstanceNorm2d(128, affine=True), nn.LeakyReLU(0.01),
            nn.Flatten(), nn.Linear(7*7*128, 1, bias=False))
        content_seg_disc: nn.Module = None
        style_seg_disc: nn.Module = None

        super().__init__(device, content_encoder, style_encoder, decoder, content_disc, style_disc,
                         source_disc, reference_disc, content_seg_disc, style_seg_disc,
                         scaler)

        self._content_dim = content_dim
        self._style_dim = style_dim
        self.apply(weights_init_resnet)

    def _cs_to_latent(self, c: Tensor, s: Tensor = None) -> Tensor:
        #z: Tensor = c + s if s is not None else c  # Addition
        if s is None:
            s = torch.zeros((len(c), self._style_dim, 1, 1)).to(self._device)
        z: Tensor = torch.cat([c, s], dim=1)
        return z


class CSDoubleEconderBeautyganModel(CSDoubleEncoderModel):
    def __init__(self, device) -> None:
        dimension = 2
        in_channels = 3
        out_channels = 3
        content_dim = 256
        style_dim = 256
        latent_dim = content_dim + style_dim
        num_blocks = [4, 4]
        planes = [64, 128, 256]

        content_encoder: nn.Module = nn.Sequential(
            nn.Conv2d(in_channels, planes[0], kernel_size=7, stride=1, padding=3, bias=False),
            nn.InstanceNorm2d(planes[0], affine=True), nn.LeakyReLU(0.01), nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            simple_resnet(dimension, num_blocks, planes,
                          transpose=False, norm='InstanceNorm', activation='LeakyReLU', pool=False),
            Permute((0, 2, 3, 1)), nn.Linear(planes[-1], content_dim))
        style_encoder: nn.Module = nn.Sequential(
            nn.Conv2d(in_channels, planes[0], kernel_size=7, stride=1, padding=3, bias=False),
            nn.InstanceNorm2d(planes[0], affine=True), nn.LeakyReLU(0.01), nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            simple_resnet(dimension, num_blocks, planes,
                          transpose=False, norm='InstanceNorm', activation='LeakyReLU', pool=False),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(planes[-1], style_dim))
        decoder: nn.Module = nn.Sequential(
            nn.Linear(latent_dim, planes[-1]), Permute((0, 3, 1, 2)),
            simple_resnet(dimension, num_blocks, planes,
                          transpose=True, norm='InstanceNorm', activation='LeakyReLU', pool=False),
            nn.ConvTranspose2d(planes[0], planes[0], kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.InstanceNorm2d(planes[0], affine=True), nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(planes[0], out_channels, kernel_size=7, stride=1, padding=3, output_padding=0),
            nn.Tanh())
        content_disc: nn.Module = nn.Sequential(
            Permute((0, 3, 1, 2)),
            nn.Conv2d(content_dim, 256, kernel_size=4, stride=2, padding=1, bias=True), nn.InstanceNorm2d(256, affine=True), nn.LeakyReLU(0.01),
            nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1, bias=True), nn.InstanceNorm2d(256, affine=True), nn.LeakyReLU(0.01),
            nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1, bias=True), nn.InstanceNorm2d(256, affine=True), nn.LeakyReLU(0.01),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(256, 1, bias=True))
        style_disc: nn.Module = None
        #style_disc: nn.Module = nn.Sequential(
        #    nn.Linear(style_dim, 256, bias=False), nn.LayerNorm(256), nn.LeakyReLU(0.01),
        #    nn.Linear(256, 64, bias=False), nn.LayerNorm(64), nn.LeakyReLU(0.01),
        #    nn.Linear(64, 1, bias=True))

        source_disc: nn.Module = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1, bias=False), nn.InstanceNorm2d(64, affine=True), nn.LeakyReLU(0.01),
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1, bias=False), nn.InstanceNorm2d(64, affine=True), nn.LeakyReLU(0.01),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False), nn.InstanceNorm2d(128, affine=True), nn.LeakyReLU(0.01),
            nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1, bias=False), nn.InstanceNorm2d(128, affine=True), nn.LeakyReLU(0.01),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False), nn.InstanceNorm2d(256, affine=True), nn.LeakyReLU(0.01),
            nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1, bias=False), nn.InstanceNorm2d(256, affine=True), nn.LeakyReLU(0.01),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(256, 1, bias=True))
        reference_disc: nn.Module = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1, bias=False), nn.InstanceNorm2d(64, affine=True), nn.LeakyReLU(0.01),
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1, bias=False), nn.InstanceNorm2d(64, affine=True), nn.LeakyReLU(0.01),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False), nn.InstanceNorm2d(128, affine=True), nn.LeakyReLU(0.01),
            nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1, bias=False), nn.InstanceNorm2d(128, affine=True), nn.LeakyReLU(0.01),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False), nn.InstanceNorm2d(256, affine=True), nn.LeakyReLU(0.01),
            nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1, bias=False), nn.InstanceNorm2d(256, affine=True), nn.LeakyReLU(0.01),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(256, 1, bias=True))

        content_seg_disc: nn.Module = None
        #content_seg_disc: nn.Module = nn.Sequential(
        #    Permute((0, 3, 1, 2)),
        #    nn.Conv2d(latent_dim, 256, kernel_size=3, stride=1, padding=1, bias=True), nn.InstanceNorm2d(256, affine=True), nn.LeakyReLU(0.01),
        #    nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True), nn.InstanceNorm2d(256, affine=True), nn.LeakyReLU(0.01),
        #    nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True), nn.InstanceNorm2d(128, affine=True), nn.LeakyReLU(0.01),
        #    nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True), nn.InstanceNorm2d(128, affine=True), nn.LeakyReLU(0.01),
        #    nn.Conv2d(128, 17, kernel_size=7, stride=1, padding=3, bias=True))
        style_seg_disc: nn.Module = nn.Sequential(
            nn.Linear(style_dim, 32*32, bias=True), View((-1, 1, 32, 32)),
            nn.Conv2d(1, 256, kernel_size=3, stride=1, padding=1, bias=True), nn.InstanceNorm2d(256, affine=True), nn.LeakyReLU(0.01),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True), nn.InstanceNorm2d(256, affine=True), nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True), nn.InstanceNorm2d(128, affine=True), nn.LeakyReLU(0.01),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True), nn.InstanceNorm2d(128, affine=True), nn.LeakyReLU(0.01),
            nn.Conv2d(128, 17, kernel_size=7, stride=1, padding=3, bias=True))

        scaler: Scaler = Scaler(2., 0.5)

        super().__init__(device, content_encoder, style_encoder, decoder, content_disc, style_disc,
                         source_disc, reference_disc, content_seg_disc, style_seg_disc,
                         scaler)

        self._content_dim = content_dim
        self._style_dim = style_dim
        self.apply(weights_init_resnet)

    def _cs_to_latent(self, c: Tensor, s: Tensor = None) -> Tensor:
        if s is None:
            s = torch.zeros((len(c), self._style_dim)).to(self._device)
        #z: Tensor = c + s
        z: Tensor = torch.cat([c, s.view((-1, 1, 1, self._style_dim)).expand((-1, 32, 32, -1))], dim=-1)
        return z

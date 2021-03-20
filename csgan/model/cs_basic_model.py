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


class CSBasicModel(BaseModel):
    def __init__(self, device, content_encoder: nn.Module, decoder: nn.Module,
                 content_disc: nn.Module, source_disc: nn.Module, scaler: Scaler = None) -> None:
        super().__init__(device)
        self._content_encoder: nn.Module = content_encoder
        self._decoder: nn.Module = decoder

        self._content_disc: nn.Module = content_disc
        self._source_disc: nn.Module = source_disc

        if scaler is None:
            scaler = Scaler(1., 0.)
        self._scaler = scaler

        self._identity_criterion = nn.L1Loss()
        self._content_criterion = nn.BCEWithLogitsLoss()
        #self._content_criterion = nn.MSELoss()

        self._source_criterion = nn.BCEWithLogitsLoss()
        #self._source_criterion = nn.MSELoss()

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
                            module in [self._content_encoder, self._decoder, self._content_disc, self._source_disc]]
        self._schedulers = [optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma) if
                            optimizer is not None else None for optimizer in self._optimizers]

    def _cs_to_latent(self, c: Tensor, s: Tensor = None) -> Tensor:
        return c

    def _predict(self, batch: Dict[str, Tensor], **kwargs) -> Dict[str, Tensor]:
        if 'x' in batch.keys():  # Style-content separate
            xp: Tensor = self._scaler.scaling(batch['x'])
            c: Tensor = self._content_encoder(xp)
            z: Tensor = self._cs_to_latent(c)
            output: Dict[str, Tensor] = {'z': z, 'c': c}
        else:
            assert('x1' in batch.keys() and 'x2' in batch.keys())  # Style change
            xp1: Tensor = self._scaler.scaling(batch['x1'])
            xp2: Tensor = self._scaler.scaling(batch['x2'])
            c1: Tensor = self._content_encoder(xp1)
            c2: Tensor = self._content_encoder(xp2)
            z1: Tensor = self._cs_to_latent(c1)
            z2: Tensor = self._cs_to_latent(c2)
            x1_idt: Tensor = self._scaler.unscaling(self._decoder(self._cs_to_latent(c1)))
            x21: Tensor = self._scaler.unscaling(self._decoder(self._cs_to_latent(c2)))
            output: Dict[str, Tensor] = {'z1': z1, 'z2': z2, 'c1': c1, 'c2': c2,
                                         'x1_idt': x1_idt, 'x21': x21}
        return output

    def _post_processing(self, batch: Dict[str, Tensor], params: Dict[str, Any],
                         global_step: int = 0) -> None:
        if global_step == 1 or global_step % params['sampling_interval'] == 0:
            self.eval()
            with torch.no_grad():
                output: Dict[str, Tensor] = self._predict(batch)
            result_dir = os.path.join(params['run_dir'], 'results')

            m: int = 4
            n: int = min(6, len(batch['x1']))
            fig = plt.figure(figsize=(10., 10.*float(n)/4.))
            for index in range(n):
                ax = fig.add_subplot(n, m, m*index+1)
                ax.imshow(batch['x1'][index, :3].detach().cpu().numpy().transpose(1, 2, 0))
                ax = fig.add_subplot(n, m, m*index+2)
                ax.imshow(batch['x2'][index, :3].detach().cpu().numpy().transpose(1, 2, 0))
                ax = fig.add_subplot(n, m, m*index+3)
                ax.imshow(output['x1_idt'][index].detach().cpu().numpy().transpose(1, 2, 0))
                ax = fig.add_subplot(n, m, m*index+4)
                ax.imshow(output['x21'][index].detach().cpu().numpy().transpose(1, 2, 0))

            plt.savefig(os.path.join(result_dir, f"sampling_{global_step}.png"), dpi=200, bbox_inches='tight')
            plt.close('all')

    def forward(self, batch: Dict[str, Tensor], params: Dict[str, Any],
                global_step: int = 0, **kwargs) -> Dict[str, Tensor]:
        output: Dict[str, Tensor] = {}
        return output

    def loss_function(self, batch: Dict[str, Tensor], output: Dict[str, Tensor], params: Dict[str, Any],
                      global_step: int = 0, **kwargs) -> Dict[str, Tensor]:
        (optimizer_content_encoder, optimizer_decoder, optimizer_content_disc, optimizer_source_disc) = self._optimizers

        # 0. Parameters
        lambda_idt: float = params['lambda_identity']
        lambda_content: float = params['lambda_content']
        lambda_source: float = params['lambda_source']

        gamma_content = params['gamma_content']
        gamma_source = params['gamma_source']

        xp1: Tensor = self._scaler.scaling(batch['x1']).detach()
        xp2: Tensor = self._scaler.scaling(batch['x2']).detach()

        # 1. Disc Loss

        with torch.no_grad():
            c1_detach: Tensor = self._content_encoder(xp1).detach()
            c2_detach: Tensor = self._content_encoder(xp2).detach()

        # 1-1. Content Disc Loss

        loss_content: Tensor = torch.FloatTensor([0.])[0].to(self._device)
        accuracy_content: Tensor = torch.FloatTensor([0.5])[0].to(self._device)
        if lambda_content > 0:
            b1_content: Tensor = self._content_disc(c1_detach)
            b2_content: Tensor = self._content_disc(c2_detach)

            loss_content: Tensor = lambda_content * (self._content_criterion(b1_content, torch.ones_like(b1_content)) +
                                                     self._content_criterion(b2_content, torch.zeros_like(b2_content))) / 2.
            if isinstance(self._content_criterion, nn.BCEWithLogitsLoss):
                correct1: Tensor = b1_content >= 0.
                correct2: Tensor = b2_content < 0.
            else:
                assert(isinstance(self._content_criterion, nn.MSELoss))
                correct1: Tensor = b1_content >= 0.5
                correct2: Tensor = b2_content < 0.5
            accuracy_content: Tensor = (correct1.sum() + correct2.sum()) / float(len(b1_content) + len(b2_content))

            optimizer_content_disc.zero_grad()
            loss_content.backward()
            optimizer_content_disc.step()

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
            if isinstance(self._source_criterion, nn.BCEWithLogitsLoss):
                correct1: Tensor = b1_source >= 0.
                correct2: Tensor = b2_source < 0.
            else:
                assert(isinstance(self._source_criterion, nn.MSELoss))
                correct1: Tensor = b1_source >= 0.5
                correct2: Tensor = b2_source < 0.5
            accuracy_source: Tensor = (correct1.sum() + correct2.sum()) / float(len(b1_source) + len(b2_source))

            optimizer_source_disc.zero_grad()
            loss_source.backward()
            optimizer_source_disc.step()

        # 2. Encoder Loss

        for module in [self._content_disc, self._source_disc]:
            if module is not None:
                module.requires_grad_(False)

        c1: Tensor = self._content_encoder(xp1)
        c2: Tensor = self._content_encoder(xp2)

        # 2-1. Content Encoder Loss

        loss_content_encoder: Tensor = torch.FloatTensor([0.])[0].to(self._device)
        if lambda_content > 0:
            b2_content: Tensor = self._content_disc(c2)
            loss_content_encoder: Tensor = lambda_content * gamma_content * (
                self._content_criterion(b2_content, torch.ones_like(b2_content)))

        # 2-3. Source Encoder Loss

        loss_source_encoder: Tensor = torch.FloatTensor([0.])[0].to(self._device)
        if lambda_source > 0:
            #xp21_detach: Tensor = self._decoder(self._cs_to_latent(c2).detach())
            #b2_source: Tensor = self._source_disc(xp21_detach)
            xp21: Tensor = self._decoder(self._cs_to_latent(c2))
            b2_source: Tensor = self._source_disc(xp21)

            loss_source_encoder: Tensor = lambda_source * gamma_source * (
                    self._source_criterion(b2_source, torch.ones_like(b2_source))) / 2.

        # 2-7. Identity Loss

        loss_idt: Tensor = torch.FloatTensor([0.])[0].to(self._device)
        if lambda_idt > 0:
            xp1_idt: Tensor = self._decoder(self._cs_to_latent(c1))
            loss_idt: Tensor = lambda_idt * self._identity_criterion(xp1_idt, xp1[:, :3])

        loss: Tensor = (loss_content_encoder + loss_source_encoder + loss_idt)

        optimizer_content_encoder.zero_grad()
        optimizer_decoder.zero_grad()
        loss.backward()
        optimizer_content_encoder.step()
        optimizer_decoder.step()

        for module in [self._content_disc, self._source_disc]:
            if module is not None:
                module.requires_grad_(True)

        loss_dict: Dict[str, Tensor] = {'loss': loss,
                                        'loss_identity': loss_idt,
                                        'loss_content': loss_content, 'accuracy_content': accuracy_content,
                                        'loss_source': loss_source, 'accuracy_source': accuracy_source}
        return loss_dict


class CSBasicMnistModel(CSBasicModel):
    def __init__(self, device) -> None:
        dimension = 2
        in_channels = 3
        content_dim = 512
        latent_dim = content_dim
        #style_dim = 64
        #latent_dim = content_dim + style_dim
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
        scaler: Scaler = Scaler(2., 0.5)
        #source_disc: nn.Module = nn.Sequential(
        #    nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1, bias=False), nn.InstanceNorm2d(64, affine=True), nn.LeakyReLU(0.01),
        #    nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False), nn.InstanceNorm2d(128, affine=True), nn.LeakyReLU(0.01),
        #    nn.Flatten(), nn.Linear(7*7*128, 1, bias=True))
        source_disc: nn.Module = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False), nn.InstanceNorm2d(64, affine=True), nn.LeakyReLU(0.01),
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1, bias=False), nn.InstanceNorm2d(64, affine=True), nn.LeakyReLU(0.01),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False), nn.InstanceNorm2d(128, affine=True), nn.LeakyReLU(0.01),
            nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1, bias=False), nn.InstanceNorm2d(128, affine=True), nn.LeakyReLU(0.01),
            nn.Flatten(), nn.Linear(7*7*128, 1, bias=True))

        super().__init__(device, content_encoder, decoder, content_disc, source_disc, scaler)

        self._content_dim = content_dim
        self.apply(weights_init_resnet)


class CSBasicBeautyganModel(CSBasicModel):
    def __init__(self, device) -> None:
        dimension = 2
        in_channels = 3
        out_channels = 3
        content_dim = 256
        latent_dim = content_dim
        #style_dim = 32
        #latent_dim = content_dim + style_dim
        num_blocks = [4, 4]
        planes = [64, 128, 256]

        content_encoder: nn.Module = nn.Sequential(
            nn.Conv2d(in_channels, planes[0], kernel_size=7, stride=1, padding=3, bias=False),
            nn.InstanceNorm2d(planes[0], affine=True), nn.LeakyReLU(0.01), nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            simple_resnet(dimension, num_blocks, planes,
                          transpose=False, norm='InstanceNorm', activation='LeakyReLU', pool=False),
            Permute((0, 2, 3, 1)), nn.Linear(planes[-1], content_dim))
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
        source_disc: nn.Module = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1, bias=False), nn.InstanceNorm2d(64, affine=True), nn.LeakyReLU(0.01),
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1, bias=False), nn.InstanceNorm2d(64, affine=True), nn.LeakyReLU(0.01),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False), nn.InstanceNorm2d(128, affine=True), nn.LeakyReLU(0.01),
            nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1, bias=False), nn.InstanceNorm2d(128, affine=True), nn.LeakyReLU(0.01),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False), nn.InstanceNorm2d(256, affine=True), nn.LeakyReLU(0.01),
            nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1, bias=False), nn.InstanceNorm2d(256, affine=True), nn.LeakyReLU(0.01),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(256, 1, bias=True))

        scaler: Scaler = Scaler(2., 0.5)

        super().__init__(device, content_encoder, decoder, content_disc, source_disc, scaler)

        self._content_dim = content_dim
        self.apply(weights_init_resnet)

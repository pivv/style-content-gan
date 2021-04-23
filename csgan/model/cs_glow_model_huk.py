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
from csgan.deep.loss import ListMSELoss, SiameseLoss, L1SiameseLoss, BinaryEntropyWithLogitsLoss
from csgan.deep.layer import View, Permute
from csgan.deep.grad import grad_scale, grad_reverse
from csgan.deep.resnet import simple_resnet, simple_bottleneck_resnet
from csgan.deep.initialize import weights_init_resnet
from csgan.deep.norm import spectral_norm
from csgan.deep.glow.model import Glow


class CSGlowModel(BaseModel):
    def __init__(self, device,
                 glow: Glow, style_w: nn.Module, content_disc: nn.Module, compatible_disc: nn.Module,
                 scaler: Scaler = None) -> None:
        super().__init__(device)
        self._glow: Glow = glow
        self._style_w: nn.Module = style_w
        self._content_disc: nn.Module = content_disc
        self._compatible_disc: nn.Module = compatible_disc

        if scaler is None:
            scaler = Scaler(1., 0.)
        self._scaler = scaler

        self._weight_cycle_criterion = ListMSELoss()

        self._content_criterion = nn.BCEWithLogitsLoss()
        self._compatible_criterion = nn.BCEWithLogitsLoss()

        self._siamese_criterion = SiameseLoss()

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
                            module in [self._glow, self._style_w,
                                       self._content_disc, self._compatible_disc]]
        self._schedulers = [optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma) if
                            optimizer is not None else None for optimizer in self._optimizers]

    @abstractmethod
    def _latent_to_cs(self, z: List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]:
        s: List[Tensor] = self._style_w(z)
        c: List[Tensor] = [z_one - s_one for s_one, z_one in zip(s, z)]
        return c, s

    @abstractmethod
    def _cs_to_latent(self, c: List[Tensor], s: List[Tensor] = None) -> List[Tensor]:
        if s is None:
            return c
        return [c_one + s_one for c_one, s_one in zip(c, s)]

    def _predict(self, batch: Dict[str, Tensor], **kwargs) -> Dict[str, Tensor]:
        if 'x' in batch.keys():  # Style-content separate
            xp: Tensor = self._scaler.scaling(batch['x'])
            z: List[Tensor] = self._glow(xp)
            c, s = self._latent_to_cs(z)
            output: Dict[str, List[Tensor]] = {'z': z, 's': s, 'c': c}
        else:
            assert('x1' in batch.keys() and 'x2' in batch.keys())  # Style change
            xp1: Tensor = self._scaler.scaling(batch['x1'])
            xp2: Tensor = self._scaler.scaling(batch['x2'])
            z1: List[Tensor] = self._glow(xp1)
            z2: List[Tensor] = self._glow(xp2)
            c1, s1 = self._latent_to_cs(z1)
            c2, s2 = self._latent_to_cs(z2)
            x1_idt: Tensor = self._scaler.unscaling(self._glow.reverse(self._cs_to_latent(c1), reconstruct=True))
            x2_idt: Tensor = self._scaler.unscaling(self._glow.reverse(self._cs_to_latent(c2, s2), reconstruct=True))
            xp12: Tensor = self._glow.reverse(self._cs_to_latent(c1, s2), reconstruct=True)
            xp21: Tensor = self._glow.reverse(self._cs_to_latent(c2), reconstruct=True)
            x12: Tensor = self._scaler.unscaling(xp12)
            x21: Tensor = self._scaler.unscaling(xp21)
            x1_cycle: Tensor = self._scaler.unscaling(self._glow.reverse(self._cs_to_latent(self._glow(xp12)),
                                                                         reconstruct=True))
            x2_cycle: Tensor = self._scaler.unscaling(self._glow.reverse(self._cs_to_latent(self._glow(xp21), s2),
                                                                         reconstruct=True))
            output: Dict[str, Tensor or List[Tensor]] = {'z1': z1, 'z2': z2, 's1': s1, 's2': s2, 'c1': c1, 'c2': c2,
                                                         'x1_idt': x1_idt, 'x2_idt': x2_idt, 'x12': x12, 'x21': x21,
                                                         'x1_cycle': x1_cycle, 'x2_cycle': x2_cycle}
        return output

    def _post_processing(self, batch: Dict[str, Tensor], params: Dict[str, Any],
                         global_step: int = 0) -> None:
        if global_step == 1 or global_step % params['sampling_interval'] == 0:
            self.eval()
            with torch.no_grad():
                output: Dict[str, Tensor or List[Tensor]] = self._predict(batch)
            result_dir = os.path.join(params['run_dir'], 'results')

            n: int = min(8, len(batch['x1']))
            fig = plt.figure(figsize=(20., 20.*float(n)/8.))
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
        (optimizer_glow, optimizer_style_w,
         optimizer_content_disc, optimizer_compatible_disc) = self._optimizers

        # 0. Parameters
        lambda_glow: float = params['lambda_glow']
        lambda_weight_cycle: float = params['lambda_weight_cycle']
        lambda_content: float = params['lambda_content']
        lambda_siamese: float = params['lambda_siamese']
        lambda_compatible: float = params['lambda_compatible']

        gamma_content: float = params['gamma_content']
        gamma_compatible: float = params['gamma_compatible']

        xp1: Tensor = self._scaler.scaling(batch['x1']).detach()
        xp2: Tensor = self._scaler.scaling(batch['x2']).detach()

        # 1. Disc Loss

        with torch.no_grad():
            z1: List[Tensor] = self._glow(xp1)
            z2: List[Tensor] = self._glow(xp2)
            z1_detach: List[Tensor] = [z1_one.detach() for z1_one in z1]
            z2_detach: List[Tensor] = [z2_one.detach() for z2_one in z2]

        c1, s1 = self._latent_to_cs(z1_detach)
        c2, s2 = self._latent_to_cs(z2_detach)
        c1_detach: List[Tensor] = [c1_one.detach() for c1_one in c1]
        s1_detach: List[Tensor] = [s1_one.detach() for s1_one in s1]
        c2_detach: List[Tensor] = [c2_one.detach() for c2_one in c2]
        s2_detach: List[Tensor] = [s2_one.detach() for s2_one in s2]

        # 1-1. Content Disc Loss

        loss_content: Tensor = torch.FloatTensor([0.])[0].to(self._device)
        accuracy_content: Tensor = torch.FloatTensor([0.5])[0].to(self._device)
        if lambda_content > 0:
            b1_content: Tensor = self._content_disc(c1_detach)
            b2_content: Tensor = self._content_disc(c2_detach)

            loss_content = lambda_content * (self._content_criterion(b1_content, torch.ones_like(b1_content)) +
                                             self._content_criterion(b2_content, torch.zeros_like(b2_content))) / 2.
            if isinstance(self._content_criterion, nn.BCEWithLogitsLoss):
                correct1: Tensor = b1_content >= 0.
                correct2: Tensor = b2_content < 0.
            else:
                assert(isinstance(self._content_criterion, nn.MSELoss))
                correct1: Tensor = b1_content >= 0.5
                correct2: Tensor = b2_content < 0.5
            accuracy_content = (correct1.sum() + correct2.sum()) / float(len(b1_content) + len(b2_content))

            optimizer_content_disc.zero_grad()
            loss_content.backward()
            optimizer_content_disc.step()

        # 1-2. Compatibility Loss

        loss_compatible: Tensor = torch.FloatTensor([0.])[0].to(self._device)
        accuracy_compatible: Tensor = torch.FloatTensor([0.5])[0].to(self._device)
        if lambda_compatible > 0:
            with torch.no_grad():
                z12_detach: List[Tensor] = self._cs_to_latent(c1_detach, s2_detach)
                #z21_detach: List[Tensor] = self._cs_to_latent(c2_detach)
                z21_detach: List[Tensor] = self._cs_to_latent(c2_detach, s1_detach)
            b1_compatible: Tensor = self._compatible_disc(z1_detach)
            b2_compatible: Tensor = self._compatible_disc(z2_detach)
            b12_compatible: Tensor = self._compatible_disc(z12_detach)
            b21_compatible: Tensor = self._compatible_disc(z21_detach)

            loss_compatible = lambda_compatible * (
                    self._compatible_criterion(b1_compatible, torch.ones_like(b1_compatible)) +
                    self._compatible_criterion(b2_compatible, torch.ones_like(b2_compatible)) +
                    self._compatible_criterion(b12_compatible, torch.zeros_like(b12_compatible)) +
                    self._compatible_criterion(b21_compatible, torch.zeros_like(b21_compatible))) / 4.
            if isinstance(self._compatible_criterion, nn.BCEWithLogitsLoss):
                correct1: Tensor = b1_compatible >= 0.
                correct2: Tensor = b2_compatible >= 0.
                correct12: Tensor = b12_compatible < 0.
                correct21: Tensor = b21_compatible < 0.
            else:
                assert(isinstance(self._compatible_criterion, nn.MSELoss))
                correct1: Tensor = b1_compatible >= 0.5
                correct2: Tensor = b2_compatible >= 0.5
                correct12: Tensor = b12_compatible < 0.5
                correct21: Tensor = b21_compatible < 0.5
            accuracy_compatible = (correct1.sum() + correct2.sum() + correct12.sum() + correct21.sum()) / float(
                len(b1_compatible) + len(b2_compatible) + len(b12_compatible) + len(b21_compatible))

            optimizer_compatible_disc.zero_grad()
            loss_compatible.backward()
            optimizer_compatible_disc.step()

        # 2. Weight Cycle Loss

        c1_idt, s2_idt = self._latent_to_cs(self._cs_to_latent(c1, s2))
        c2_idt, s1_idt = self._latent_to_cs(self._cs_to_latent(c2, s1))
        loss_weight_cycle: Tensor = torch.FloatTensor([0.])[0].to(self._device)
        if lambda_weight_cycle > 0:
            loss_weight_cycle = lambda_weight_cycle * (
                    self._weight_cycle_criterion(c1_idt, c1_detach) + self._weight_cycle_criterion(c2_idt, c2_detach) +
                    self._weight_cycle_criterion(s1_idt, s1_detach) + self._weight_cycle_criterion(s2_idt, s2_detach)) / 4.

        optimizer_style_w.zero_grad()
        loss_weight_cycle.backward()
        optimizer_style_w.step()

        for module in [self._content_disc, self._compatible_disc]:
            if module is not None:
                module.requires_grad_(False)

        # 3. Encoder Loss

        z1, log_p1, log_det1 = self._glow.forward_with_loss(xp1)
        z2, log_p2, log_det2 = self._glow.forward_with_loss(xp2)

        c1, s1 = self._latent_to_cs(z1)
        c2, s2 = self._latent_to_cs(z2)

        # 3-1. Glow Loss

        loss_glow: Tensor = torch.FloatTensor([0.])[0].to(self._device)
        if lambda_glow > 0:
            loss_glow1, _, _ = self._glow.loss_function(z1, log_p1, log_det1)
            loss_glow2, _, _ = self._glow.loss_function(z2, log_p2, log_det2)
            loss_glow = lambda_glow * (loss_glow1 + loss_glow2) / 2.

        # 3-2. Content Encoder Loss

        loss_content_encoder: Tensor = torch.FloatTensor([0.])[0].to(self._device)
        if lambda_content > 0:
            b2_content: Tensor = self._content_disc(c2)
            loss_content_encoder = lambda_content * gamma_content * (
                self._content_criterion(b2_content, torch.ones_like(b2_content)))

        # 3-3. Compatibility Encoder Loss

        loss_compatible_encoder: Tensor = torch.FloatTensor([0.])[0].to(self._device)
        if lambda_compatible > 0:
            z12: List[Tensor] = self._cs_to_latent(c1_detach, s2)
            #z21: List[Tensor] = self._cs_to_latent(c2_detach)
            z21: List[Tensor] = self._cs_to_latent(c2_detach, s1)
            b12_compatible: Tensor = self._compatible_disc(z12)
            b21_compatible: Tensor = self._compatible_disc(z21)

            loss_compatible_encoder = lambda_compatible * gamma_compatible * (
                    self._compatible_criterion(b12_compatible, torch.ones_like(b12_compatible)) +
                    self._compatible_criterion(b21_compatible, torch.ones_like(b21_compatible))) / 2.

        # 3-4. Siamese Loss

        loss_siamese: Tensor = torch.FloatTensor([0.])[0].to(self._device)
        squared_norm_s1: Tensor = sum((s1_one * s1_one).flatten(start_dim=1).sum(dim=1) for s1_one in s1) / sum(
            s1_one.flatten(start_dim=1).size(1) for s1_one in s1)
        squared_norm_s2: Tensor = sum((s2_one * s2_one).flatten(start_dim=1).sum(dim=1) for s2_one in s2) / sum(
            s2_one.flatten(start_dim=1).size(1) for s2_one in s2)
        if lambda_siamese > 0:
            loss_siamese = lambda_siamese * squared_norm_s1.mean()
        norm_s1: Tensor = torch.sqrt(squared_norm_s1).mean()
        norm_s2: Tensor = torch.sqrt(squared_norm_s2).mean()

        loss: Tensor = (loss_glow + loss_content_encoder + loss_compatible_encoder + loss_siamese)

        optimizer_glow.zero_grad()
        optimizer_style_w.zero_grad()
        loss.backward()
        optimizer_glow.step()
        optimizer_style_w.step()

        for module in [self._content_disc, self._compatible_disc]:
            if module is not None:
                module.requires_grad_(True)

        loss_dict: Dict[str, Tensor] = {'loss': loss, 'loss_glow': loss_glow, 'loss_weight_cycle': loss_weight_cycle,
                                        'loss_content': loss_content, 'accuracy_content': accuracy_content,
                                        'loss_compatible': loss_compatible, 'accuracy_compatible': accuracy_compatible,
                                        'loss_siamese': loss_siamese, 'norm_s1': norm_s1, 'norm_s2': norm_s2}
        return loss_dict


class BlockwiseWeight(nn.Module):
    def __init__(self, img_size: int, in_channel: int, n_block: int):
        super().__init__()
        self._conv_list = nn.ModuleList()
        cur_in_channel: int = in_channel
        for iblock in range(n_block):
            if iblock < n_block - 1:
                cur_in_channel *= 2
            else:
                cur_in_channel *= 4
            pad: int = 2 ** (n_block - 1 - iblock)
            conv = nn.Conv2d(cur_in_channel, cur_in_channel, kernel_size=pad*2+1, stride=1, padding=pad, bias=False)
            self._conv_list.append(conv)

    def forward(self, z: List[Tensor]) -> List[Tensor]:
        s: List[Tensor] = [conv(z_one) for conv, z_one in zip(self._conv_list, z)]
        return s


class PyramidDiscriminator(nn.Module):
    def __init__(self, img_size: int, in_channel: int, n_block: int):
        super().__init__()
        self._layer_list = nn.ModuleList()
        cur_in_channel: int = in_channel
        last_in_channel: int = 0
        for iblock in range(n_block):
            if iblock < n_block - 1:
                cur_in_channel *= 2
            else:
                cur_in_channel *= 4
            layer = nn.Sequential(
                nn.Conv2d(last_in_channel + cur_in_channel, cur_in_channel*2, kernel_size=4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(cur_in_channel*2, affine=True), nn.LeakyReLU(0.01))
            self._layer_list.append(layer)
            last_in_channel = cur_in_channel*2

        self._final_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(last_in_channel, 1, bias=True))

    def forward(self, z: List[Tensor]) -> Tensor:
        input: Tensor = None
        for layer, z_one in zip(self._layer_list, z):
            input = torch.cat([input, z_one], dim=1) if input is not None else z_one
            input = layer(input)
        input = self._final_layer(input)
        return input


class CSGlowMnistModel(CSGlowModel):
    def __init__(self, device) -> None:
        content_dim = 512
        style_dim = content_dim
        latent_dim = content_dim
        #style_dim = 64
        #latent_dim = content_dim + style_dim

        img_size = 28
        in_channel = 3
        n_flow = 32
        n_block = 2

        glow: Glow = Glow(img_size, in_channel, n_flow, n_block, affine=True, conv_lu=True)

        style_w: nn.Module = BlockwiseWeight(img_size, in_channel, n_block)
        content_disc: nn.Module = PyramidDiscriminator(img_size, in_channel, n_block)
        compatible_disc: nn.Module = PyramidDiscriminator(img_size, in_channel, n_block)
        scaler: Scaler = Scaler(2., 0.5)

        super().__init__(device, glow, style_w, content_disc, compatible_disc, scaler)

        self._content_dim = content_dim
        self._style_dim = style_dim
        self.apply(weights_init_resnet)


import sys
import os

from typing import List, Callable, Union, Any, TypeVar, Tuple, Dict
from torch import Tensor

from abc import abstractmethod

import time

from collections import defaultdict

import itertools

import numpy as np

import cv2

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader

import torchvision

from tensorboardX import SummaryWriter

import matplotlib.pyplot as plt

from .base_model import BaseModel
from csgan.util.scaler import Scaler
from csgan.util.histogram_matching import histogram_matching
from csgan.deep.loss import SiameseLoss, BinaryEntropyWithLogitsLoss
from csgan.deep.layer import View, Permute
from csgan.deep.grad import grad_scale, grad_reverse
from csgan.deep.resnet import SimpleResidualBlock, simple_resnet, simple_bottleneck_resnet
from csgan.deep.initialize import weights_init_xavier
from csgan.deep.norm import spectral_norm


class PairedCycleGanModel(BaseModel):
    def __init__(self, device, G1: nn.Module, G2: nn.Module,
                 D1: nn.Module, D2: nn.Module, vgg: nn.Module,
                 scaler: Scaler = None) -> None:
        super().__init__(device)
        self._G1 = G1
        self._G2 = G2
        self._D1 = D1
        self._D2 = D2
        if scaler is None:
            scaler = Scaler(1., 0.)
        self._scaler = scaler
        self._vgg = vgg
        self._vgg.requires_grad_(False)
        self._vgg.eval()

        self._l1_criterion = torch.nn.L1Loss()
        self._l2_criterion = torch.nn.MSELoss()
        self._gan_criterion = torch.nn.MSELoss()

        self.to(self._device)

    def _update_optimizers(self, loss_dict: Dict[str, Tensor], params: Dict[str, Any],
                           global_step: int = 0) -> None:
        # Not do here.
        pass

    def _update_schedulers(self, params: Dict[str, Any], global_step: int = 0) -> None:
        pass

    def _set_optimizers(self, params: Dict[str, Any]) -> None:
        weight_decay: float = params['weight_decay']
        optimizer_G = optim.Adam(itertools.chain(self._G1.parameters(), self._G2.parameters()),
                                 params['learning_rate'], (0.5, 0.999), weight_decay=weight_decay)
        optimizer_D1 = optim.Adam(filter(lambda p: p.requires_grad, self._D1.parameters()),
                                  params['learning_rate'], weight_decay=weight_decay)
        optimizer_D2 = optim.Adam(filter(lambda p: p.requires_grad, self._D2.parameters()),
                                  params['learning_rate'], weight_decay=weight_decay)

        self._optimizers = [optimizer_G, optimizer_D1, optimizer_D2]
        self._schedulers = [None, None, None]

    def _predict(self, batch: Dict[str, Tensor], **kwargs) -> Dict[str, Tensor]:
        assert('x1' in batch.keys() and 'x2' in batch.keys())  # Style change
        xp1: Tensor = self._scaler.scaling(batch['x1'])
        xp2: Tensor = self._scaler.scaling(batch['x2'])
        x1_idt: Tensor = self._scaler.unscaling(self._G2(xp1))
        x2_idt: Tensor = self._scaler.unscaling(self._G1(xp2, xp2))
        xp12: Tensor = self._G1(xp1, xp2)
        xp21: Tensor = self._G2(xp2)
        x12: Tensor = self._scaler.unscaling(xp12)
        x21: Tensor = self._scaler.unscaling(xp21)
        x1_cycle: Tensor = self._scaler.unscaling(self._G2(xp12))
        x2_cycle: Tensor = self._scaler.unscaling(self._G1(xp21, xp2))
        output: Dict[str, Tensor] = {'x1_idt': x1_idt, 'x2_idt': x2_idt, 'x12': x12, 'x21': x21,
                                     'x1_cycle': x1_cycle, 'x2_cycle': x2_cycle}
        return output

    def _post_processing(self, batch: Dict[str, Tensor], params: Dict[str, Any],
                         global_step: int = 0) -> None:
        if global_step == 1 or global_step % params['sampling_interval'] == 0:
            output: Dict[str, Tensor] = self._predict(batch)
            result_dir = os.path.join(params['run_dir'], 'results')

            fig = plt.figure(figsize=(20, 15))
            for index in range(6):
                ax = fig.add_subplot(6, 8, 8*index+1)
                ax.imshow(batch['x1'][index, :3].detach().cpu().numpy().transpose(1, 2, 0))
                ax = fig.add_subplot(6, 8, 8*index+2)
                ax.imshow(batch['x2'][index, :3].detach().cpu().numpy().transpose(1, 2, 0))
                ax = fig.add_subplot(6, 8, 8*index+3)
                ax.imshow(output['x1_idt'][index].detach().cpu().numpy().transpose(1, 2, 0))
                ax = fig.add_subplot(6, 8, 8*index+4)
                ax.imshow(output['x2_idt'][index].detach().cpu().numpy().transpose(1, 2, 0))
                ax = fig.add_subplot(6, 8, 8*index+5)
                ax.imshow(output['x12'][index].detach().cpu().numpy().transpose(1, 2, 0))
                ax = fig.add_subplot(6, 8, 8*index+6)
                ax.imshow(output['x21'][index].detach().cpu().numpy().transpose(1, 2, 0))
                ax = fig.add_subplot(6, 8, 8*index+7)
                ax.imshow(output['x1_cycle'][index].detach().cpu().numpy().transpose(1, 2, 0))
                ax = fig.add_subplot(6, 8, 8*index+8)
                ax.imshow(output['x2_cycle'][index].detach().cpu().numpy().transpose(1, 2, 0))

            plt.savefig(os.path.join(result_dir, f"sampling_{global_step}.png"), dpi=200, bbox_inches='tight')
            plt.close('all')

    def forward(self, batch: Dict[str, Tensor], params: Dict[str, Any],
                global_step: int = 0, **kwargs) -> Dict[str, Tensor]:
        output: Dict[str, Tensor] = {}
        return output

    def _masked_histogram_criterion(self, xp1: Tensor, xp2: Tensor, mask1: Tensor, mask2: Tensor) -> Tensor:
        # x1: bs x ch x h x w
        loss: Tensor = torch.FloatTensor([0.])[0].to(self._device)
        for ix in range(xp1.size(0)):
            index1 = mask1[ix].nonzero()
            index2 = mask2[ix].nonzero()
            img1 = self._scaler.unscaling(xp1)[ix] * 255
            img2 = self._scaler.unscaling(xp2)[ix] * 255
            vec1 = img1[:, index1[:, 0], index1[:, 1]]
            vec2 = img2[:, index2[:, 0], index2[:, 1]]
            vec_match = histogram_matching(vec1.detach().cpu(), vec2.detach().cpu()).to(self._device)
            crit = torch.nn.L1Loss(reduction='sum')
            loss += crit(vec1, vec_match)
        return loss / xp1.numel()

    def loss_function(self, batch: Dict[str, Tensor], output: Dict[str, Tensor], params: Dict[str, Any],
                      global_step: int = 0, **kwargs) -> Dict[str, Tensor]:
        # 0. Parameters
        ndis: int = params['ndis']  # 1
        content_layer: int = params['content_layer']  # 20
        lambda_disc: float = params['lambda_disc']  # 1.
        lambda_idt: float = params['lambda_identity']  # 5.
        lambda_cycle: float = params['lambda_cycle']  # 10.
        lambda_vgg: float = params['lambda_vgg']  # 0.05
        lambda_lip: float = params['lambda_lip']  # 1.
        lambda_skin: float = params['lambda_skin']  # 0.1
        lambda_eye: float = params['lambda_eye']  # 1.

        [optimizer_G, optimizer_D1, optimizer_D2] = self._optimizers

        xp1: Tensor = self._scaler.scaling(batch['x1'])
        xp2: Tensor = self._scaler.scaling(batch['x2'])
        seg1: Tensor = batch['seg1']
        seg2: Tensor = batch['seg2']

        # ================== Train D ================== #
        with torch.no_grad():
            fake1 = self._G1(xp1, xp2)
            fake2 = self._G2(xp2)
            fake1 = fake1.detach()
            fake2 = fake2.detach()

        loss_d = torch.FloatTensor([0.])[0].to(self._device)
        loss_d1_real = torch.FloatTensor([0.])[0].to(self._device)
        loss_d2_real = torch.FloatTensor([0.])[0].to(self._device)
        if lambda_disc > 0:
            out = self._D1(xp2)
            loss_d1_real = lambda_disc * self._gan_criterion(out, torch.ones_like(out))
            out = self._D1(fake1)
            loss_d1_fake = lambda_disc * self._gan_criterion(out, torch.zeros_like(out))
            # Backward + Optimize
            loss_d1 = (loss_d1_real + loss_d1_fake) * 0.5
            optimizer_D1.zero_grad()
            loss_d1.backward()
            optimizer_D1.step()

            out = self._D2(xp1)
            loss_d2_real = lambda_disc * self._gan_criterion(out, torch.ones_like(out))
            # Fake
            out = self._D2(fake2)
            loss_d2_fake = lambda_disc * self._gan_criterion(out, torch.zeros_like(out))
            # Backward + Optimize
            loss_d2 = (loss_d2_real + loss_d2_fake) * 0.5
            optimizer_D2.zero_grad()
            loss_d2.backward()
            optimizer_D2.step()

            loss_d = loss_d1 + loss_d2

        # ================== Train G ================== #
        loss_g = torch.FloatTensor([0.])[0].to(self._device)
        loss_idt = torch.FloatTensor([0.])[0].to(self._device)
        loss_g1_fake = torch.FloatTensor([0.])[0].to(self._device)
        loss_g2_fake = torch.FloatTensor([0.])[0].to(self._device)
        loss_cycle = torch.FloatTensor([0.])[0].to(self._device)
        loss_vgg = torch.FloatTensor([0.])[0].to(self._device)
        loss_his = torch.FloatTensor([0.])[0].to(self._device)
        if global_step % ndis == 0:
            self._D1.requires_grad_(False)
            self._D2.requires_grad_(False)
            # adversarial loss, i.e. L_trans,v in the paper

            # identity loss
            if lambda_idt > 0:
                idt11 = self._G1(xp1, xp1)
                idt12 = self._G2(xp1)
                idt21 = self._G1(xp2, xp2)
                loss_idt11 = self._l1_criterion(idt11, xp1) * lambda_idt
                loss_idt12 = self._l1_criterion(idt12, xp1) * lambda_idt
                loss_idt21 = self._l1_criterion(idt21, xp2) * lambda_idt
                # loss_idt
                loss_idt = (loss_idt11 + loss_idt12 + loss_idt21) * 0.5

            if lambda_disc > 0:
                # fake1 in class B,
                fake1 = self._G1(xp1, xp2)
                fake2 = self._G2(xp2)

                out = self._D1(fake1)
                loss_g1_fake = lambda_disc * self._gan_criterion(out, torch.ones_like(out))
                # GAN loss D2(G2(B))
                out = self._D2(fake2)
                loss_g2_fake = lambda_disc * self._gan_criterion(out, torch.ones_like(out))

            # cycle loss
            if lambda_cycle > 0:
                rec2 = self._G1(fake2, fake1)
                rec1 = self._G2(fake1)
                loss_g1_cycle = self._l1_criterion(rec1, xp1) * lambda_cycle
                loss_g2_cycle = self._l1_criterion(rec2, xp2) * lambda_cycle
                loss_cycle = (loss_g1_cycle + loss_g2_cycle) * 0.5

            # vgg loss
            if lambda_vgg > 0:
                with torch.no_grad():
                    vgg_org = self._vgg.features[:content_layer+1](xp1).detach()
                vgg_fake1 = self._vgg.features[:content_layer+1](fake1)
                loss_g1_vgg = self._l2_criterion(vgg_fake1, vgg_org) * lambda_vgg

                with torch.no_grad():
                    vgg_ref = self._vgg.features[:content_layer+1](xp2).detach()
                vgg_fake2 = self._vgg.features[:content_layer+1](fake2)
                loss_g2_vgg = self._l2_criterion(vgg_fake2, vgg_ref) * lambda_vgg
                loss_vgg = (loss_g1_vgg + loss_g2_vgg) * 0.5

            # color_histogram loss
            loss_g1_his = torch.FloatTensor([0.])[0].to(self._device)
            loss_g2_his = torch.FloatTensor([0.])[0].to(self._device)

            # Convert tensor to variable
            # mask attribute: 0:background 1:face 2:left-eyebrown 3:right-eyebrown 4:left-eye 5: right-eye 6: nose
            # 7: upper-lip 8: teeth 9: under-lip 10:hair 11: left-ear 12: right-ear 13: neck 14: glasses
            # 15: left-eyeshadow 16: right-eyeshadow
            if lambda_lip > 0:
                seg1_lip: Tensor = (seg1 == 7).float() + (seg1 == 9).float()
                seg2_lip: Tensor = (seg2 == 7).float() + (seg2 == 9).float()
                assert((seg1_lip > 0).any() and (seg2_lip > 0).any())
                g1_lip_loss_his = self._masked_histogram_criterion(fake1, xp2, seg1_lip, seg2_lip) * lambda_lip
                g2_lip_loss_his = self._masked_histogram_criterion(fake2, xp1, seg2_lip, seg1_lip) * lambda_lip
                loss_g1_his += g1_lip_loss_his
                loss_g2_his += g2_lip_loss_his
            if lambda_skin > 0:
                seg1_skin: Tensor = (seg1 == 1).float() + (seg1 == 6).float() + (seg1 == 13).float()
                seg2_skin: Tensor = (seg2 == 1).float() + (seg2 == 6).float() + (seg2 == 13).float()
                assert((seg1_skin > 0).any() and (seg2_skin > 0).any())
                g1_skin_loss_his = self._masked_histogram_criterion(fake1, xp2, seg1_skin, seg2_skin) * lambda_skin
                g2_skin_loss_his = self._masked_histogram_criterion(fake2, xp1, seg2_skin, seg1_skin) * lambda_skin
                loss_g1_his += g1_skin_loss_his
                loss_g2_his += g2_skin_loss_his
            if lambda_eye > 0:
                seg1_eye_left: Tensor = (seg1 == 15).float()
                seg1_eye_right: Tensor = (seg1 == 16).float()
                seg2_eye_left: Tensor = (seg2 == 15).float()
                seg2_eye_right: Tensor = (seg2 == 16).float()
                assert ((seg1_eye_left > 0).any() and (seg2_eye_left > 0).any() and
                        (seg1_eye_right > 0).any() and (seg2_eye_right > 0).any())
                g1_eye_left_loss_his = self._masked_histogram_criterion(fake1, xp2, seg1_eye_left, seg2_eye_left) * lambda_eye
                g2_eye_left_loss_his = self._masked_histogram_criterion(fake2, xp1, seg2_eye_left, seg1_eye_left) * lambda_eye
                g1_eye_right_loss_his = self._masked_histogram_criterion(fake1, xp2, seg1_eye_right, seg2_eye_right) * lambda_eye
                g2_eye_right_loss_his = self._masked_histogram_criterion(fake2, xp1, seg2_eye_right, seg1_eye_right) * lambda_eye
                loss_g1_his += g1_eye_left_loss_his + g1_eye_right_loss_his
                loss_g2_his += g2_eye_left_loss_his + g2_eye_right_loss_his

            loss_his = loss_g1_his + loss_g2_his

            # Combined loss
            loss_g = loss_g1_fake + loss_g2_fake + loss_cycle + loss_vgg + loss_idt + loss_his

            optimizer_G.zero_grad()
            loss_g.backward()
            optimizer_G.step()

            self._D1.requires_grad_(True)
            self._D2.requires_grad_(True)

        loss_dict: Dict[str, Tensor] = {'loss_d': loss_d, 'loss_g': loss_g,
                                        'loss_d1_real': loss_d1_real, 'loss_d2_real': loss_d2_real,
                                        'loss_g1_fake': loss_g1_fake, 'loss_g2_fake': loss_g2_fake,
                                        'loss_cycle': loss_cycle, 'loss_vgg': loss_vgg,
                                        'loss_idt': loss_idt, 'loss_his': loss_his}
        return loss_dict


class GeneratorReferenceMakeup(nn.Module):
    """Generator. Encoder-Decoder Architecture."""
    # input 2 images and output 2 images as well
    def __init__(self, conv_dim=64, repeat_num=6, input_nc=3):
        super(GeneratorReferenceMakeup, self).__init__()

        # Branch input
        layers_1 = []
        layers_1.append(nn.Conv2d(input_nc, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers_1.append(nn.InstanceNorm2d(conv_dim, affine=True))
        layers_1.append(nn.ReLU(inplace=True))
        layers_1.append(nn.Conv2d(conv_dim, conv_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
        layers_1.append(nn.InstanceNorm2d(conv_dim*2, affine=True))
        layers_1.append(nn.ReLU(inplace=True))
        self.input_branch_1 = nn.Sequential(*layers_1)

        # Branch input
        layers_2 = []
        layers_2.append(nn.Conv2d(input_nc, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers_2.append(nn.InstanceNorm2d(conv_dim, affine=True))
        layers_2.append(nn.ReLU(inplace=True))
        layers_2.append(nn.Conv2d(conv_dim, conv_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
        layers_2.append(nn.InstanceNorm2d(conv_dim*2, affine=True))
        layers_2.append(nn.ReLU(inplace=True))
        self.input_branch_2 = nn.Sequential(*layers_2)

        # Down-Sampling, branch merge
        layers = []
        curr_dim = conv_dim*2
        layers.append(nn.Conv2d(curr_dim*2, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
        curr_dim = curr_dim * 2
        # Bottleneck
        for i in range(repeat_num):
            layers.append(SimpleResidualBlock(dim_in=curr_dim, dim_out=curr_dim))
        # Up-Sampling
        for i in range(2):
            layers.append(nn.InstanceNorm2d(curr_dim, affine=True))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            curr_dim = curr_dim // 2
        self.main = nn.Sequential(*layers)

        layers_1 = []
        for i in range(2):
            layers_1.append(nn.InstanceNorm2d(curr_dim, affine=True))
            layers_1.append(nn.ReLU(inplace=True))
            layers_1.append(nn.Conv2d(curr_dim, curr_dim, kernel_size=3, stride=1, padding=1, bias=False))
        layers_1.append(nn.InstanceNorm2d(curr_dim, affine=True))
        layers_1.append(nn.ReLU(inplace=True))
        layers_1.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers_1.append(nn.Tanh())
        self.output_branch_1 = nn.Sequential(*layers_1)

    def forward(self, x, y):
        input_x = self.input_branch_1(x)
        input_y = self.input_branch_2(y)
        input_fuse = torch.cat((input_x, input_y), dim=1)
        out = self.main(input_fuse)
        out_A = self.output_branch_1(out)
        #out_A = out_A + x
        return out_A


class GeneratorDeMakeup(nn.Module):
    """Generator. Encoder-Decoder Architecture."""
    # input 2 images and output 2 images as well
    def __init__(self, conv_dim=64, repeat_num=6, input_nc=3):
        super(GeneratorDeMakeup, self).__init__()

        # Branch input
        layers_1 = []
        layers_1.append(nn.Conv2d(input_nc, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers_1.append(nn.InstanceNorm2d(conv_dim, affine=True))
        layers_1.append(nn.ReLU(inplace=True))
        layers_1.append(nn.Conv2d(conv_dim, conv_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
        layers_1.append(nn.InstanceNorm2d(conv_dim*2, affine=True))
        layers_1.append(nn.ReLU(inplace=True))
        self.input_branch_1 = nn.Sequential(*layers_1)

        # Down-Sampling, branch merge
        layers = []
        curr_dim = conv_dim*2
        layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
        curr_dim = curr_dim * 2
        # Bottleneck
        for i in range(repeat_num):
            layers.append(SimpleResidualBlock(dim_in=curr_dim, dim_out=curr_dim))
        # Up-Sampling
        for i in range(2):
            layers.append(nn.InstanceNorm2d(curr_dim, affine=True))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            curr_dim = curr_dim // 2
        self.main = nn.Sequential(*layers)

        layers_1 = []
        for i in range(2):
            layers_1.append(nn.InstanceNorm2d(curr_dim, affine=True))
            layers_1.append(nn.ReLU(inplace=True))
            layers_1.append(nn.Conv2d(curr_dim, curr_dim, kernel_size=3, stride=1, padding=1, bias=False))
        layers_1.append(nn.InstanceNorm2d(curr_dim, affine=True))
        layers_1.append(nn.ReLU(inplace=True))
        layers_1.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers_1.append(nn.Tanh())
        self.output_branch_1 = nn.Sequential(*layers_1)

    def forward(self, x):
        input_x = self.input_branch_1(x)
        out = self.main(input_x)
        out_A = self.output_branch_1(out)
        #out_A = out_A + x
        return out_A


class Discriminator(nn.Module):
    """Discriminator. PatchGAN."""
    def __init__(self, image_size=128, conv_dim=64, repeat_num=3, input_nc=3, norm='SN'):
        super(Discriminator, self).__init__()

        layers = []
        if norm == 'SN':
            layers.append(spectral_norm(nn.Conv2d(input_nc, conv_dim, kernel_size=4, stride=2, padding=1)))
        else:
            layers.append(nn.Conv2d(input_nc, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01, inplace=True))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            if norm == 'SN':
                layers.append(spectral_norm(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1)))
            else:
                layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01, inplace=True))
            curr_dim = curr_dim * 2

        #k_size = int(image_size / np.power(2, repeat_num))
        if norm == 'SN':
            layers.append(spectral_norm(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=1, padding=1)))
        else:
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=1, padding=1))
        layers.append(nn.LeakyReLU(0.01, inplace=True))
        curr_dim = curr_dim * 2

        self.main = nn.Sequential(*layers)

        if norm == 'SN':
            self.conv1 = spectral_norm(nn.Conv2d(curr_dim, 1, kernel_size=4, stride=1, padding=1, bias=False))
        else:
            self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=4, stride=1, padding=1, bias=False)

        # conv1 remain the last square size, 256*256-->30*30
        #self.conv2 = SpectralNorm(nn.Conv2d(curr_dim, 1, kernel_size=k_size, bias=False))
        #conv2 output a single number

    def forward(self, x):
        h = self.main(x)
        #out_real = self.conv1(h)
        out_makeup = self.conv1(h)
        #return out_real.squeeze(), out_makeup.squeeze()
        return out_makeup.squeeze()


class PairedCycleGanTestModel(PairedCycleGanModel):
    def __init__(self, device) -> None:
        G1: nn.Module = GeneratorReferenceMakeup(64, 6, 3)
        G2: nn.Module = GeneratorDeMakeup(64, 6, 3)
        D1: nn.Module = Discriminator(256, 64, 3, 3, 'SN')
        D2: nn.Module = Discriminator(256, 64, 3, 3, 'SN')
        vgg: nn.Module = torchvision.models.vgg19(pretrained=True)
        scaler: Scaler = Scaler(2., 0.5)

        super().__init__(device, G1, G2, D1, D2, vgg, scaler)
        self.apply(weights_init_xavier)

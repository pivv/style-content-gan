import sys
import os

from torch import nn
from torch.autograd import grad
import torch

from .resnet import ResidualBlock, BottleneckBlock


def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_normal_(m.weight.data, gain=1.0)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight.data, gain=1.0)


def weights_init_resnet(m):
    if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    if isinstance(m, (nn.BatchNorm1d, nn.InstanceNorm1d, nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    if isinstance(m, ResidualBlock):
        if m.bn2 is not None:
            nn.init.constant_(m.bn2.weight, 0)
    if isinstance(m, BottleneckBlock):
        if m.bn3 is not None:
            nn.init.constant_(m.bn3.weight, 0)

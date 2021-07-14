import sys
import os

from typing import List, Callable, Union, Any, TypeVar, Tuple, Dict
from torch import Tensor

import torch
from torch import nn
from torch.nn import functional as F


class ListMSELoss(nn.modules.loss._Loss):
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)

    def forward(self, input: List[Tensor], target: List[Tensor] = None) -> Tensor:
        if target is not None:
            input = [input_one - target_one for input_one, target_one in zip(input, target)]
        output: Tensor = torch.cat([(input_one ** 2).flatten(start_dim=1) for input_one in input], dim=1).mean(dim=1)
        #output: Tensor = sum((input_one ** 2).flatten(start_dim=1).sum(dim=1) for input_one in input) / sum(
        #    input_one.flatten(start_dim=1).size(1) for input_one in input)
        #output: Tensor = sum((input_one * input_one).flatten(start_dim=1).mean(dim=1) for input_one in input) / float(len(input))
        if self.reduction is 'none':
            pass
        elif self.reduction is 'mean':
            output = output.mean()
        else:
            assert(self.reduction is 'sum')
            output = output.sum()
        return output


class ListSiameseLoss(nn.modules.loss._Loss):
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)

    def forward(self, input: List[Tensor], target: List[Tensor] = None, margin: float or Tensor = 1.) -> Tensor:
        if target is not None:
            input = [input_one - target_one for input_one, target_one in zip(input, target)]
        output: Tensor = torch.clamp(margin - torch.sqrt(
            torch.cat([(input_one ** 2).flatten(start_dim=1) for input_one in input], dim=1).mean(dim=1)
        ), min=0.) ** 2
        #output: Tensor = torch.clamp(margin - torch.sqrt(
        #    sum((input_one ** 2).flatten(start_dim=1).sum(dim=1) for input_one in input) / sum(
        #    input_one.flatten(start_dim=1).size(1) for input_one in input)
        #), min=0.) ** 2
        if self.reduction is 'none':
            pass
        elif self.reduction is 'mean':
            output = output.mean()
        else:
            assert(self.reduction is 'sum')
            output = output.sum()
        return output


class SiameseLoss(nn.modules.loss._Loss):
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)

    def forward(self, input: Tensor, target: Tensor = None, margin: float or Tensor = 1.) -> Tensor:
        if target is not None:
            input = input - target
        output = torch.clamp(margin - torch.sqrt(torch.flatten(input**2, start_dim=1).mean(dim=1)), min=0.) ** 2
        if self.reduction is 'none':
            pass
        elif self.reduction is 'mean':
            output = output.mean()
        else:
            assert(self.reduction is 'sum')
            output = output.sum()
        return output


class L1SiameseLoss(nn.modules.loss._Loss):
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)

    def forward(self, input: Tensor, target: Tensor = None, margin: float or Tensor = 1.) -> Tensor:
        if target is not None:
            input = input - target
        output = torch.clamp(margin - torch.flatten(input.abs(), start_dim=1).mean(dim=1), min=0.)
        if self.reduction is 'none':
            pass
        elif self.reduction is 'mean':
            output = output.mean()
        else:
            assert(self.reduction is 'sum')
            output = output.sum()
        return output


class BinaryEntropyWithLogitsLoss(nn.modules.loss._Loss):
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)

    def forward(self, input: Tensor) -> Tensor:
        output = - (torch.sigmoid(input) * F.logsigmoid(input) + torch.sigmoid(1. - input) * F.logsigmoid(1. - input))
        if self.reduction is 'none':
            pass
        elif self.reduction is 'mean':
            output = output.mean()
        else:
            assert(self.reduction is 'sum')
            output = output.sum()
        return output

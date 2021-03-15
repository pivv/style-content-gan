# -*- coding: utf-8 -*-

import torch


class GradScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gamma):
        ctx.save_for_backward(gamma)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        gamma, = ctx.saved_tensors
        #return grad_output
        return grad_output * gamma, None


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gamma):
        ctx.save_for_backward(gamma)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        gamma, = ctx.saved_tensors
        #return grad_output
        return grad_output.neg() * gamma, None


def grad_scale(x, gamma=1.):
    return GradScale.apply(x, torch.FloatTensor([gamma])[0])


def grad_reverse(x, gamma=1.):
    return GradReverse.apply(x, torch.FloatTensor([gamma])[0])

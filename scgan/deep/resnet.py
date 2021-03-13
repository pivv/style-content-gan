import sys
import os

from torch import nn
from torch.autograd import grad
import torch


class MyConv(nn.Module):
    def __init__(self, dimension, input_dim, output_dim, kernel_size,
                 stride=1, dilation=1, bias=False, transpose=False):
        super().__init__()
        if dilation == 1:
            padding = (kernel_size - 1) // 2
        else:
            assert(kernel_size == 3 and stride == 1 and not transpose)
            padding = dilation
        if transpose:
            if stride == 1:
                if dimension == 1:
                    self.conv = nn.Conv1d(input_dim, output_dim, kernel_size, stride=stride, padding=padding, bias=bias)
                elif dimension == 2:
                    self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride=stride, padding=padding, bias=bias)
                else:
                    raise NotImplementedError(f'Not Implemented for dimension {dimension}.')
            elif kernel_size == 1:
                self.conv = ConvMeanPool(dimension, input_dim, output_dim, kernel_size, stride, bias,
                                         transpose=True, conv_first=False)
            else:
                assert(kernel_size == 3 and stride == 2)
                if dimension == 1:
                    self.conv = nn.Sequential(
                        nn.ConvTranspose1d(input_dim, output_dim, kernel_size=4, stride=2, padding=1, bias=bias),
                        nn.ReflectionPad1d((1, 0)),
                        nn.AvgPool1d(2, stride=1),
                    )
                elif dimension == 2:
                    self.conv = nn.Sequential(
                        nn.ConvTranspose2d(input_dim, output_dim, kernel_size=4, stride=2, padding=1, bias=bias),
                        nn.ReflectionPad2d((1, 0, 1, 0)),
                        nn.AvgPool2d(2, stride=1),
                    )
                else:
                    raise NotImplementedError(f'Not Implemented for dimension {dimension}.')
            #output_padding = stride - 1
            #self.conv = nn.ConvTranspose1d(input_dim, output_dim, kernel_size, stride=stride,
            #                               padding=padding, bias=bias, output_padding=output_padding)
        else:
            if dimension == 1:
                self.conv = nn.Conv1d(input_dim, output_dim, kernel_size, stride=stride, padding=padding,
                                      dilation=dilation, bias=bias)
            elif dimension == 2:
                self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride=stride, padding=padding,
                                      dilation=dilation, bias=bias)
            else:
                raise NotImplementedError(f'Not Implemented for dimension {dimension}.')

    def forward(self, input):
        output = self.conv(input)
        return output


class ConvMeanPool(nn.Module):
    def __init__(self, dimension, input_dim, output_dim, kernel_size,
                 stride=2, bias=False, transpose=False, conv_first=True):
        super().__init__()
        self.conv_first = conv_first
        self.conv = MyConv(dimension, input_dim, output_dim, kernel_size, stride=1, bias=bias, transpose=False)
        if dimension == 1:
            if transpose:
                self.pool = nn.Upsample(scale_factor=stride, mode='nearest')
            else:
                self.pool = nn.AvgPool1d(stride)
        elif dimension == 2:
            if transpose:
                self.pool = nn.Upsample(scale_factor=stride, mode='nearest')
            else:
                self.pool = nn.AvgPool2d(stride)
        else:
            raise NotImplementedError(f'Not Implemented for dimension {dimension}.')

    def forward(self, input):
        output = input
        if self.conv_first:
            output = self.conv(output)
            output = self.pool(output)
        else:
            output = self.pool(output)
            output = self.conv(output)
        return output


class ResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, dimension, input_dim, output_dim, transpose,
                 kernel_size=3, stride=2, norm=None, activation='ReLU', pool=True):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.norm = norm
        self.activation = activation
        self.pool = pool

        inplace = (self.norm != 'InstanceNorm')
        if self.activation == 'ReLU':
            self.relu1 = nn.ReLU(inplace=inplace)
            self.relu2 = nn.ReLU(inplace=inplace)
        elif self.activation == 'LeakyReLU':
            self.relu1 = nn.LeakyReLU(0.01, inplace=inplace)
            self.relu2 = nn.LeakyReLU(0.01, inplace=inplace)
        else:
            raise Exception('invalid Activation.')

        if self.norm is None:
            self.bn_shortcut = None
            self.bn1 = None
            self.bn2 = None
        elif self.norm == 'InstanceNorm':
            if dimension == 1:
                if stride == 1 and input_dim == output_dim:
                    self.bn_shortcut = None
                else:
                    self.bn_shortcut = nn.InstanceNorm1d(output_dim, affine=True)
                self.bn1 = nn.InstanceNorm1d(output_dim, affine=True)
                self.bn2 = nn.InstanceNorm1d(output_dim, affine=True)
            elif dimension == 2:
                if stride == 1 and input_dim == output_dim:
                    self.bn_shortcut = None
                else:
                    self.bn_shortcut = nn.InstanceNorm2d(output_dim, affine=True)
                self.bn1 = nn.InstanceNorm2d(output_dim, affine=True)
                self.bn2 = nn.InstanceNorm2d(output_dim, affine=True)
            else:
                raise NotImplementedError(f'Not Implemented for dimension {dimension}.')
        elif self.norm == 'BatchNorm':
            if dimension == 1:
                if stride == 1 and input_dim == output_dim:
                    self.bn_shortcut = None
                else:
                    self.bn_shortcut = nn.BatchNorm1d(output_dim)
                self.bn1 = nn.BatchNorm1d(output_dim)
                self.bn2 = nn.BatchNorm1d(output_dim)
            elif dimension == 2:
                if stride == 1 and input_dim == output_dim:
                    self.bn_shortcut = None
                else:
                    self.bn_shortcut = nn.BatchNorm2d(output_dim)
                self.bn1 = nn.BatchNorm2d(output_dim)
                self.bn2 = nn.BatchNorm2d(output_dim)
            else:
                raise NotImplementedError(f'Not Implemented for dimension {dimension}.')
        else:
            raise Exception('invalid Normalization.')

        if stride == 1 and input_dim == output_dim:
            self.conv_shortcut = None
        elif stride == 1 or not pool:
            self.conv_shortcut = MyConv(dimension, input_dim, output_dim, kernel_size=1, stride=stride,
                                        transpose=transpose)
        else:
            self.conv_shortcut = ConvMeanPool(dimension, input_dim, output_dim, kernel_size=1, stride=stride,
                                              transpose=transpose, conv_first=transpose)
        if stride == 1 or not pool:
            self.conv_1 = MyConv(dimension, input_dim, output_dim, kernel_size=kernel_size, stride=stride,
                                 transpose=transpose)
        else:
            self.conv_1 = ConvMeanPool(dimension, input_dim, output_dim, kernel_size=kernel_size, stride=stride,
                                       transpose=transpose, conv_first=(not transpose))
        self.conv_2 = MyConv(dimension, output_dim, output_dim, kernel_size=kernel_size, transpose=transpose)

    def forward(self, input):
        if self.stride == 1 and self.input_dim == self.output_dim:
            shortcut = input
        else:
            shortcut = self.conv_shortcut(input)
            if self.norm is not None:
                shortcut = self.bn_shortcut(shortcut)
        output = input
        output = self.conv_1(output)
        if self.norm is not None:
            output = self.bn1(output)
        output = self.relu1(output)
        output = self.conv_2(output)
        if self.norm is not None:
            output = self.bn2(output)
        output = self.relu2(shortcut + output)

        return output


class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, dimension, input_dim, output_dim, transpose,
                 kernel_size=3, stride=2, norm=None, activation='ReLU', pool=True):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.norm = norm
        self.activation = activation
        self.pool = pool

        assert(output_dim % self.expansion == 0)
        squeeze_dim = output_dim // self.expansion

        inplace = (self.norm != 'InstanceNorm')
        if self.activation == 'ReLU':
            self.relu1 = nn.ReLU(inplace=inplace)
            self.relu2 = nn.ReLU(inplace=inplace)
            self.relu3 = nn.ReLU(inplace=inplace)
        elif self.activation == 'LeakyReLU':
            self.relu1 = nn.LeakyReLU(0.01, inplace=inplace)
            self.relu2 = nn.LeakyReLU(0.01, inplace=inplace)
            self.relu3 = nn.LeakyReLU(0.01, inplace=inplace)
        else:
            raise Exception('invalid Activation.')

        if self.norm is None:
            self.bn_shortcut = None
            self.bn1 = None
            self.bn2 = None
            self.bn3 = None
        elif self.norm == 'InstanceNorm':
            if dimension == 1:
                if stride == 1 and input_dim == output_dim:
                    self.bn_shortcut = None
                else:
                    self.bn_shortcut = nn.InstanceNorm1d(output_dim, affine=True)
                self.bn1 = nn.InstanceNorm1d(squeeze_dim, affine=True)
                self.bn2 = nn.InstanceNorm1d(squeeze_dim, affine=True)
                self.bn3 = nn.InstanceNorm1d(output_dim, affine=True)
            elif dimension == 2:
                if stride == 1 and input_dim == output_dim:
                    self.bn_shortcut = None
                else:
                    self.bn_shortcut = nn.InstanceNorm2d(output_dim, affine=True)
                self.bn1 = nn.InstanceNorm2d(squeeze_dim, affine=True)
                self.bn2 = nn.InstanceNorm2d(squeeze_dim, affine=True)
                self.bn3 = nn.InstanceNorm2d(output_dim, affine=True)
            else:
                raise NotImplementedError(f'Not Implemented for dimension {dimension}.')
        elif self.norm == 'BatchNorm':
            if dimension == 1:
                if stride == 1 and input_dim == output_dim:
                    self.bn_shortcut = None
                else:
                    self.bn_shortcut = nn.BatchNorm1d(output_dim)
                self.bn1 = nn.BatchNorm1d(squeeze_dim)
                self.bn2 = nn.BatchNorm1d(squeeze_dim)
                self.bn3 = nn.BatchNorm1d(output_dim)
            elif dimension == 2:
                if stride == 1 and input_dim == output_dim:
                    self.bn_shortcut = None
                else:
                    self.bn_shortcut = nn.BatchNorm2d(output_dim)
                self.bn1 = nn.BatchNorm2d(squeeze_dim)
                self.bn2 = nn.BatchNorm2d(squeeze_dim)
                self.bn3 = nn.BatchNorm2d(output_dim)
            else:
                raise NotImplementedError(f'Not Implemented for dimension {dimension}.')
        else:
            raise Exception('invalid Normalization.')

        if stride == 1 and input_dim == output_dim:
            self.conv_shortcut = None
        elif stride == 1 or not pool:
            self.conv_shortcut = MyConv(dimension, input_dim, output_dim, kernel_size=1, stride=stride,
                                        transpose=transpose)
        else:
            self.conv_shortcut = ConvMeanPool(dimension, input_dim, output_dim, kernel_size=1, stride=stride,
                                              transpose=transpose, conv_first=transpose)
        self.conv_1 = MyConv(dimension, input_dim, squeeze_dim, kernel_size=1, transpose=transpose)
        if stride == 1 or not pool:
            self.conv_2 = MyConv(dimension, squeeze_dim, squeeze_dim, kernel_size=kernel_size, stride=stride,
                                 transpose=transpose)
        else:
            self.conv_2 = ConvMeanPool(dimension, squeeze_dim, squeeze_dim, kernel_size=kernel_size, stride=stride,
                                       transpose=transpose, conv_first=(not transpose))
        self.conv_3 = MyConv(dimension, squeeze_dim, output_dim, kernel_size=1, transpose=transpose)

    def forward(self, input):
        if self.stride == 1 and self.input_dim == self.output_dim:
            shortcut = input
        else:
            shortcut = self.conv_shortcut(input)
            if self.norm is not None:
                shortcut = self.bn_shortcut(shortcut)
        output = input
        output = self.conv_1(output)
        if self.norm is not None:
            output = self.bn1(output)
        output = self.relu1(output)
        output = self.conv_2(output)
        if self.norm is not None:
            output = self.bn2(output)
        output = self.relu2(output)
        output = self.conv_3(output)
        if self.norm is not None:
            output = self.bn3(output)
        output = self.relu3(shortcut + output)

        return output


def weights_init(m):
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


def make_residual_layer(dimension, block, inplanes, planes, num_block,
                        transpose=False, stride=2,
                        norm=None, activation='ReLU', pool=True):
    if num_block == 1:
        layer = block(dimension, inplanes, planes, transpose, kernel_size=3, stride=stride,
                      norm=norm, activation=activation, pool=pool)
        return layer
    else:
        layer = []
        layer.append(block(dimension, inplanes, planes, transpose, kernel_size=3, stride=stride,
                           norm=norm, activation=activation, pool=pool))
        inplanes = planes
        for _ in range(1, num_block):
            layer.append(block(dimension, inplanes, planes, transpose, kernel_size=3, stride=1,
                               norm=norm, activation=activation, pool=pool))
        return nn.Sequential(*layer)


class ResNet(nn.Module):
    def __init__(self, dimension, block, num_blocks, planes,
                 transpose=False, stride=2, skip_connections=False,
                 norm=None, activation='ReLU', pool=True):
        super().__init__()
        layers = []
        assert(len(num_blocks) == len(planes)-1)
        for iblock, num_block in enumerate(num_blocks):
            inplane, outplane = planes[iblock], planes[iblock+1]
            #outplanes = min(max_plane, inplane * 2)
            if transpose:
                if skip_connections and iblock < len(num_blocks) - 1:
                    layer = make_residual_layer(dimension, block, 2 * outplane, inplane, num_block,
                                                transpose, stride, norm, activation, pool)
                else:
                    layer = make_residual_layer(dimension, block, outplane, inplane, num_block,
                                                transpose, stride, norm, activation, pool)
            else:
                layer = make_residual_layer(dimension, block, inplane, outplane, num_block,
                                            transpose, stride, norm, activation, pool)
            #inplane = outplane
            layers.append(layer)
        if transpose:
            layers = layers[::-1]
        self.layers = nn.ModuleList(layers)
        self.skip_connections = skip_connections

    def forward(self, x, encoder_xs=None, return_xs=False):
        xs = []
        if return_xs:
            xs.append(x)
        for ilayer, layer in enumerate(self.layers):
            x = layer(x)
            if self.skip_connections:
                assert(encoder_xs is not None)
                x = torch.cat([x, encoder_xs[-1-1-ilayer]], dim=1)
            if return_xs:
                xs.append(x)

        if return_xs:
            return x, xs
        else:
            return x


def simple_resnet(dimension, num_blocks, planes, transpose, norm='BatchNorm', activation='ReLU', pool=False):
    return ResNet(dimension, BottleneckBlock, num_blocks, planes, transpose,
                  stride=2, norm=norm, activation=activation, pool=pool)

"""
StyleMapGAN
Copyright (c) 2021-present NAVER Corp.
This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import math
import random
import functools
import operator
import numpy as np

import sys
# sys.path.append('/mnt/bd/xcxc/code/StyleMapGAN_final')

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function, Variable

from training.op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d
# from training.base_transformer import Transformer

import copy

def conv1x1(in_channel, out_channel, stride=1):
    downsample = stride > 1
    return ConvLayer(
            in_channel, out_channel, 1, downsample=downsample, activate=False, bias=False
        )

class style_mix_down3(nn.Module):
    def __init__(self, base=512, c_stride=4, stride=1):
        super(style_mix_down3, self).__init__()
        self.stride = stride
        self.c_stride = c_stride
        self.cq = conv1x1(base, int(base/c_stride), stride)
        self.ck = conv1x1(base, int(base/c_stride), stride)
        self.cv = conv1x1(base, base, stride)

        self.softmax = nn.Softmax(dim=-1)
        self.scale = base ** 0.5

        self.gamma = nn.Parameter(torch.zeros(1))
        
        self.top = ConvLayer(2 * base, base, 1)

    def forward(self,
                vt: torch.Tensor, # target
                vs: torch.Tensor) -> torch.Tensor: # source
        batch, ch, height, width = vt.size()
        h_q = self.cq(vt).view(batch, int(ch/self.c_stride), -1).permute(0, 2, 1)
        h_k = self.ck(vs).view(batch, int(ch/self.c_stride), -1)
        energy = torch.bmm(h_q, h_k) / self.scale
        attention = self.softmax(energy).permute(0, 2, 1)
        h_v = self.cv(vs).view(batch, ch, -1)

        h = torch.bmm(h_v, attention)
        h = h.view(batch, ch, int(height/self.stride), int(width/self.stride))
        
        if self.stride > 1:
            h = F.interpolate(h, scale_factor=self.stride, mode="bilinear", align_corners=True)
        return h * self.gamma + vt


class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image, noise=None):
        if noise is None:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_()

        return image + self.weight * noise


def _repeat_tuple(t, n):
    r"""Repeat each element of `t` for `n` times.
    This can be used to translate padding arg used by Conv and Pooling modules
    to the ones used by `F.pad`.
    """
    return tuple(x for x in t for _ in range(n))


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k


class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, batch):
        out = self.input.repeat(batch, 1, 1, 1)

        return out


class ScaledLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()

        self.negative_slope = negative_slope

    def forward(self, input):
        out = F.leaky_relu(input, negative_slope=self.negative_slope)

        return out * math.sqrt(2)


class Upsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel) * (factor ** 2)
        self.register_buffer("kernel", kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad)

        return out


class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)

        self.register_buffer("kernel", kernel)

        self.pad = pad

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, pad=self.pad)

        return out


class EqualLinear(nn.Module):
    def __init__(
        self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)

        else:
            out = F.linear(
                input, self.weight * self.scale, bias=self.bias * self.lr_mul
            )

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})"
        )


class EqualConv2d(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        stride=1,
        padding=0,
        lr_mul=1,
        bias=True,
        bias_init=0,
        conv_transpose2d=False,
        activation=False,
    ):
        super().__init__()

        self.out_channel = out_channel
        self.kernel_size = kernel_size

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size).div_(lr_mul)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2) * lr_mul

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel).fill_(bias_init))

            self.lr_mul = lr_mul
        else:
            self.lr_mul = None

        self.conv_transpose2d = conv_transpose2d

        if activation:
            self.activation = ScaledLeakyReLU(0.2)
            # self.activation = FusedLeakyReLU(out_channel)
        else:
            self.activation = False

    def forward(self, input):
        if self.lr_mul != None:
            bias = self.bias * self.lr_mul
        else:
            bias = None

        if self.conv_transpose2d:
            # out = F.conv_transpose2d(
            #     input,
            #     self.weight.transpose(0, 1) * self.scale,
            #     bias=bias,
            #     stride=self.stride,
            #     # padding=self.padding,
            #     padding=0,
            # )

            # group version for fast training
            batch, in_channel, height, width = input.shape
            input_temp = input.view(1, batch * in_channel, height, width)
            weight = self.weight.unsqueeze(0).repeat(batch, 1, 1, 1, 1)
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
            )
            out = F.conv_transpose2d(
                input_temp,
                weight * self.scale,
                bias=bias,
                padding=self.padding,
                stride=2,
                groups=batch,
            )
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        else:
            out = F.conv2d(
                input,
                self.weight * self.scale,
                bias=bias,
                stride=self.stride,
                padding=self.padding,
            )

        if self.activation:
            out = self.activation(out)

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},"
            f" {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})"
        )


class EqualConv1dGroup(nn.Module):  # 1d conv group
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        bias=True,
        bias_init=0,
        lr_mul=1,
        activation=False,
    ):
        super().__init__()

        self.out_channel = out_channel
        self.in_channel = in_channel
        self.kernel_size = kernel_size

        self.weight = nn.Parameter(
            torch.randn(out_channel, 1, kernel_size).div_(lr_mul)
        )
        self.scale = (1 / math.sqrt(kernel_size)) * lr_mul

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel).fill_(bias_init))

            self.lr_mul = lr_mul
        else:
            self.lr_mul = None

        if activation:
            self.activation = ScaledLeakyReLU(0.2)
        else:
            self.activation = None

    def forward(self, input):
        if self.lr_mul != None:
            bias = self.bias * self.lr_mul
        else:
            bias = None

        out = F.conv1d(
            input, self.weight * self.scale, bias=bias, groups=self.in_channel
        )

        if self.activation:
            out = self.activation(out)

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},"
            f" {self.weight.shape[2]})"
        )


class ConvLayer(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        upsample=False,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        bias=True,
        activate=True,
        lr_mul=1,
    ):
        assert not (upsample and downsample)
        layers = []

        if upsample:
            stride = 2
            self.padding = 0
            layers.append(
                EqualConv2d(
                    in_channel,
                    out_channel,
                    kernel_size,
                    padding=self.padding,
                    stride=stride,
                    bias=bias and not activate,
                    conv_transpose2d=True,
                    lr_mul=lr_mul,
                )
            )

            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            layers.append(Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor))

        else:

            if downsample:
                factor = 2
                p = (len(blur_kernel) - factor) + (kernel_size - 1)
                pad0 = (p + 1) // 2
                pad1 = p // 2

                layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

                stride = 2
                self.padding = 0

            else:
                stride = 1
                self.padding = kernel_size // 2

            layers.append(
                EqualConv2d(
                    in_channel,
                    out_channel,
                    kernel_size,
                    padding=self.padding,
                    stride=stride,
                    bias=bias and not activate,
                )
            )

        if activate:
            if bias:
                layers.append(FusedLeakyReLU(out_channel))

            else:
                layers.append(ScaledLeakyReLU(0.2))

        super().__init__(*layers)


class ResBlock(nn.Module):
    # downsampling
    def __init__(
        self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1], return_features=False, downsample=True, upsample=False,
    ):
        super().__init__()

        self.downsample = downsample
        self.upsample = upsample

        self.conv1 = ConvLayer(in_channel, in_channel, 3)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, upsample=self.upsample, downsample=self.downsample)
        self.skip = ConvLayer(
            in_channel, out_channel, 1, upsample=self.upsample, downsample=self.downsample, activate=False, bias=False
        )
        self.return_features = return_features

    def forward(self, input):
        out1 = self.conv1(input)
        out2 = self.conv2(out1)

        skip = self.skip(input)
        out = (out2 + skip) / math.sqrt(2)

        if self.return_features:
            return out, out1, out2
        else:
            return out


class Discriminator(nn.Module):
    def __init__(self, size, channel_multiplier=2, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        convs = [ConvLayer(3, channels[size], 1)]

        log_size = int(math.log(size, 2))

        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]

            convs.append(ResBlock(in_channel, out_channel, blur_kernel))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3)
        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, channels[4], activation="fused_lrelu"),
            EqualLinear(channels[4], 1),
        )

    def forward(self, input):
        out = self.convs(input)

        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)
        out = self.final_conv(out)
        out = out.view(batch, -1)

        out = self.final_linear(out)
        return out


class StyledConv(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        blur_kernel,
        normalize_mode,
        upsample=False,
        activate=True,
    ):
        super().__init__()

        self.conv = ModulatedConv2d(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=upsample,
            blur_kernel=blur_kernel,
            normalize_mode=normalize_mode,
        )

        if activate:
            self.activate = FusedLeakyReLU(out_channel)
        else:
            self.activate = None

    def forward(self, input, style):
        out = self.conv(input, style)

        if self.activate is not None:
            out = self.activate(out)
        return out


class StyledConvNoise(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        blur_kernel,
        normalize_mode,
        upsample=False,
        activate=True,
    ):
        super().__init__()

        self.conv = ModulatedConv2d(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=upsample,
            blur_kernel=blur_kernel,
            normalize_mode=normalize_mode,
        )

        if activate:
            self.activate = FusedLeakyReLU(out_channel)
        else:
            self.activate = None
        
        self.noise = NoiseInjection()

    def forward(self, input, style, noise=None):
        out = self.conv(input, style)
        out = self.noise(out, noise=noise)

        if self.activate is not None:
            out = self.activate(out)
        return out



class ModulatedConv2d(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        normalize_mode,
        blur_kernel,
        upsample=False,
        downsample=False,
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            self.blur = Blur(blur_kernel, pad=(pad0, pad1))

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )

        self.normalize_mode = normalize_mode
        if normalize_mode == "InstanceNorm2d":
            self.norm = nn.InstanceNorm2d(in_channel, affine=False)
        elif normalize_mode == "BatchNorm2d":
            self.norm = nn.BatchNorm2d(in_channel, affine=False)

        self.beta = None

        self.gamma = EqualConv2d(
            style_dim,
            in_channel,
            kernel_size=3,
            padding=1,
            stride=1,
            bias=True,
            bias_init=1,
        )

        self.beta = EqualConv2d(
            style_dim,
            in_channel,
            kernel_size=3,
            padding=1,
            stride=1,
            bias=True,
            bias_init=0,
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, "
            f"upsample={self.upsample}, downsample={self.downsample})"
        )

    def forward(self, input, stylecode):
        assert stylecode is not None
        batch, in_channel, height, width = input.shape
        repeat_size = input.shape[3] // stylecode.shape[3]

        gamma = self.gamma(stylecode)
        if self.beta:
            beta = self.beta(stylecode)
        else:
            beta = 0

        weight = self.scale * self.weight
        weight = weight.repeat(batch, 1, 1, 1, 1)

        if self.normalize_mode in ["InstanceNorm2d", "BatchNorm2d"]:
            input = self.norm(input)
        elif self.normalize_mode == "LayerNorm":
            input = nn.LayerNorm(input.shape[1:], elementwise_affine=False)(input)
        elif self.normalize_mode == "GroupNorm":
            input = nn.GroupNorm(2 ** 3, input.shape[1:], affine=False)(input)
        elif self.normalize_mode == None:
            pass
        else:
            print("not implemented normalization")

        input = input * gamma + beta

        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )

        if self.upsample:
            input = input.view(1, batch * in_channel, height, width)
            weight = weight.view(
                batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size
            )
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
            )
            out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)

        elif self.downsample:
            input = self.blur(input)
            _, _, height, width = input.shape
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        else:
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=self.padding, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        return out


class StyledResBlock(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        style_dim,
        blur_kernel,
        normalize_mode,
        global_feature_channel=None,
        up_sample=True
    ):
        super().__init__()

        if style_dim is None:
            if global_feature_channel is not None:
                self.conv1 = StyledConv(
                    in_channel,
                    out_channel,
                    3,
                    in_channel + global_feature_channel,
                    blur_kernel=blur_kernel,
                    upsample=up_sample,
                    normalize_mode=normalize_mode,
                )
                self.conv2 = StyledConv(
                    out_channel,
                    out_channel,
                    3,
                    out_channel + global_feature_channel,
                    blur_kernel=blur_kernel,
                    normalize_mode=normalize_mode,
                )
            else:
                self.conv1 = StyledConv(
                    in_channel,
                    out_channel,
                    3,
                    in_channel,
                    blur_kernel=blur_kernel,
                    upsample=up_sample,
                    normalize_mode=normalize_mode,
                )
                self.conv2 = StyledConv(
                    out_channel,
                    out_channel,
                    3,
                    out_channel,
                    blur_kernel=blur_kernel,
                    normalize_mode=normalize_mode,
                )
        else:
            self.conv1 = StyledConv(
                in_channel,
                out_channel,
                3,
                style_dim,
                blur_kernel=blur_kernel,
                upsample=up_sample,
                normalize_mode=normalize_mode,
            )
            self.conv2 = StyledConv(
                out_channel,
                out_channel,
                3,
                style_dim,
                blur_kernel=blur_kernel,
                normalize_mode=normalize_mode,
            )

        self.skip = ConvLayer(
            in_channel, out_channel, 1, upsample=up_sample, activate=False, bias=False
        )

    def forward(self, input, stylecodes):
        out = self.conv1(input, stylecodes[0])
        out = self.conv2(out, stylecodes[1])

        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)

        return out


class StyledResBlockNoise(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        style_dim,
        blur_kernel,
        normalize_mode,
        global_feature_channel=None,
        up_sample=True
    ):
        super().__init__()

        if style_dim is None:
            if global_feature_channel is not None:
                self.conv1 = StyledConvNoise(
                    in_channel,
                    out_channel,
                    3,
                    in_channel + global_feature_channel,
                    blur_kernel=blur_kernel,
                    upsample=up_sample,
                    normalize_mode=normalize_mode,
                )
                self.conv2 = StyledConvNoise(
                    out_channel,
                    out_channel,
                    3,
                    out_channel + global_feature_channel,
                    blur_kernel=blur_kernel,
                    normalize_mode=normalize_mode,
                )
            else:
                self.conv1 = StyledConvNoise(
                    in_channel,
                    out_channel,
                    3,
                    in_channel,
                    blur_kernel=blur_kernel,
                    upsample=up_sample,
                    normalize_mode=normalize_mode,
                )
                self.conv2 = StyledConvNoise(
                    out_channel,
                    out_channel,
                    3,
                    out_channel,
                    blur_kernel=blur_kernel,
                    normalize_mode=normalize_mode,
                )
        else:
            self.conv1 = StyledConvNoise(
                in_channel,
                out_channel,
                3,
                style_dim,
                blur_kernel=blur_kernel,
                upsample=up_sample,
                normalize_mode=normalize_mode,
            )
            self.conv2 = StyledConvNoise(
                out_channel,
                out_channel,
                3,
                style_dim,
                blur_kernel=blur_kernel,
                normalize_mode=normalize_mode,
            )

        self.skip = ConvLayer(
            in_channel, out_channel, 1, upsample=up_sample, activate=False, bias=False
        )

    def forward(self, input, stylecodes, noise):
        out = self.conv1(input, stylecodes[0], noise[0])
        out = self.conv2(out, stylecodes[1], noise[1])

        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)

        return out



class ToRGB(nn.Module):
    def __init__(self, in_channel, style_dim, upsample, blur_kernel):
        super().__init__()

        if upsample:
            self.upsample = Upsample(blur_kernel)

        self.conv = ModulatedConv2d(
            in_channel, 3, 1, style_dim, blur_kernel=blur_kernel, normalize_mode=None
        )

        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, input, style, skip=None):
        out = self.conv(input, style)
        out = out + self.bias

        if skip is not None:
            skip = self.upsample(skip)

            out = out + skip

        return out


class ModulatedAdainConv2d(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            demodulate=True,
            upsample=False,
            downsample=False,
            blur_kernel=[1, 3, 3, 1],
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            self.blur = Blur(blur_kernel, pad=(pad0, pad1))

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )

        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)

        self.demodulate = demodulate

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, '
            f'upsample={self.upsample}, downsample={self.downsample})'
        )

    def forward(self, input, style):
        batch, in_channel, height, width = input.shape

        style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
        weight = self.scale * self.weight * style

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )

        if self.upsample:
            input = input.view(1, batch * in_channel, height, width)
            weight = weight.view(
                batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size
            )
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
            )
            out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)

        elif self.downsample:
            input = self.blur(input)
            _, _, height, width = input.shape
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        else:
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=self.padding, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        return out

class StyledAdainConv(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        blur_kernel,
        upsample=False,
        activate=True,
    ):
        super().__init__()

        self.conv = ModulatedAdainConv2d(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=upsample,
            blur_kernel=blur_kernel,
        )

        if activate:
            self.activate = FusedLeakyReLU(out_channel)
        else:
            self.activate = None

    def forward(self, input, style):

        out = self.conv(input, style)

        if self.activate is not None:
            out = self.activate(out)
        return out

class Encoder_return_32(nn.Module):
    def __init__(
        self,
        size,
        style_dim,
        latent_spatial_size,
        channel_multiplier,
        blur_kernel=[1, 3, 3, 1],
        in_ch=3
    ):
        super().__init__()

        channels = {
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        self.from_rgb = ConvLayer(in_ch, channels[size], 1)
        self.convs = nn.ModuleList()

        log_size = int(math.log(size, 2))
        self.log_size = log_size

        in_channel = channels[size]
        end = int(math.log(latent_spatial_size, 2))

        for i in range(self.log_size, end, -1):
            out_channel = channels[2 ** (i - 1)]

            self.convs.append(
                ResBlock(in_channel, out_channel, blur_kernel, return_features=True)
            )

            in_channel = out_channel

        # self.convs.append(
        #     ResBlock(in_channel, in_channel, blur_kernel, downsample=False,
        #         return_features=True, dilation=2)
        # )
        # self.convs.append(
        #     ResBlock(in_channel, in_channel, blur_kernel, downsample=False,
        #         return_features=True, dilation=2)
        # )

        self.final_conv = ConvLayer(in_channel, style_dim, 3)

    def forward(self, input):
        out = self.from_rgb(input)

        out_feat = []
        for convs in self.convs:
            out, _, _ = convs(out)
            if out.shape[-1] in [32, 64]:
                out_feat.append(out)

        out = self.final_conv(out)

        return out, out_feat # spatial style code


class Encoder_32(nn.Module):
    def __init__(
            self,
            size,
            style_dim,
            latent_spatial_size,
            channel_multiplier,
            blur_kernel=[1, 3, 3, 1],
            in_ch=3
    ):
        super().__init__()

        channels = {
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        self.from_rgb = ConvLayer(in_ch, channels[size], 1)
        self.convs = nn.ModuleList()

        log_size = int(math.log(size, 2))
        self.log_size = log_size

        in_channel = channels[size]
        end = int(math.log(latent_spatial_size, 2))

        for i in range(self.log_size, end, -1):
            out_channel = channels[2 ** (i - 1)]

            self.convs.append(
                ResBlock(in_channel, out_channel, blur_kernel, return_features=True)
            )

            in_channel = out_channel

        self.convs.append(
            ResBlock(in_channel, in_channel, blur_kernel, return_features=True, downsample=False)
        )

        self.final_conv = ConvLayer(in_channel, style_dim, 3)

    def forward(self, input):
        out = self.from_rgb(input)

        for convs in self.convs:
            out, _, _ = convs(out)

        out = self.final_conv(out)
        return out


class Decoder_32(nn.Module):
    def __init__(
        self,
        size,
        style_dim,
        latent_spatial_size,
        channel_multiplier,
        blur_kernel,
        normalize_mode,
        lr_mul,
        small_generator,
    ):
        super().__init__()

        self.size = size

        channels = {
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        self.input = ConstantInput(
            channels[latent_spatial_size], size=latent_spatial_size
        )

        self.log_size = int(math.log(size, 2))

        if small_generator:
            stylecode_dim = style_dim
        else:
            stylecode_dim = channels[latent_spatial_size]

        self.conv1 = StyledConv(
            channels[latent_spatial_size],
            channels[latent_spatial_size],
            3,
            stylecode_dim,
            blur_kernel=blur_kernel,
            normalize_mode=normalize_mode,
        )

        in_channel = channels[latent_spatial_size]

        self.start_index = int(math.log(latent_spatial_size, 2)) + 1  # if 4x4 -> 3
        self.convs = nn.ModuleList()
        self.convs_latent = nn.ModuleList()

        self.convs_latent.append(
            ConvLayer(
                style_dim, stylecode_dim, 3, bias=True, activate=True, lr_mul=lr_mul
            )
        )

        self.convs_latent.append(
            ConvLayer(
                style_dim, stylecode_dim, 3, bias=True, activate=True, lr_mul=lr_mul
            )
        )

        self.convs_latent.append(
            ConvLayer(
                style_dim, stylecode_dim, 3, bias=True, activate=True, lr_mul=lr_mul
            )
        )

        self.convs_latent.append(
            ConvLayer(
                style_dim, stylecode_dim, 3, bias=True, activate=True, lr_mul=lr_mul
            )
        )

        for i in range(self.start_index, self.log_size + 1): 
            if small_generator:
                stylecode_dim_prev, stylecode_dim_next = style_dim, style_dim
            else:
                stylecode_dim_prev = channels[2 ** (i - 1)]
                stylecode_dim_next = channels[2 ** i]
            self.convs_latent.append(
                ConvLayer(
                    stylecode_dim_prev,
                    stylecode_dim_next,
                    3,
                    upsample=True,
                    bias=True,
                    activate=True,
                    lr_mul=lr_mul,
                )
            )
            self.convs_latent.append(
                ConvLayer(
                    stylecode_dim_next,
                    stylecode_dim_next,
                    3,
                    bias=True,
                    activate=True,
                    lr_mul=lr_mul,
                )
            )

        if small_generator:
            stylecode_dim = style_dim
        else:
            stylecode_dim = None
        
        self.convs.append(
                StyledResBlock(
                    in_channel,
                    in_channel,
                    stylecode_dim,
                    blur_kernel,
                    normalize_mode=normalize_mode,
                    up_sample=False
                )
            )

        for i in range(self.start_index, self.log_size + 1): # 4，5， 6， 7，8
            out_channel = channels[2 ** i]
            self.convs.append(
                StyledResBlock(
                    in_channel,
                    out_channel,
                    stylecode_dim,
                    blur_kernel,
                    normalize_mode=normalize_mode,
                )
            )

            in_channel = out_channel

        if small_generator:
            stylecode_dim = style_dim
        else:
            stylecode_dim = channels[size]

        # add adain to to_rgb
        self.to_rgb = StyledConv(
            channels[size],
            3,
            1,
            stylecode_dim,
            blur_kernel=blur_kernel,
            normalize_mode=normalize_mode,
        )

        self.latent_spatial_size = latent_spatial_size
    
    def warp(self, x, flow):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow
        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow
        """
        B, C, H, W = x.size()
        Bf, Cf, Hf, Wf = flow.size() 

        flo = F.upsample(flow, size = (H,W),mode='bilinear', align_corners = True)  # resize flow to x
        
        # mesh grid 
        xs = np.linspace(-1, 1, W)
        xs = np.meshgrid(xs, xs)
        xs = np.stack(xs, 2)
        xs = torch.Tensor(xs).unsqueeze(0).repeat(B, 1,1,1).cuda()

        vgrid = Variable(xs, requires_grad=False) + flo.permute(0,2,3,1)         
        output = nn.functional.grid_sample(x, vgrid, align_corners = True)
        
        return output

    def forward(self, style_code, flow):
        style_code_lat = style_code

        style_codes = []
        style_code = self.warp(style_code, flow)

        for i, up_layer in enumerate(self.convs_latent):
            style_code = up_layer(style_code)
            # if i == 3:
            if style_code.shape[-1] == 32:
                style_code = style_code + style_code_lat
            style_codes.append(style_code)

        out = self.input(style_code.shape[0])
        out = self.conv1(out, style_codes[0])

        for i in range(len(self.convs)):
            out = self.convs[i](out, [style_codes[2 * i + 1], style_codes[2 * i + 2]])
        image = self.to_rgb(out, style_codes[-1])
        return image


class Decoder_cons_8_return_32(nn.Module):
    def __init__(
            self,
            size,
            style_dim,
            latent_spatial_size,
            channel_multiplier,
            blur_kernel,
            normalize_mode,
            lr_mul,
            small_generator,
    ):
        super().__init__()

        self.size = size

        channels = {
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        self.input = ConstantInput(
            channels[latent_spatial_size], size=latent_spatial_size
        )

        self.log_size = int(math.log(size, 2))

        if small_generator:
            stylecode_dim = style_dim
        else:
            stylecode_dim = channels[latent_spatial_size]

        self.conv1 = StyledConv(
            channels[latent_spatial_size],
            channels[latent_spatial_size],
            3,
            stylecode_dim,
            blur_kernel=blur_kernel,
            normalize_mode=normalize_mode,
        )

        in_channel = channels[latent_spatial_size]

        self.start_index = int(math.log(latent_spatial_size, 2)) + 1  # if 4x4 -> 3
        self.convs = nn.ModuleList()
        self.convs_latent = nn.ModuleList()

        # self.convs_latent.append(
        #     ConvLayer(
        #         style_dim, stylecode_dim, 3, bias=True, activate=True, lr_mul=lr_mul
        #     )
        # )
        # self.convs_latent.append(
        #     ConvLayer(
        #         stylecode_dim, stylecode_dim, 3, bias=True, activate=True, lr_mul=lr_mul
        #     )
        # )

        # self.convs_latent.append(
        #     ConvLayer(
        #         stylecode_dim, stylecode_dim, 3, bias=True, activate=True, lr_mul=lr_mul
        #     )
        # )

        # self.convs_latent.append(
        #     ConvLayer(
        #         stylecode_dim, stylecode_dim, 3, bias=True, activate=True, lr_mul=lr_mul
        #     )
        # )

        self.convs_latent.append(
            ConvLayer(
                stylecode_dim, stylecode_dim, 3, bias=True, activate=True, lr_mul=lr_mul
            )
        )

        self.convs_latent.append(
            ConvLayer(
                stylecode_dim, stylecode_dim, 3, bias=True, activate=True, lr_mul=lr_mul
            )
        )

        for i in range(self.start_index, self.log_size + 1):
            if small_generator:
                stylecode_dim_prev, stylecode_dim_next = style_dim, style_dim
            else:
                stylecode_dim_prev = channels[2 ** (i - 1)]
                stylecode_dim_next = channels[2 ** i]
            self.convs_latent.append(
                ConvLayer(
                    stylecode_dim_prev,
                    stylecode_dim_next,
                    3,
                    upsample=True,
                    bias=True,
                    activate=True,
                    lr_mul=lr_mul,
                )
            )
            self.convs_latent.append(
                ConvLayer(
                    stylecode_dim_next,
                    stylecode_dim_next,
                    3,
                    bias=True,
                    activate=True,
                    lr_mul=lr_mul,
                )
            )

        if small_generator:
            stylecode_dim = style_dim
        else:
            stylecode_dim = None

        # self.convs.append(
        #         StyledResBlock(
        #             in_channel,
        #             in_channel,
        #             stylecode_dim,
        #             blur_kernel,
        #             normalize_mode=normalize_mode,
        #             upsample=False
        #         )
        #     )

        # self.convs.append(
        #         StyledResBlock(
        #             in_channel,
        #             in_channel,
        #             stylecode_dim,
        #             blur_kernel,
        #             normalize_mode=normalize_mode,
        #             upsample=False
        #         )
        #     )

        for i in range(self.start_index, self.log_size + 1):  # 4，5， 6， 7，8
            out_channel = channels[2 ** i]
            self.convs.append(
                StyledResBlock(
                    in_channel,
                    out_channel,
                    stylecode_dim,
                    blur_kernel,
                    normalize_mode=normalize_mode,
                )
            )

            in_channel = out_channel

        if small_generator:
            stylecode_dim = style_dim
        else:
            stylecode_dim = channels[size]

        # add adain to to_rgb
        self.to_rgb = StyledConv(
            channels[size],
            3,
            1,
            stylecode_dim,
            blur_kernel=blur_kernel,
            normalize_mode=normalize_mode,
        )

        self.latent_spatial_size = latent_spatial_size
        # self.sigmoid = nn.Sequential(*[EqualConv2d(512, 512, 3, 1, 1), nn.Sigmoid()])

    def forward(self, style_code, tgt_feat):

        style_codes = []
        # style_code_multi = style_code
        # style_code = style_code[0]

        for up_layer in self.convs_latent:
            style_code = up_layer(style_code)
            style_codes.append(style_code)
            # if style_code.shape[-1] in [8, 16, 32, 64]:
            #     idx = int(math.log(style_code.shape[-1] // 8, 2))
            #     style_code = style_code + style_code_multi[idx]
            if style_code.shape[-1] in [32]:
                idx = style_code.shape[-1] // 32 - 1
                style_code = style_code + tgt_feat[idx]
            # if style_code.shape[-1] in [64]:
            #     idx = style_code.shape[-1] // 32 - 1
            #     style_code = style_code + (1 - self.sigmoid(style_code)) * tgt_feat[idx]

        out = self.input(style_code.shape[0])
        out = self.conv1(out, style_codes[0])

        for i in range(len(self.convs)):
            out = self.convs[i](out, [style_codes[2 * i + 1], style_codes[2 * i + 2]])
        image = self.to_rgb(out, style_codes[-1])
        return image


class Generator_32(nn.Module):
    def __init__(self,
        size,
        style_dim,
        latent_spatial_size,
        lr_mul,
        channel_multiplier,
        normalize_mode,
        blur_kernel=[1, 3, 3, 1],
        small_generator=False,
    ):
        super().__init__()

        self.latent_spatial_size = latent_spatial_size
        self.style_dim = style_dim
        self.decoder = Decoder_32(
            size,
            style_dim,
            latent_spatial_size,
            channel_multiplier=channel_multiplier,
            blur_kernel=blur_kernel,
            normalize_mode=normalize_mode,
            lr_mul=lr_mul,
            small_generator=small_generator,
        )  # always 1, always zero padding

    def forward(
        self,
        input,
        return_stylecode=False,
        input_is_stylecode=False,
        mix_space=None,
        mask=None,
        calculate_mean_stylemap=False,
        truncation=None,
        truncation_mean_latent=None,
    ):  
        style_code = input[0]
        flow = input[1]
        image = self.decoder(style_code, flow)

        return image


class Generator_globalatt_return_32(nn.Module):
    def __init__(self,
        size,
        style_dim,
        latent_spatial_size,
        lr_mul,
        channel_multiplier,
        normalize_mode,
        blur_kernel=[1, 3, 3, 1],
        small_generator=False,
    ):
        super().__init__()

        self.latent_spatial_size = latent_spatial_size
        self.style_dim = style_dim
        self.style_mix = style_mix_down3(style_dim)
        self.decoder = Decoder_cons_8_return_32(
            size,
            style_dim,
            latent_spatial_size,
            channel_multiplier=channel_multiplier,
            blur_kernel=blur_kernel,
            normalize_mode=normalize_mode,
            lr_mul=lr_mul,
            small_generator=small_generator,
        )  # always 1, always zero padding
        # self.GAP = nn.AdaptiveAvgPool2d(1)
        # self.conv = nn.ConvTranspose2d(512, 512, 8)

    def forward(
        self,
        input,
        return_stylecode=False,
        input_is_stylecode=False,
        mix_space=None,
        mask=None,
        calculate_mean_stylemap=False,
        truncation=None,
        truncation_mean_latent=None,
    ):
        tgt_code = input[0]
        src_code = input[1]
        tgt_feat = input[2]

        # mix_code = self.style_mix(self.GAP(tgt_code), self.GAP(src_code))
        # mix_code = self.conv(mix_code)

        mix_code = self.style_mix(tgt_code, src_code)
        image = self.decoder(mix_code, tgt_feat)

        return image


if __name__ == "__main__":
    img = torch.randn(1, 3, 256, 256).cuda()
    # net_p = Encoder_Pose(256, 512, 8, 2).cuda()
    # pose = net_p(img)
    # net_a = Encoder_App(256, 512, 32, 2).cuda()
    # app = net_a(img)
    # G = Decoder_Pose_App_woG(256,
    #                          512, 
    #                          32, 
    #                          channel_multiplier=2,
    #                          blur_kernel=[1, 3, 3, 1],
    #                          normalize_mode='LayerNorm',
    #                          lr_mul=0.1,
    #                          small_generator=False).cuda()
    # out = G(app, pose)
    # print(out.shape)
    net = Encoder_16(256, 512, 16, 2).cuda()
    pose, flow = net(img, True)
    print(pose, flow)
"""CycleGAN (Zhu et al. 2017): ResNet generator (9 blocks, ~11.4M params) + 70x70 PatchGAN, LSGAN loss.

Written from the paper; architecture follows junyanz/pytorch-CycleGAN-and-pix2pix for comparability.
"""
from __future__ import annotations

import random

import torch
import torch.nn as nn


class ResnetBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3), nn.InstanceNorm2d(dim), nn.ReLU(True),
            nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3), nn.InstanceNorm2d(dim),
        )

    def forward(self, x):
        return x + self.block(x)


class ResnetGenerator(nn.Module):
    def __init__(self, in_ch: int = 3, out_ch: int = 3, ngf: int = 64, n_blocks: int = 9):
        super().__init__()
        layers = [nn.ReflectionPad2d(3), nn.Conv2d(in_ch, ngf, 7), nn.InstanceNorm2d(ngf), nn.ReLU(True)]
        ch = ngf
        for _ in range(2):  # downsample
            layers += [nn.Conv2d(ch, ch * 2, 3, stride=2, padding=1), nn.InstanceNorm2d(ch * 2), nn.ReLU(True)]
            ch *= 2
        layers += [ResnetBlock(ch) for _ in range(n_blocks)]
        for _ in range(2):  # upsample
            layers += [nn.ConvTranspose2d(ch, ch // 2, 3, stride=2, padding=1, output_padding=1),
                       nn.InstanceNorm2d(ch // 2), nn.ReLU(True)]
            ch //= 2
        layers += [nn.ReflectionPad2d(3), nn.Conv2d(ch, out_ch, 7), nn.Tanh()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class PatchDiscriminator(nn.Module):
    def __init__(self, in_ch: int = 3, ndf: int = 64, n_layers: int = 3):
        super().__init__()
        layers = [nn.Conv2d(in_ch, ndf, 4, 2, 1), nn.LeakyReLU(0.2, True)]
        mult = 1
        for n in range(1, n_layers + 1):
            prev, mult = mult, min(2**n, 8)
            stride = 2 if n < n_layers else 1
            layers += [nn.Conv2d(ndf * prev, ndf * mult, 4, stride, 1), nn.InstanceNorm2d(ndf * mult), nn.LeakyReLU(0.2, True)]
        layers += [nn.Conv2d(ndf * mult, 1, 4, 1, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def init_weights(m: nn.Module, gain: float = 0.02) -> None:
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, gain)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class ImagePool:
    """History buffer of generated images for discriminator updates (Shrivastava et al. 2017)."""

    def __init__(self, size: int = 50):
        self.size, self.images = size, []

    def query(self, images: torch.Tensor) -> torch.Tensor:
        if self.size == 0:
            return images
        out = []
        for img in images.detach():
            img = img.unsqueeze(0)
            if len(self.images) < self.size:
                self.images.append(img)
                out.append(img)
            elif random.random() > 0.5:
                j = random.randrange(self.size)
                out.append(self.images[j].clone())
                self.images[j] = img
            else:
                out.append(img)
        return torch.cat(out)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())

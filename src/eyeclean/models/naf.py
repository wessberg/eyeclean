"""NAF-style building blocks (after NAFNet, Chen et al., ECCV 2022; independent implementation)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, c: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(c))
        self.bias = nn.Parameter(torch.zeros(c))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # statistics in FP32 even under BF16 autocast
        xf = x.float()
        mu = xf.mean(1, keepdim=True)
        var = (xf - mu).pow(2).mean(1, keepdim=True)
        y = (xf - mu) / torch.sqrt(var + self.eps)
        return (y * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)).to(x.dtype)


class SimpleGate(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = x.chunk(2, dim=1)
        return a * b


class NAFBlock(nn.Module):
    def __init__(self, c: int, dw_expand: int = 2, ffn_expand: int = 2):
        super().__init__()
        dw = c * dw_expand
        self.norm1 = LayerNorm2d(c)
        self.conv1 = nn.Conv2d(c, dw, 1)
        self.conv2 = nn.Conv2d(dw, dw, 3, padding=1, groups=dw)
        self.sg = SimpleGate()
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(dw // 2, dw // 2, 1))
        self.conv3 = nn.Conv2d(dw // 2, c, 1)
        self.norm2 = LayerNorm2d(c)
        ff = c * ffn_expand
        self.conv4 = nn.Conv2d(c, ff, 1)
        self.conv5 = nn.Conv2d(ff // 2, c, 1)
        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(self.norm1(x))
        y = self.sg(self.conv2(y))
        y = y * self.sca(y)
        y = self.conv3(y)
        x = x + y * self.beta
        z = self.conv5(self.sg(self.conv4(self.norm2(x))))
        return x + z * self.gamma


class Stage(nn.Module):
    def __init__(self, c: int, n: int, checkpoint: bool = False):
        super().__init__()
        self.blocks = nn.ModuleList([NAFBlock(c) for _ in range(n)])
        self.checkpoint = checkpoint

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for b in self.blocks:
            if self.checkpoint and self.training and x.requires_grad:
                x = torch.utils.checkpoint.checkpoint(b, x, use_reentrant=False)
            else:
                x = b(x)
        return x


class Down(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, 2, stride=2)

    def forward(self, x):
        return self.conv(x)


class Up(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out * 4, 1, bias=False)
        self.ps = nn.PixelShuffle(2)

    def forward(self, x):
        return self.ps(self.conv(x))


def conv_block(c_in: int, c_out: int) -> nn.Sequential:
    g = 8 if c_out % 8 == 0 else 1
    return nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1), nn.GroupNorm(g, c_out), nn.SiLU(),
                         nn.Conv2d(c_out, c_out, 3, padding=1), nn.GroupNorm(g, c_out), nn.SiLU())


def count_parameters(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def resize_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    if x.shape[-2:] == ref.shape[-2:]:
        return x
    return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

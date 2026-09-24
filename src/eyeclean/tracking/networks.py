"""S3FD face detector and FAN landmark network definitions.

Network definitions adapted from face-alignment (https://github.com/1adrianb/face-alignment),
Copyright (c) 2017, Adrian Bulat, BSD 3-Clause License (see third_party/face_alignment/
face-alignment-master.zip:LICENSE).  Only the layer definitions are reused; all pre- and
post-processing is reimplemented on CUDA in eyeclean.tracking.detector.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class L2Norm(nn.Module):
    def __init__(self, n_channels: int, scale: float = 1.0):
        super().__init__()
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.empty(n_channels).fill_(scale))

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        return x / norm * self.weight.view(1, -1, 1, 1)


class S3FD(nn.Module):
    def __init__(self):
        super().__init__()
        C = nn.Conv2d
        self.conv1_1, self.conv1_2 = C(3, 64, 3, 1, 1), C(64, 64, 3, 1, 1)
        self.conv2_1, self.conv2_2 = C(64, 128, 3, 1, 1), C(128, 128, 3, 1, 1)
        self.conv3_1, self.conv3_2, self.conv3_3 = C(128, 256, 3, 1, 1), C(256, 256, 3, 1, 1), C(256, 256, 3, 1, 1)
        self.conv4_1, self.conv4_2, self.conv4_3 = C(256, 512, 3, 1, 1), C(512, 512, 3, 1, 1), C(512, 512, 3, 1, 1)
        self.conv5_1, self.conv5_2, self.conv5_3 = C(512, 512, 3, 1, 1), C(512, 512, 3, 1, 1), C(512, 512, 3, 1, 1)
        self.fc6 = C(512, 1024, 3, 1, 3)
        self.fc7 = C(1024, 1024, 1, 1, 0)
        self.conv6_1, self.conv6_2 = C(1024, 256, 1, 1, 0), C(256, 512, 3, 2, 1)
        self.conv7_1, self.conv7_2 = C(512, 128, 1, 1, 0), C(128, 256, 3, 2, 1)
        self.conv3_3_norm, self.conv4_3_norm, self.conv5_3_norm = L2Norm(256, 10), L2Norm(512, 8), L2Norm(512, 5)
        self.conv3_3_norm_mbox_conf, self.conv3_3_norm_mbox_loc = C(256, 4, 3, 1, 1), C(256, 4, 3, 1, 1)
        self.conv4_3_norm_mbox_conf, self.conv4_3_norm_mbox_loc = C(512, 2, 3, 1, 1), C(512, 4, 3, 1, 1)
        self.conv5_3_norm_mbox_conf, self.conv5_3_norm_mbox_loc = C(512, 2, 3, 1, 1), C(512, 4, 3, 1, 1)
        self.fc7_mbox_conf, self.fc7_mbox_loc = C(1024, 2, 3, 1, 1), C(1024, 4, 3, 1, 1)
        self.conv6_2_mbox_conf, self.conv6_2_mbox_loc = C(512, 2, 3, 1, 1), C(512, 4, 3, 1, 1)
        self.conv7_2_mbox_conf, self.conv7_2_mbox_loc = C(256, 2, 3, 1, 1), C(256, 4, 3, 1, 1)

    def forward(self, x):
        r = F.relu
        h = r(self.conv1_2(r(self.conv1_1(x))))
        h = F.max_pool2d(h, 2, 2)
        h = r(self.conv2_2(r(self.conv2_1(h))))
        h = F.max_pool2d(h, 2, 2)
        h = r(self.conv3_3(r(self.conv3_2(r(self.conv3_1(h))))))
        f3 = h
        h = F.max_pool2d(h, 2, 2)
        h = r(self.conv4_3(r(self.conv4_2(r(self.conv4_1(h))))))
        f4 = h
        h = F.max_pool2d(h, 2, 2)
        h = r(self.conv5_3(r(self.conv5_2(r(self.conv5_1(h))))))
        f5 = h
        h = F.max_pool2d(h, 2, 2)
        h = r(self.fc7(r(self.fc6(h))))
        f7 = h
        h = r(self.conv6_2(r(self.conv6_1(h))))
        f6 = h
        h = r(self.conv7_2(r(self.conv7_1(h))))
        f72 = h
        f3, f4, f5 = self.conv3_3_norm(f3), self.conv4_3_norm(f4), self.conv5_3_norm(f5)
        cls1 = self.conv3_3_norm_mbox_conf(f3)
        ch = torch.chunk(cls1, 4, 1)
        cls1 = torch.cat([torch.max(torch.max(ch[0], ch[1]), ch[2]), ch[3]], dim=1)  # max-out background
        return [cls1, self.conv3_3_norm_mbox_loc(f3), self.conv4_3_norm_mbox_conf(f4), self.conv4_3_norm_mbox_loc(f4),
                self.conv5_3_norm_mbox_conf(f5), self.conv5_3_norm_mbox_loc(f5), self.fc7_mbox_conf(f7), self.fc7_mbox_loc(f7),
                self.conv6_2_mbox_conf(f6), self.conv6_2_mbox_loc(f6), self.conv7_2_mbox_conf(f72), self.conv7_2_mbox_loc(f72)]


def _conv3x3(i, o, s=1, p=1, b=False):
    return nn.Conv2d(i, o, 3, s, p, bias=b)


class ConvBlock(nn.Module):
    def __init__(self, i: int, o: int):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(i)
        self.conv1 = _conv3x3(i, o // 2)
        self.bn2 = nn.BatchNorm2d(o // 2)
        self.conv2 = _conv3x3(o // 2, o // 4)
        self.bn3 = nn.BatchNorm2d(o // 4)
        self.conv3 = _conv3x3(o // 4, o // 4)
        self.downsample = nn.Sequential(nn.BatchNorm2d(i), nn.ReLU(True), nn.Conv2d(i, o, 1, 1, bias=False)) if i != o else None

    def forward(self, x):
        o1 = self.conv1(F.relu(self.bn1(x)))
        o2 = self.conv2(F.relu(self.bn2(o1)))
        o3 = self.conv3(F.relu(self.bn3(o2)))
        o3 = torch.cat((o1, o2, o3), 1)
        return o3 + (self.downsample(x) if self.downsample is not None else x)


class HourGlass(nn.Module):
    def __init__(self, num_modules: int, depth: int, feat: int):
        super().__init__()
        self.depth = depth
        self._gen(depth, feat)

    def _gen(self, level, feat):
        self.add_module(f"b1_{level}", ConvBlock(feat, feat))
        self.add_module(f"b2_{level}", ConvBlock(feat, feat))
        if level > 1:
            self._gen(level - 1, feat)
        else:
            self.add_module(f"b2_plus_{level}", ConvBlock(feat, feat))
        self.add_module(f"b3_{level}", ConvBlock(feat, feat))

    def _fwd(self, level, x):
        up1 = self._modules[f"b1_{level}"](x)
        low1 = self._modules[f"b2_{level}"](F.avg_pool2d(x, 2, stride=2))
        low2 = self._fwd(level - 1, low1) if level > 1 else self._modules[f"b2_plus_{level}"](low1)
        low3 = self._modules[f"b3_{level}"](low2)
        return up1 + F.interpolate(low3, scale_factor=2, mode="nearest")

    def forward(self, x):
        return self._fwd(self.depth, x)


class FAN(nn.Module):
    def __init__(self, num_modules: int = 4):
        super().__init__()
        self.num_modules = num_modules
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = ConvBlock(64, 128)
        self.conv3 = ConvBlock(128, 128)
        self.conv4 = ConvBlock(128, 256)
        for m in range(num_modules):
            self.add_module(f"m{m}", HourGlass(1, 4, 256))
            self.add_module(f"top_m_{m}", ConvBlock(256, 256))
            self.add_module(f"conv_last{m}", nn.Conv2d(256, 256, 1, 1, 0))
            self.add_module(f"bn_end{m}", nn.BatchNorm2d(256))
            self.add_module(f"l{m}", nn.Conv2d(256, 68, 1, 1, 0))
            if m < num_modules - 1:
                self.add_module(f"bl{m}", nn.Conv2d(256, 256, 1, 1, 0))
                self.add_module(f"al{m}", nn.Conv2d(68, 256, 1, 1, 0))

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)), True)
        x = F.avg_pool2d(self.conv2(x), 2, stride=2)
        x = self.conv4(self.conv3(x))
        prev = x
        out = None
        for i in range(self.num_modules):
            ll = self._modules[f"top_m_{i}"](self._modules[f"m{i}"](prev))
            ll = F.relu(self._modules[f"bn_end{i}"](self._modules[f"conv_last{i}"](ll)), True)
            out = self._modules[f"l{i}"](ll)
            if i < self.num_modules - 1:
                prev = prev + self._modules[f"bl{i}"](ll) + self._modules[f"al{i}"](out)
        return out

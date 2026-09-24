"""EyeGuard: compact CUDA U-Net for ocular anatomy, protection masks and reflection localisation.

Input  (B, 4, S, S): asinh-normalised RGB crop + exposure token channel.
Output dict of logits:
  masks (B, 6, S, S): ocular surface, iris, pupil, protected (lids/lashes/tear), reflection, clipped
  present (B,): an open, visible eye is present in the crop
  refl_present (B,) [optional head, v2]: a *visible* designated reflection is present on the eye -
      smooth maximum of a full-resolution evidence map, so one small catchlight suffices
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .naf import conv_block

GUARD_MASKS = ("ocular", "iris", "pupil", "protected", "reflection", "clipped")


class EyeGuard(nn.Module):
    def __init__(self, widths=(24, 48, 96, 192), in_ch: int = 4, presence_head: bool = False):
        super().__init__()
        self.widths = tuple(widths)
        self.presence_head = bool(presence_head)
        self.enc = nn.ModuleList()
        c_prev = in_ch
        for w in widths:
            self.enc.append(conv_block(c_prev, w))
            c_prev = w
        self.mid = conv_block(widths[-1], widths[-1])
        self.dec = nn.ModuleList()
        for i in range(len(widths) - 1, 0, -1):
            self.dec.append(conv_block(widths[i] + widths[i - 1], widths[i - 1]))
        self.head = nn.Conv2d(widths[0], len(GUARD_MASKS), 1)
        self.present = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(widths[-1], 64), nn.SiLU(), nn.Linear(64, 1))
        if self.presence_head:
            self.refl_ev = nn.Conv2d(widths[0], 1, 1)

    def forward(self, x: torch.Tensor, return_features: bool = False) -> dict:
        skips = []
        h = x
        for i, blk in enumerate(self.enc):
            h = blk(h)
            skips.append(h)
            if i < len(self.enc) - 1:
                h = F.max_pool2d(h, 2)
        h = self.mid(h)
        present = self.present(h).squeeze(1)
        for j, blk in enumerate(self.dec):
            skip = skips[-2 - j]
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = blk(torch.cat([h, skip], 1))
        out = {"masks": self.head(h), "present": present}
        if self.presence_head:
            ev = self.refl_ev(h).flatten(1).float()
            # normalised log-sum-exp (temperature 4): between the spatial mean and the maximum of the evidence
            out["refl_present"] = (torch.logsumexp(4.0 * ev, 1) - math.log(ev.shape[1])) / 4.0
        if return_features:
            out['features'] = h
        return out

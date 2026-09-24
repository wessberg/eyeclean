"""ReflectionPresence: per-eye "a clearly visible designated reflection is present" classifier (v2).

A separate small CUDA network on top of a *frozen* EyeGuard: input = the guard's normalised crop +
exposure token (4 channels) and its six mask probabilities; a strided conv stack produces a coarse
evidence map that only counts on the visible ocular surface (a closed eye has no surface, so its
evidence is suppressed), and a normalised log-sum-exp (a smooth maximum) turns it into one logit per
eye, so a single small catchlight suffices.  The conservative gate edits an eye only when this
probability is above a validation-selected threshold.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def presence_input(guard_in: torch.Tensor, guard_probs: torch.Tensor) -> torch.Tensor:
    """guard_in (B,4,S,S) = EyeGuard input; guard_probs (B,6,S,S) sigmoid masks -> (B,10,S,S)."""
    return torch.cat([guard_in, guard_probs], 1)


class ReflectionPresence(nn.Module):
    def __init__(self, in_ch: int = 10, widths=(48, 64, 96)):
        super().__init__()
        layers, c = [], in_ch
        for w in widths:
            layers += [nn.Conv2d(c, w, 3, stride=2, padding=1), nn.SiLU()]
            c = w
        layers += [nn.Conv2d(c, c, 3, padding=1), nn.SiLU(), nn.Conv2d(c, c, 3, padding=1), nn.SiLU(), nn.Conv2d(c, c, 3, padding=1), nn.SiLU()]
        self.body = nn.Sequential(*layers)
        self.ev = nn.Conv2d(c, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: presence_input(...) -> (B,) logits."""
        f = self.body(x)
        ev = self.ev(f).float()
        ocular = F.adaptive_avg_pool2d(x[:, 4:5].float(), ev.shape[-2:])  # guard ocular-surface probability
        ev = ev - 30.0 * (ocular < 0.05).float()  # evidence only where the eye surface is visible
        ev = ev.flatten(1)
        return (torch.logsumexp(4.0 * ev, 1) - math.log(ev.shape[1])) / 4.0

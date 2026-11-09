"""GuardTX: a transformer eye-region detector with EyeGuard's interface.

EyeRestoreTX's encoder-decoder (Restormer-style: transposed channel attention and gated feed-forward blocks) on the
guard's 4-channel input (the normalised crop and an exposure token), with EyeGuard's outputs:
  masks         (B, 6, S, S) logits: ocular, iris, pupil, protected, reflection, clipped
  present       (B,) logit: an eye is in the crop (pooled from the bottleneck)
  refl_present  (B,) logit: visible reflection, a normalised log-sum-exp of a per-pixel evidence map (as EyeGuard)
It is meant to be initialised from a trained EyeRestoreTX (`init_from_restorer`): the restorer has learnt, on the same
data, where a catchlight is and what lies under it.  About 57M parameters.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .eyeguard import GUARD_MASKS
from .eyerestore_tx import EyeRestoreTX

GUARD_IN = 4
# restore_input's channels: normalised crop (0-2), guard probs 0-4 (3-7), clipped (8), exposure token (9), coords (10-11);
# guard_input's: normalised crop (0-2), exposure token (3)
_INTRO_FROM_RESTORER = (0, 1, 2, 9)


class GuardTX(nn.Module):
    def __init__(self, dim: int = 64, blocks=(6, 8, 8, 10), dec_blocks=(8, 8, 6), refine: int = 6, heads=(2, 4, 8, 16),
                 expand: float = 2.66, checkpoint: bool = False):
        super().__init__()
        self.net = EyeRestoreTX(dim=dim, blocks=blocks, dec_blocks=dec_blocks, refine=refine, heads=heads, expand=expand,
                                in_ch=GUARD_IN, checkpoint=checkpoint)
        del self.net.head_res, self.net.head_aux          # the restorer's output heads are not used
        c2, c4 = dim * 2, dim * 8
        self.head = nn.Conv2d(c2, len(GUARD_MASKS), 3, padding=1)
        self.present = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(c4, 64), nn.SiLU(), nn.Linear(64, 1))
        self.refl_ev = nn.Conv2d(c2, 1, 1)

    def forward(self, x: torch.Tensor, return_features: bool = False) -> dict:
        n = self.net
        h1 = n._run(n.enc1, n.intro(x))
        h2 = n._run(n.enc2, n.down1(h1))
        h3 = n._run(n.enc3, n.down2(h2))
        h4 = n._run(n.mid, n.down3(h3))
        present = self.present(h4).squeeze(1)
        d3 = n._run(n.dec3, n.red3(torch.cat([n.up3(h4), h3], 1)))
        d2 = n._run(n.dec2, n.red2(torch.cat([n.up2(d3), h2], 1)))
        d1 = n._run(n.dec1, torch.cat([n.up1(d2), h1], 1))
        d1 = n._run(n.refine, d1)
        ev = self.refl_ev(d1).flatten(1).float()
        out = {"masks": self.head(d1), "present": present,
               "refl_present": (torch.logsumexp(4.0 * ev, 1) - math.log(ev.shape[1])) / 4.0}
        if return_features:
            out["features"] = d1
        return out

    @torch.no_grad()
    def init_from_restorer(self, restorer_sd: dict) -> list[str]:
        """Copy a trained EyeRestoreTX's weights (keys without the "restore_single." prefix) into the backbone.  The
        input conv keeps the restorer's filters for the channels both inputs share; the restorer's heads are skipped.
        Returns the backbone keys left at their initialisation."""
        own = self.net.state_dict()
        copied = set()
        for k, v in restorer_sd.items():
            if k in own and own[k].shape == v.shape:
                own[k].copy_(v)
                copied.add(k)
        w = restorer_sd.get("intro.weight")
        if w is not None and w.shape[1] > max(_INTRO_FROM_RESTORER):
            own["intro.weight"].copy_(w[:, list(_INTRO_FROM_RESTORER)])
            copied.add("intro.weight")
        self.net.load_state_dict(own)
        return sorted(k for k in own if k not in copied)


def build_guard_tx(cfg: dict | None = None, checkpoint: bool = False) -> GuardTX:
    cfg = cfg or {}
    return GuardTX(dim=cfg.get("dim", 64), blocks=tuple(cfg.get("blocks", (6, 8, 8, 10))),
                   dec_blocks=tuple(cfg.get("dec_blocks", (8, 8, 6))), refine=cfg.get("refine", 6),
                   expand=cfg.get("expand", 2.66), checkpoint=checkpoint)

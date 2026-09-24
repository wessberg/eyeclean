"""EyeRestoreTX: a transformer restorer, drop-in for EyeRestore's single-frame path.

Why a transformer
-----------------
The CNN restorer (21.15M params) has been the fill's weak point for many rounds.  The user asked
for a transformer and a large model, and the earlier evidence against transformers does not stand:
Round 11's transformer-vs-CNN null ran 2,000 updates per arm, and Round 16 showed that horizon
cannot detect the effect it was testing (the CNN itself was still underfit at 60,000).

Architecture
------------
Restormer-style, chosen because it is built for restoration at high resolution rather than for
classification:

  * MDTA - multi-Dconv head transposed attention.  Attention is computed across CHANNELS, not
    pixels, so cost is linear in pixel count instead of quadratic.  A 256x256 crop has 65,536
    pixels; full spatial attention there is not affordable, and window attention alone gives up the
    global context that makes a catchlight distinguishable from a real highlight.
  * GDFN - gated depthwise feed-forward, which controls how much of each feature survives.
  * A U-shaped encoder/decoder so the model still sees the eye at several scales, with more blocks
    where the receptive field matters most.

Interface is IDENTICAL to EyeRestore: forward(x) -> {residual, support, conf, synth}, the residual
head zero-initialised so an untrained model is a passthrough, and the same auxiliary biases.  That
keeps the gate, the bundle format and every downstream diagnostic working unchanged.

Token geometry: no patch embedding and no positional embedding, so there is nothing to interpolate
when the crop size changes; the 256 crop divides cleanly through 128/64/32 for the three downsamples.

Geometry input (Round 34, optional)
-----------------------------------
``geo_ch > 0`` adds a second input convolution, ``intro_geo``, for eye-geometry maps (pupil and
iris coverage plus an availability flag), summed into the first feature map.  It is ZERO-INITIALISED
and has no bias, so a model built with it produces bit-identical output to one without it until
training moves those weights - which is what lets a geometry arm be compared against a
budget-matched control warm-started from the same checkpoint.  ``geo_ch = 0`` (the default) builds
exactly the pre-Round-34 module: no extra parameter, the same state dict, every bundle unchanged.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

RESTORE_IN = 12


class BiasFreeLayerNorm(nn.Module):
    """LayerNorm over channels without mean subtraction - standard for restoration, where removing
    the mean discards the very low-frequency level the fill has to get right."""

    def __init__(self, c: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(c))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sigma = x.var(1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight.view(1, -1, 1, 1)


class MDTA(nn.Module):
    """Attention across channels: O(C^2 * HW) rather than O((HW)^2)."""

    def __init__(self, c: int, heads: int):
        super().__init__()
        self.heads = heads
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))
        self.qkv = nn.Conv2d(c, c * 3, 1, bias=False)
        self.qkv_dw = nn.Conv2d(c * 3, c * 3, 3, padding=1, groups=c * 3, bias=False)
        self.proj = nn.Conv2d(c, c, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        q, k, v = self.qkv_dw(self.qkv(x)).chunk(3, dim=1)
        q = q.reshape(b, self.heads, c // self.heads, h * w)
        k = k.reshape(b, self.heads, c // self.heads, h * w)
        v = v.reshape(b, self.heads, c // self.heads, h * w)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        out = (attn.softmax(dim=-1) @ v).reshape(b, c, h, w)
        return self.proj(out)


class GDFN(nn.Module):
    """Gated depthwise feed-forward."""

    def __init__(self, c: int, expand: float = 2.66):
        super().__init__()
        hid = int(c * expand)
        self.project_in = nn.Conv2d(c, hid * 2, 1, bias=False)
        self.dw = nn.Conv2d(hid * 2, hid * 2, 3, padding=1, groups=hid * 2, bias=False)
        self.project_out = nn.Conv2d(hid, c, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.dw(self.project_in(x)).chunk(2, dim=1)
        return self.project_out(F.gelu(a) * b)


class Block(nn.Module):
    def __init__(self, c: int, heads: int, expand: float):
        super().__init__()
        self.n1, self.attn = BiasFreeLayerNorm(c), MDTA(c, heads)
        self.n2, self.ffn = BiasFreeLayerNorm(c), GDFN(c, expand)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.n1(x))
        return x + self.ffn(self.n2(x))


class Downsample(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(c, c // 2, 3, padding=1, bias=False), nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(c, c * 2, 3, padding=1, bias=False), nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


class EyeRestoreTX(nn.Module):
    def __init__(self, dim: int = 64, blocks=(6, 8, 8, 10), dec_blocks=(8, 8, 6), refine: int = 6,
                 heads=(2, 4, 8, 16), expand: float = 2.66, in_ch: int = RESTORE_IN,
                 checkpoint: bool = True, geo_ch: int = 0):
        super().__init__()
        self.use_checkpoint = checkpoint
        self.geo_ch = geo_ch
        c1, c2, c3, c4 = dim, dim * 2, dim * 4, dim * 8
        self.intro = nn.Conv2d(in_ch, c1, 3, padding=1)
        self.intro_geo = None
        if geo_ch > 0:
            self.intro_geo = nn.Conv2d(geo_ch, c1, 3, padding=1, bias=False)
            nn.init.zeros_(self.intro_geo.weight)
        self.enc1 = nn.ModuleList([Block(c1, heads[0], expand) for _ in range(blocks[0])])
        self.down1 = Downsample(c1)
        self.enc2 = nn.ModuleList([Block(c2, heads[1], expand) for _ in range(blocks[1])])
        self.down2 = Downsample(c2)
        self.enc3 = nn.ModuleList([Block(c3, heads[2], expand) for _ in range(blocks[2])])
        self.down3 = Downsample(c3)
        self.mid = nn.ModuleList([Block(c4, heads[3], expand) for _ in range(blocks[3])])
        self.up3 = Upsample(c4)
        self.red3 = nn.Conv2d(c4, c3, 1, bias=False)
        self.dec3 = nn.ModuleList([Block(c3, heads[2], expand) for _ in range(dec_blocks[0])])
        self.up2 = Upsample(c3)
        self.red2 = nn.Conv2d(c3, c2, 1, bias=False)
        self.dec2 = nn.ModuleList([Block(c2, heads[1], expand) for _ in range(dec_blocks[1])])
        self.up1 = Upsample(c2)
        self.dec1 = nn.ModuleList([Block(c2, heads[0], expand) for _ in range(dec_blocks[2])])
        self.refine = nn.ModuleList([Block(c2, heads[0], expand) for _ in range(refine)])
        # heads match EyeRestore exactly so the gate and bundle are unchanged
        self.head_res = nn.Conv2d(c2, 3, 3, padding=1)
        nn.init.zeros_(self.head_res.weight)
        nn.init.zeros_(self.head_res.bias)
        self.head_aux = nn.Conv2d(c2, 3, 3, padding=1)
        with torch.no_grad():
            self.head_aux.bias.copy_(torch.tensor([-2.0, 2.0, -3.0]))

    def _run(self, blocks, x):
        for b in blocks:
            if self.use_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(b, x, use_reentrant=False)
            else:
                x = b(x)
        return x

    def forward(self, x: torch.Tensor, return_features: bool = False, geo: torch.Tensor | None = None) -> dict:
        h0 = self.intro(x)
        if self.intro_geo is not None:
            if geo is None:
                raise ValueError("this restorer was built with geometry input (geo_ch > 0) and needs `geo`")
            h0 = h0 + self.intro_geo(geo)
        h1 = self._run(self.enc1, h0)
        h2 = self._run(self.enc2, self.down1(h1))
        h3 = self._run(self.enc3, self.down2(h2))
        h4 = self._run(self.mid, self.down3(h3))
        d3 = self._run(self.dec3, self.red3(torch.cat([self.up3(h4), h3], 1)))
        d2 = self._run(self.dec2, self.red2(torch.cat([self.up2(d3), h2], 1)))
        d1 = self._run(self.dec1, torch.cat([self.up1(d2), h1], 1))
        d1 = self._run(self.refine, d1)
        aux = self.head_aux(d1)
        out = {"residual": self.head_res(d1), "support": aux[:, 0:1],
               "conf": aux[:, 1:2], "synth": aux[:, 2:3]}
        if return_features:
            out["features"] = d1
        return out


def build_restore_tx(cfg: dict | None = None, checkpoint: bool = True) -> EyeRestoreTX:
    cfg = cfg or {}
    return EyeRestoreTX(dim=cfg.get("dim", 64), blocks=tuple(cfg.get("blocks", (6, 8, 8, 10))),
                        dec_blocks=tuple(cfg.get("dec_blocks", (8, 8, 6))),
                        refine=cfg.get("refine", 6), expand=cfg.get("expand", 2.66),
                        checkpoint=checkpoint, geo_ch=cfg.get("geo_ch", 0))

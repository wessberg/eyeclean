"""EyeRestore: deterministic residual restoration with optional five-frame temporal fusion.

Single-frame input channels (RESTORE_IN = 12):
  0-2  asinh-normalised RGB crop (working space)
  3-7  EyeGuard probabilities: ocular, iris, pupil, protected, reflection
  8    clipped (sensor-saturated) mask computed from the input
  9    exposure token (log2 exposure / 10, broadcast)
  10-11 iris-centred coordinates (x-cx)/r, (y-cy)/r from EyeGuard geometry (clamped)
Outputs:
  residual (B,3,S,S)   added in normalised space (zero-initialised head: passthrough at init)
  support  (B,1,S,S)   refined correction-support logit
  conf     (B,1,S,S)   reconstruction-confidence logit (calibrated later on held-out data)
  synth    (B,1,S,S)   recoverable (0) vs requires synthesis (1) logit

Temporal fusion: per-frame encoder features at 1/4 and 1/8 resolution; the centre frame attends
to a 3x3 spatial neighbourhood in each of the five frames (local SDPA, invalid frames masked:
clip edges, cuts, closed-eye/lost-track frames).  Neighbour keys also see reflection/clipped
masks so bright moving highlights need not be treated as anatomy.  A zero-initialised gate
makes the untrained temporal path identical to the single-frame model.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .naf import Down, Stage, Up

RESTORE_IN = 12


class LocalTemporalAttention(nn.Module):
    def __init__(self, c: int, heads: int = 4, frames: int = 5, window: int = 3, extra_k: int = 2):
        super().__init__()
        self.c, self.h, self.T, self.k = c, heads, frames, window
        self.q = nn.Conv2d(c, c, 1)
        self.kv = nn.Conv2d(c + extra_k, 2 * c, 1)
        self.out = nn.Conv2d(c, c, 1)
        self.tpos = nn.Parameter(torch.zeros(frames, c))
        self.gate = nn.Parameter(torch.zeros(1, c, 1, 1))  # zero-init: starts as single-frame

    def forward(self, feats: torch.Tensor, extra: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """feats (B,T,C,H,W); extra (B,T,2,H,W) masks; valid (B,T) bool. Returns fused centre (B,C,H,W)."""
        B, T, C, H, W = feats.shape
        ctr = T // 2
        q = self.q(feats[:, ctr])  # (B,C,H,W)
        kvin = torch.cat([feats + self.tpos.view(1, T, C, 1, 1).to(feats.dtype), extra.to(feats.dtype)], 2)
        kv = self.kv(kvin.flatten(0, 1))  # (B*T, 2C, H, W)
        k_, v_ = kv.chunk(2, 1)
        pad = self.k // 2
        # unfold 3x3 neighbourhoods: (B*T, C*k*k, H*W)
        ku = F.unfold(k_, self.k, padding=pad).view(B, T, C, self.k * self.k, H * W)
        vu = F.unfold(v_, self.k, padding=pad).view(B, T, C, self.k * self.k, H * W)
        hd = C // self.h
        # tokens per pixel: T*k*k ; layout (B*H*W, heads, tokens, hd)
        ku = ku.permute(0, 4, 2, 1, 3).reshape(B * H * W, self.h, hd, T * self.k * self.k).transpose(2, 3)
        vu = vu.permute(0, 4, 2, 1, 3).reshape(B * H * W, self.h, hd, T * self.k * self.k).transpose(2, 3)
        qq = q.permute(0, 2, 3, 1).reshape(B * H * W, self.h, 1, hd)
        mask = valid.view(B, 1, T, 1).expand(B, H * W, T, self.k * self.k).reshape(B * H * W, 1, 1, T * self.k * self.k)
        att = F.scaled_dot_product_attention(qq, ku, vu, attn_mask=mask)
        att = att.view(B, H, W, C).permute(0, 3, 1, 2)
        return feats[:, ctr] + self.gate * self.out(att)


class EyeRestore(nn.Module):
    def __init__(self, widths=(32, 64, 128, 256), enc_blocks=(2, 2, 4, 6), dec_blocks=(2, 2, 2), mid_width: int = 512,
                 mid_blocks: int = 8, in_ch: int = RESTORE_IN, frames: int = 5, checkpoint: bool = True):
        super().__init__()
        self.widths, self.frames = tuple(widths), frames
        self.intro = nn.Conv2d(in_ch, widths[0], 3, padding=1)
        self.enc = nn.ModuleList([Stage(w, n, checkpoint) for w, n in zip(widths, enc_blocks)])
        self.down = nn.ModuleList([Down(widths[i], widths[i + 1]) for i in range(len(widths) - 1)] + [Down(widths[-1], mid_width)])
        self.mid = Stage(mid_width, mid_blocks, checkpoint)
        self.up = nn.ModuleList([Up(mid_width, widths[-1])] + [Up(widths[i + 1], widths[i]) for i in range(len(widths) - 2, -1, -1)])
        dec_ws = list(reversed(widths))  # 256,128,64,32
        dec_n = [2] + list(dec_blocks)
        self.dec = nn.ModuleList([Stage(w, n, checkpoint) for w, n in zip(dec_ws, dec_n)])
        c0 = widths[0]
        self.head_res = nn.Conv2d(c0, 3, 3, padding=1)
        nn.init.zeros_(self.head_res.weight)
        nn.init.zeros_(self.head_res.bias)
        self.head_aux = nn.Conv2d(c0, 3, 3, padding=1)  # support, conf, synth
        with torch.no_grad():
            self.head_aux.bias.copy_(torch.tensor([-2.0, 2.0, -3.0]))
        self.temporal = nn.ModuleDict({
            "l2": LocalTemporalAttention(widths[2], heads=4, frames=frames),
            "l3": LocalTemporalAttention(widths[3], heads=8, frames=frames),
        })

    # --- encoder / decoder ------------------------------------------------------------------
    def encode(self, x: torch.Tensor) -> list[torch.Tensor]:
        h = self.intro(x)
        skips = []
        for i, st in enumerate(self.enc):
            h = st(h)
            skips.append(h)
            h = self.down[i](h)
        return skips + [h]  # [l0, l1, l2, l3, bottom]

    def decode(self, feats: list[torch.Tensor], x: torch.Tensor) -> dict:
        l0, l1, l2, l3, bottom = feats
        h = self.mid(bottom)
        skips = [l3, l2, l1, l0]
        for i, (up, st) in enumerate(zip(self.up, self.dec)):
            h = up(h) + skips[i]
            h = st(h)
        res = self.head_res(h)
        aux = self.head_aux(h)
        return {"residual": res, "support": aux[:, 0:1], "conf": aux[:, 1:2], "synth": aux[:, 2:3]}

    def forward(self, x: torch.Tensor) -> dict:
        """Single-frame path. x: (B, RESTORE_IN, S, S)."""
        return self.decode(self.encode(x), x)

    def forward_video(self, xs: torch.Tensor, valid: torch.Tensor, cached: list[list[torch.Tensor]] | None = None) -> dict:
        """xs (B,T,C,S,S) window centred on the frame to restore; valid (B,T) bool.
        ``cached`` may provide per-frame encoder features to avoid recomputation (inference)."""
        B, T = xs.shape[:2]
        if cached is None:
            enc = self.encode(xs.flatten(0, 1))
            per = [[e.view(B, T, *e.shape[1:])[:, t] for e in enc] for t in range(T)]
        else:
            per = cached
        ctr = T // 2
        feats = list(per[ctr])
        valid = valid.clone()
        valid[:, ctr] = True
        for lvl, idx in (("l2", 2), ("l3", 3)):
            stack = torch.stack([per[t][idx] for t in range(T)], 1)
            Hh, Ww = stack.shape[-2:]
            # neighbour reflection/clipped hints at this resolution
            m = xs[:, :, [7, 8]]
            m = F.interpolate(m.flatten(0, 1).float(), size=(Hh, Ww), mode="area").view(B, T, 2, Hh, Ww)
            feats[idx] = self.temporal[lvl](stack, m, valid)
        return self.decode(feats, xs[:, ctr])


def build_restore(model_cfg: dict, checkpoint: bool = True) -> EyeRestore:
    return EyeRestore(widths=tuple(model_cfg["widths"]), enc_blocks=tuple(model_cfg["encoder_blocks"]),
                      frames=model_cfg["temporal_frames"], checkpoint=checkpoint)

"""Model input construction shared by training and inference (all CUDA, FP32 math)."""

from __future__ import annotations

import math

import torch

from ..color import transforms as CT

ALPHA = CT.ASINH_ALPHA


def crop_exposure(crop_w: torch.Tensor) -> torch.Tensor:
    """(B,3,S,S) working-space crops -> (B,) exposure reference e = 2 x median AP1 luminance."""
    lum = 0.2722 * crop_w[:, 0] + 0.6741 * crop_w[:, 1] + 0.0537 * crop_w[:, 2]
    B = lum.shape[0]
    med = torch.quantile(lum.reshape(B, -1)[:, ::4].float(), 0.5, dim=1)
    return torch.clamp(med * 2.0, min=1e-4)


def normalize(x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
    return CT.asinh_forward(x, e, ALPHA)


def denormalize(y: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
    return CT.asinh_inverse(y, e, ALPHA)


def clipped_mask(crop_w: torch.Tensor, white: torch.Tensor | float, working_color: str = "linear-acescg") -> torch.Tensor:
    """Pixels at the sensor/encoding white level in any source channel (B,1,S,S)."""
    lin = CT.working_to_linear709(crop_w, working_color)
    w = white if isinstance(white, torch.Tensor) else torch.tensor(float(white), device=crop_w.device)
    w = w.view(-1, 1, 1, 1) if w.dim() == 1 else w
    return (lin >= 0.985 * w).any(1, keepdim=True).float()


def iris_geometry(iris_p: torch.Tensor, pupil_p: torch.Tensor, S: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Centroid and equivalent radius of the apparent iris+pupil disk; falls back to the crop
    centre with a default radius when too little iris is visible. Returns cx, cy, r, visible_frac."""
    m = (iris_p + pupil_p).clamp(0, 1)[:, 0]
    B = m.shape[0]
    ys, xs = torch.meshgrid(torch.arange(S, device=m.device, dtype=torch.float32),
                            torch.arange(S, device=m.device, dtype=torch.float32), indexing="ij")
    area = m.sum((1, 2))
    cx = (m * xs).sum((1, 2)) / area.clamp(min=1e-3)
    cy = (m * ys).sum((1, 2)) / area.clamp(min=1e-3)
    r = torch.sqrt(area / math.pi)
    ok = area > (0.01 * S * S)
    default_r = torch.full_like(r, S / 7.0)
    cx = torch.where(ok, cx, torch.full_like(cx, S / 2 - 0.5))
    cy = torch.where(ok, cy, torch.full_like(cy, S / 2 - 0.5))
    r = torch.where(ok, torch.maximum(r, torch.full_like(r, 3.0)), default_r)
    return cx, cy, r, area / (math.pi * (S / 7.0) ** 2)


def coord_maps(cx, cy, r, S: int) -> torch.Tensor:
    ys, xs = torch.meshgrid(torch.arange(S, device=cx.device, dtype=torch.float32),
                            torch.arange(S, device=cx.device, dtype=torch.float32), indexing="ij")
    u = (xs[None] - cx.view(-1, 1, 1)) / r.view(-1, 1, 1)
    v = (ys[None] - cy.view(-1, 1, 1)) / r.view(-1, 1, 1)
    return torch.stack([u, v], 1).clamp(-3, 3) / 3.0


def guard_input(crop_w: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
    S = crop_w.shape[-1]
    tok = (torch.log2(e) / 10.0).view(-1, 1, 1, 1).expand(-1, 1, S, S)
    return torch.cat([normalize(crop_w, e), tok], 1)


def restore_input(crop_w: torch.Tensor, e: torch.Tensor, guard_probs: torch.Tensor, clipped: torch.Tensor) -> torch.Tensor:
    """guard_probs: (B,6,S,S) sigmoid of EyeGuard masks (ocular, iris, pupil, protected, reflection, clipped)."""
    S = crop_w.shape[-1]
    cx, cy, r, _ = iris_geometry(guard_probs[:, 1:2], guard_probs[:, 2:3], S)
    tok = (torch.log2(e) / 10.0).view(-1, 1, 1, 1).expand(-1, 1, S, S)
    return torch.cat([normalize(crop_w, e), guard_probs[:, 0:5], clipped, tok, coord_maps(cx, cy, r, S)], 1)


def hole_mask(guard_probs: torch.Tensor, thresh: float = 0.5, dilate_px: int = 2) -> torch.Tensor:
    """Round 23 inference hole: the guard's reflection channel above ``thresh``, dilated ``dilate_px``
    crop pixels.  (B,1,S,S) float in {0,1}.  Measured on the plate: covers 94.5% of the true catchlight
    at 3.75x its area."""
    m = (guard_probs[:, 4:5] > thresh).float()
    if dilate_px > 0:
        m = torch.nn.functional.max_pool2d(m, 2 * dilate_px + 1, stride=1, padding=dilate_px)
    return (m > 0.5).float()


def hole_restore_input(crop_w: torch.Tensor, e: torch.Tensor, guard_probs: torch.Tensor, clipped: torch.Tensor,
                       hole: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Masked-inpainting restorer input (gate restore_input "hole"): blank the source under ``hole`` and present the
    hole as the reflection channel.

    The blank is the per-sample mean of the visible ocular region (working space, per channel) - a flat patch at the
    eye's own level, not a black hole with a hard edge to learn around.  The clipped channel is cleared inside the
    hole (nothing is clipped in a blank).  Returns the restorer input and the blanked crop; the residual is predicted
    relative to the BLANKED crop, so inside the hole it has to be the whole fill.  Training and inference both build
    their input here, so the two cannot drift apart.
    """
    vis = (guard_probs[:, 0:1] > 0.5).float() * (1.0 - hole)
    fallback = (1.0 - hole)
    num = (crop_w * vis).sum((2, 3), keepdim=True)
    den = vis.sum((2, 3), keepdim=True)
    num_f = (crop_w * fallback).sum((2, 3), keepdim=True)
    den_f = fallback.sum((2, 3), keepdim=True).clamp(min=1)
    mean = torch.where(den > 0, num / den.clamp(min=1), num_f / den_f)
    x_h = crop_w * (1.0 - hole) + mean * hole
    probs_h = guard_probs.clone()
    probs_h[:, 4:5] = hole
    clip_h = clipped * (1.0 - hole)
    return restore_input(x_h, e, probs_h, clip_h), x_h


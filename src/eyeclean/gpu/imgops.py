"""CUDA image operators used by the production path (no NumPy/OpenCV).

Finite-support morphology, box/Gaussian filters, antialiased resizing and similarity-transform
crop/inverse warps built on torch CUDA kernels.  CPU reference versions live only in tests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


def _as4(x: torch.Tensor) -> tuple[torch.Tensor, int]:
    d = x.dim()
    if d == 2:
        return x[None, None], 2
    if d == 3:
        return x[None], 3
    return x, 4


def _back(x: torch.Tensor, d: int) -> torch.Tensor:
    return x[0, 0] if d == 2 else (x[0] if d == 3 else x)


def dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """Binary dilation with a (2r+1) square structuring element (finite support)."""
    if radius <= 0:
        return mask.clone()
    x, d = _as4(mask.float())
    y = F.max_pool2d(x, 2 * radius + 1, stride=1, padding=radius)
    return _back(y, d) > 0.5


def erode(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """Binary erosion; pixels outside the frame count as background (erodes at borders)."""
    if radius <= 0:
        return mask.clone()
    x, d = _as4(mask.float())
    x = F.pad(x, (radius,) * 4, value=0.0)
    y = -F.max_pool2d(-x, 2 * radius + 1, stride=1, padding=0)
    return _back(y, d) > 0.5


def disk_erode(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """Approximate disk erosion by iterated 3x3 cross/square alternation."""
    out = mask
    for i in range(radius):
        x, d = _as4(out.float())
        x = F.pad(x, (1, 1, 1, 1), value=0.0)
        if i % 2 == 0:
            k = torch.tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=x.dtype, device=x.device)
            s = F.conv2d(x, k[None, None])
            out = _back(s, d) > 4.5
        else:
            y = -F.max_pool2d(-x, 3, stride=1)
            out = _back(y, d) > 0.5
    return out


def box_blur(x: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return x.clone()
    t, d = _as4(x.float())
    C = t.shape[1]
    k = 2 * radius + 1
    t = F.pad(t, (radius, radius, 0, 0), mode="replicate")
    t = F.conv2d(t, torch.full((C, 1, 1, k), 1.0 / k, device=t.device), groups=C)
    t = F.pad(t, (0, 0, radius, radius), mode="replicate")
    t = F.conv2d(t, torch.full((C, 1, k, 1), 1.0 / k, device=t.device), groups=C)
    return _back(t, d)


def gaussian_kernel1d(sigma: float, device, truncate: float = 3.0) -> torch.Tensor:
    r = max(1, int(math.ceil(truncate * sigma)))
    xs = torch.arange(-r, r + 1, device=device, dtype=torch.float32)
    k = torch.exp(-0.5 * (xs / max(sigma, 1e-6)) ** 2)
    return k / k.sum()


def gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0.05:
        return x.clone()
    t, d = _as4(x.float())
    C = t.shape[1]
    k = gaussian_kernel1d(sigma, t.device)
    r = (k.numel() - 1) // 2
    t = F.pad(t, (r, r, 0, 0), mode="replicate")
    t = F.conv2d(t, k.view(1, 1, 1, -1).expand(C, 1, 1, -1), groups=C)
    t = F.pad(t, (0, 0, r, r), mode="replicate")
    t = F.conv2d(t, k.view(1, 1, -1, 1).expand(C, 1, -1, 1), groups=C)
    return _back(t, d)


def feather_inside(hard: torch.Tensor, radius: int) -> torch.Tensor:
    """Feather weights A in [0, 1] that are exactly zero outside ``hard`` (finite support):
    A = H * clamp(2*(box(H) - 0.5), 0, 1)."""
    if radius <= 0:
        return hard.float()
    b = box_blur(hard.float(), radius)
    return torch.clamp(2.0 * (b - 0.5), 0.0, 1.0) * hard.float()


def resize(x: torch.Tensor, size: tuple[int, int], mode: str = "bilinear") -> torch.Tensor:
    """Antialiased resize on CUDA. x: (C,H,W) or (N,C,H,W); size=(H,W)."""
    t, d = _as4(x)
    aa = mode in ("bilinear", "bicubic") and (size[0] < t.shape[-2] or size[1] < t.shape[-1])
    if mode == "nearest":
        y = F.interpolate(t, size=size, mode="nearest")
    else:
        y = F.interpolate(t, size=size, mode=mode, align_corners=False, antialias=aa)
    return _back(y, d)


# ----------------------------------------------------------------- similarity crops
@dataclass
class CropTransform:
    """Maps frame pixel coords (x, y) -> crop pixel coords (u, v):
    [u, v] = s * R(-theta) @ ([x, y] - c) + size/2 - 0.5   (pixel centres)."""

    cx: float
    cy: float
    scale: float  # crop pixels per frame pixel
    theta: float  # radians, rotation of the eye axis in the frame
    size: int

    def as_dict(self) -> dict:
        return {"cx": self.cx, "cy": self.cy, "scale": self.scale, "theta": self.theta, "size": self.size}

    def frame_box(self, W: int, H: int, pad: int = 2) -> tuple[int, int, int, int]:
        half = (self.size / 2.0) / self.scale * math.sqrt(2.0) + pad
        x0 = max(0, int(math.floor(self.cx - half)))
        y0 = max(0, int(math.floor(self.cy - half)))
        x1 = min(W, int(math.ceil(self.cx + half)) + 1)
        y1 = min(H, int(math.ceil(self.cy + half)) + 1)
        return x0, y0, x1, y1


def _crop_grid(tf: CropTransform, H: int, W: int, device) -> torch.Tensor:
    """Sampling grid (1, S, S, 2) in normalized frame coords for each crop pixel."""
    S = tf.size
    v, u = torch.meshgrid(torch.arange(S, device=device, dtype=torch.float32),
                          torch.arange(S, device=device, dtype=torch.float32), indexing="ij")
    du = (u + 0.5 - S / 2.0) / tf.scale
    dv = (v + 0.5 - S / 2.0) / tf.scale
    c, s = math.cos(tf.theta), math.sin(tf.theta)
    x = tf.cx + c * du - s * dv
    y = tf.cy + s * du + c * dv
    gx = (x + 0.5) / W * 2.0 - 1.0  # align_corners=False convention
    gy = (y + 0.5) / H * 2.0 - 1.0
    return torch.stack([gx, gy], dim=-1)[None]


def extract_crop(frame: torch.Tensor, tf: CropTransform, mode: str = "bilinear", padding: str = "zeros") -> torch.Tensor:
    """frame: (C, H, W) CUDA -> (C, S, S). Prefilters when downsampling (scale < 0.75).
    Pixels outside the frame are zero (use ``valid_crop`` for the matching validity mask)."""
    C, H, W = frame.shape
    if tf.scale < 0.75:
        # antialias: blur only the ROI that feeds the crop, then sample it
        x0, y0, x1, y1 = tf.frame_box(W, H, pad=int(4 / tf.scale))
        roi = gaussian_blur(frame[:, y0:y1, x0:x1], sigma=0.45 / tf.scale)
        sub = CropTransform(tf.cx - x0, tf.cy - y0, tf.scale, tf.theta, tf.size)
        grid = _crop_grid(sub, roi.shape[1], roi.shape[2], frame.device)
        if padding == "zeros":
            # zero outside the true frame even though the ROI edge was blurred with replication
            valid = F.grid_sample(torch.ones((1, 1, H, W), device=frame.device), _crop_grid(tf, H, W, frame.device),
                                  mode="nearest", padding_mode="zeros", align_corners=False)[0]
            out = F.grid_sample(roi[None], grid, mode=mode, padding_mode="border", align_corners=False)[0]
            return out * valid
        return F.grid_sample(roi[None], grid, mode=mode, padding_mode=padding, align_corners=False)[0]
    grid = _crop_grid(tf, H, W, frame.device)
    return F.grid_sample(frame[None], grid, mode=mode, padding_mode=padding, align_corners=False)[0]


def valid_crop(H: int, W: int, tf: CropTransform, device) -> torch.Tensor:
    """(1, S, S) mask of crop pixels whose preimage lies inside the frame."""
    grid = _crop_grid(tf, H, W, device)
    return F.grid_sample(torch.ones((1, 1, H, W), device=device), grid, mode="bilinear", padding_mode="zeros",
                         align_corners=False)[0]


def inverse_warp(crop: torch.Tensor, tf: CropTransform, box: tuple[int, int, int, int], mode: str = "bilinear") -> torch.Tensor:
    """Map crop-space tensor (C, S, S) back onto the frame ROI ``box`` -> (C, h, w).
    Frame pixels whose preimage falls outside the crop receive 0.  Prefilters when the crop is
    higher resolution than the frame (scale > 1.5)."""
    x0, y0, x1, y1 = box
    dev = crop.device
    src = crop
    if tf.scale > 1.5:
        src = gaussian_blur(crop, sigma=0.45 * tf.scale)
    ys, xs = torch.meshgrid(torch.arange(y0, y1, device=dev, dtype=torch.float32),
                            torch.arange(x0, x1, device=dev, dtype=torch.float32), indexing="ij")
    c, s = math.cos(tf.theta), math.sin(tf.theta)
    dx, dy = xs - tf.cx, ys - tf.cy
    u = tf.scale * (c * dx + s * dy) + tf.size / 2.0 - 0.5
    v = tf.scale * (-s * dx + c * dy) + tf.size / 2.0 - 0.5
    gx = (u + 0.5) / tf.size * 2.0 - 1.0
    gy = (v + 0.5) / tf.size * 2.0 - 1.0
    grid = torch.stack([gx, gy], dim=-1)[None]
    out = F.grid_sample(src[None], grid, mode=mode, padding_mode="zeros", align_corners=False)[0]
    return out


def warp_crop_to_crop(src: torch.Tensor, tf_src: CropTransform, tf_dst: CropTransform, mode: str = "bilinear") -> torch.Tensor:
    """Resample a crop-space tensor (C, S, S) taken with ``tf_src`` into the crop geometry ``tf_dst``
    of another frame of the same track (Round 23: temporal union of the guard's reflection mask).
    Both transforms are similarities onto the same frame, so the mapping is exact up to the eye's
    own motion between the two frames.  Pixels with no preimage are 0."""
    S = tf_dst.size
    v, u = torch.meshgrid(torch.arange(S, device=src.device, dtype=torch.float32),
                          torch.arange(S, device=src.device, dtype=torch.float32), indexing="ij")
    du = (u + 0.5 - S / 2.0) / tf_dst.scale
    dv = (v + 0.5 - S / 2.0) / tf_dst.scale
    c, s = math.cos(tf_dst.theta), math.sin(tf_dst.theta)
    x = tf_dst.cx + c * du - s * dv
    y = tf_dst.cy + s * du + c * dv
    cs, ss = math.cos(tf_src.theta), math.sin(tf_src.theta)
    dx, dy = x - tf_src.cx, y - tf_src.cy
    us = tf_src.scale * (cs * dx + ss * dy) + tf_src.size / 2.0 - 0.5
    vs = tf_src.scale * (-ss * dx + cs * dy) + tf_src.size / 2.0 - 0.5
    grid = torch.stack([(us + 0.5) / tf_src.size * 2.0 - 1.0, (vs + 0.5) / tf_src.size * 2.0 - 1.0], dim=-1)[None]
    return F.grid_sample(src[None].float(), grid, mode=mode, padding_mode="zeros", align_corners=False)[0]

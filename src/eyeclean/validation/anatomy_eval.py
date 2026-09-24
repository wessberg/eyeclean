"""Independent, non-learned pupil evaluator (validated against render labels before use).

Inside the labelled iris+pupil disk of an eye (evaluation may read labels), the pupil is the
dark connected region containing the labelled pupil centre.  The threshold is the midpoint
between a low luminance percentile (pupil) and the median of the outer iris ring.  It shares no
weights or code with EyeGuard/EyeRestore, so it can score whether predictions damaged the pupil.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from ..gpu.imgops import dilate


def _flood(seed_mask: torch.Tensor, allowed: torch.Tensor, iters: int = 256) -> torch.Tensor:
    cur = seed_mask & allowed
    for _ in range(iters):
        nxt = (F.max_pool2d(cur[None, None].float(), 3, 1, 1)[0, 0] > 0.5) & allowed
        if bool((nxt == cur).all()):
            break
        cur = nxt
    return cur


def pupil_area(lum: torch.Tensor, iris_lab: torch.Tensor, pupil_lab: torch.Tensor) -> float | None:
    """lum: (H,W) luminance crop; *_lab: coverage labels. Returns measured pupil area (px) or None.

    Radial-profile method (robust to dark / textured irises): around the labelled pupil centre,
    the mean luminance of the visible iris+pupil disk is profiled over radius; the pupil edge is
    the first half-way crossing between the pupil core level and the outer iris level, walking
    outward (sub-bin interpolated); the area counts visible disk pixels inside that radius."""
    disk = dilate((iris_lab + pupil_lab) > 0.5, 1)
    if int(disk.sum()) < 30 or float(pupil_lab.sum()) < 4:
        return None
    ys, xs = torch.nonzero(pupil_lab > 0.5, as_tuple=True)
    cy, cx = ys.float().mean(), xs.float().mean()
    H, W = lum.shape
    gy, gx = torch.meshgrid(torch.arange(H, device=lum.device, dtype=torch.float32),
                            torch.arange(W, device=lum.device, dtype=torch.float32), indexing="ij")
    d = torch.sqrt((gy - cy) ** 2 + (gx - cx) ** 2)
    R = float(d[disk].max())
    if R < 4:
        return None
    step = 0.25
    nb = int(R / step) + 1
    idx = (d[disk] / step).long().clamp(max=nb - 1)
    s = torch.zeros(nb, device=lum.device).index_add_(0, idx, lum[disk].float())
    c = torch.zeros(nb, device=lum.device).index_add_(0, idx, torch.ones_like(idx, dtype=torch.float32))
    ok = c > 0
    prof = torch.where(ok, s / c.clamp(min=1), torch.zeros_like(s))
    for i in range(1, nb):  # carry the last value over empty bins
        if not bool(ok[i]):
            prof[i] = prof[i - 1]
    prof = F.avg_pool1d(F.pad(prof[None, None], (2, 2), mode="replicate"), 5, 1)[0, 0]  # 5-bin (1.25 px) box smoothing
    # half-way crossing walking outward from the dark pupil core (robust to collarette/crypt rings
    # further out, which can brighten more steeply than the pupil edge itself)
    lo, mid, hi = max(1, int(0.08 * R / step)), max(2, int(0.5 * R / step)), max(3, int(0.85 * R / step))
    if hi <= mid + 1 or mid <= lo:
        return None
    k0 = int(torch.argmin(prof[lo:mid])) + lo
    p0 = float(prof[k0])
    p1 = float(prof[mid:hi].median())
    if p1 <= p0 * 1.05:
        return None  # no measurable pupil/iris contrast
    thr = 0.5 * (p0 + p1)
    above = (prof[k0:hi] >= thr).nonzero()
    if above.numel() == 0:
        return None
    k = int(above[0]) + k0
    a, b = float(prof[k - 1]), float(prof[k])
    frac = (thr - a) / (b - a) if b > a else 0.0
    r_edge = (k - 1 + frac) * step + 0.5 * step  # bin centre convention
    return float((disk & (d <= r_edge)).sum())


def pupil_area_threshold(lum: torch.Tensor, iris_lab: torch.Tensor, pupil_lab: torch.Tensor) -> float | None:
    """Earlier threshold + flood-fill method (kept for reference; fails validation on dark CG irises)."""
    disk = dilate((iris_lab + pupil_lab) > 0.5, 1)
    if int(disk.sum()) < 30 or float(pupil_lab.sum()) < 4:
        return None
    vals = lum[disk]
    lo = torch.quantile(vals, 0.05)
    ring = (iris_lab > 0.5) & ~dilate(pupil_lab > 0.5, 2)
    if int(ring.sum()) < 10:
        return None
    hi = lum[ring].median()
    thr = 0.5 * (lo + hi)
    dark = disk & (lum < thr)
    ys, xs = torch.nonzero(pupil_lab > 0.5, as_tuple=True)
    cy, cx = int(ys.float().mean().round()), int(xs.float().mean().round())
    seed = torch.zeros_like(dark)
    r = 1
    seed[max(0, cy - r):cy + r + 1, max(0, cx - r):cx + r + 1] = True
    comp = _flood(seed, dark)
    return float(comp.sum())

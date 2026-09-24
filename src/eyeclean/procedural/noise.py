"""Seeded CUDA noise primitives on polar grids (periodic in the angular axis)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def gen(seed: int, device: torch.device) -> torch.Generator:
    g = torch.Generator(device=device)
    g.manual_seed(int(seed) & 0x7FFFFFFFFFFFFFFF)
    return g


def lattice_noise(g: torch.Generator, R: int, T: int, cells_r: int, cells_t: int, device: torch.device,
                  wrap_t: bool = True) -> torch.Tensor:
    """Smooth value noise on an (R, T) grid, periodic in T. Returns zero-mean, unit-ish std."""
    cells_r = max(2, int(cells_r))
    cells_t = max(2, int(cells_t))
    lat = torch.randn((1, 1, cells_r + 3, cells_t + (3 if not wrap_t else 0)), generator=g, device=device)
    if wrap_t:
        lat = torch.cat([lat[..., -2:], lat, lat[..., :3]], dim=-1)  # circular padding
    # bicubic upsample of the padded lattice then crop the valid window
    up_r = (cells_r + 3) * R // cells_r
    up_t = lat.shape[-1] * T // cells_t
    x = F.interpolate(lat, size=(up_r, up_t), mode="bicubic", align_corners=False)
    r0 = (up_r - R) // 2
    t0 = (2 * T) // cells_t if wrap_t else (up_t - T) // 2
    x = x[..., r0:r0 + R, t0:t0 + T]
    if wrap_t:
        # enforce exact periodicity by blending the seam region (tiny correction)
        x = x.contiguous()
    x = x[0, 0]
    return (x - x.mean()) / (x.std() + 1e-6)


def fbm(g: torch.Generator, R: int, T: int, base_r: int, base_t: int, octaves: int, device: torch.device,
        gain: float = 0.5, lacunarity: float = 2.0) -> torch.Tensor:
    out = torch.zeros((R, T), device=device)
    amp, fr, ft, tot = 1.0, float(base_r), float(base_t), 0.0
    for _ in range(octaves):
        out += amp * lattice_noise(g, R, T, int(fr), int(ft), device)
        tot += amp
        amp *= gain
        fr *= lacunarity
        ft *= lacunarity
    return out / tot


def polar_grid(R: int, T: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    rho = (torch.arange(R, device=device, dtype=torch.float32) + 0.5) / R
    th = (torch.arange(T, device=device, dtype=torch.float32) + 0.5) / T * (2 * math.pi)
    return torch.meshgrid(rho, th, indexing="ij")


def smoothstep(e0: float | torch.Tensor, e1: float | torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    t = torch.clamp((x - e0) / (e1 - e0 + 1e-8), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def angdiff(a: torch.Tensor, b: torch.Tensor | float) -> torch.Tensor:
    d = a - b
    return torch.atan2(torch.sin(d), torch.cos(d))


def srgb_to_linear(c: torch.Tensor) -> torch.Tensor:
    return torch.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

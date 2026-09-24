"""Procedural iris textures on CUDA in polar parameterization.

Rows = normalized radius rho (0 = pupil margin, 1 = iris root hidden under the limbus),
columns = angle theta in [0, 2pi) (periodic).  Output: linear-Rec.709 albedo (3, R, T) and a
relief/height map (1, R, T) used for shading bump.  Everything is seeded and deterministic.

Families are deliberately structurally different, not recolorings of one pattern:
  structure: silky_fibers, crypt_rich, furrowed, spotty, velvet, spoke (held-out test family)
  color:     dark_brown, brown, amber_hazel, green, gray, blue (+ optional central heterochromia)
These are synthetic modeling assumptions, not clinically validated iris anatomy.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import torch

from .noise import angdiff, fbm, gen, lattice_noise, polar_grid, smoothstep

STRUCTURE_FAMILIES = ("silky_fibers", "crypt_rich", "furrowed", "spotty", "velvet", "spoke")
HELDOUT_STRUCTURE = "spoke"
COLOR_FAMILIES = ("dark_brown", "brown", "amber_hazel", "green", "gray", "blue")

# (stroma base, fiber highlight, collarette/pupillary-zone, limbal ring) as linear albedo
_PALETTES = {
    "dark_brown": ((0.030, 0.015, 0.008), (0.060, 0.032, 0.016), (0.045, 0.022, 0.010), (0.012, 0.007, 0.004)),
    "brown": ((0.075, 0.038, 0.017), (0.13, 0.07, 0.032), (0.055, 0.028, 0.012), (0.020, 0.012, 0.007)),
    "amber_hazel": ((0.085, 0.065, 0.025), (0.16, 0.11, 0.04), (0.14, 0.075, 0.02), (0.025, 0.020, 0.010)),
    "green": ((0.055, 0.080, 0.040), (0.10, 0.14, 0.07), (0.11, 0.07, 0.025), (0.015, 0.022, 0.012)),
    "gray": ((0.090, 0.098, 0.105), (0.19, 0.20, 0.21), (0.10, 0.09, 0.07), (0.025, 0.028, 0.032)),
    "blue": ((0.050, 0.085, 0.150), (0.13, 0.18, 0.27), (0.08, 0.08, 0.07), (0.012, 0.018, 0.035)),
}


@dataclass
class IrisParams:
    seed: int
    structure: str
    color: str
    heterochromia: float = 0.0  # 0..1 strength of a distinct pupillary-zone color
    collarette_rho: float = 0.35
    collarette_jag: float = 0.05
    fiber_count: int = 360
    fiber_contrast: float = 0.5
    fiber_wave: float = 0.15
    crypt_count: int = 12
    crypt_strength: float = 0.5
    furrow_count: int = 3
    furrow_strength: float = 0.3
    spot_count: int = 3
    limbal_ring: float = 0.6
    limbal_width: float = 0.08
    brightness: float = 1.0
    hue_jitter: tuple[float, float, float] = (1.0, 1.0, 1.0)
    shared_seed: int = 0  # correlated left/right structure
    shared_weight: float = 0.35
    ruff_width: float = 0.03
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["hue_jitter"] = list(self.hue_jitter)
        return d


def sample_identity_iris(rng_seed: int, allow_heldout: bool, family_override: str | None = None,
                         color_override: str | None = None) -> dict:
    """Sample correlated left/right iris parameters for one identity (CPU RNG, metadata only)."""
    import random

    r = random.Random(rng_seed)
    fams = [f for f in STRUCTURE_FAMILIES if allow_heldout or f != HELDOUT_STRUCTURE]
    if not allow_heldout and family_override == HELDOUT_STRUCTURE:
        raise ValueError("held-out iris family requested for a non-test identity")
    structure = family_override or r.choice(fams)
    color = color_override or r.choices(COLOR_FAMILIES, weights=(3, 3, 2, 2, 1.5, 2.5))[0]
    base = dict(
        structure=structure, color=color,
        heterochromia=r.random() * 0.8 if r.random() < 0.35 else 0.0,
        collarette_rho=r.uniform(0.25, 0.45), collarette_jag=r.uniform(0.02, 0.08),
        fiber_count=r.randint(180, 520), fiber_contrast=r.uniform(0.25, 0.8), fiber_wave=r.uniform(0.03, 0.3),
        crypt_count=r.randint(4, 26), crypt_strength=r.uniform(0.25, 0.85),
        furrow_count=r.randint(1, 6), furrow_strength=r.uniform(0.1, 0.55),
        spot_count=r.randint(0, 8), limbal_ring=r.uniform(0.25, 0.95), limbal_width=r.uniform(0.04, 0.14),
        brightness=r.uniform(0.7, 1.35), ruff_width=r.uniform(0.015, 0.045),
        shared_seed=r.randint(0, 2**31 - 1), shared_weight=r.uniform(0.2, 0.5),
    )
    hue = (r.uniform(0.85, 1.15), r.uniform(0.85, 1.15), r.uniform(0.85, 1.15))
    out = {}
    for eye in ("L", "R"):
        p = dict(base)
        p["seed"] = r.randint(0, 2**31 - 1)
        # correlated but not identical: small per-eye jitter
        p["brightness"] = base["brightness"] * r.uniform(0.94, 1.06)
        p["collarette_rho"] = base["collarette_rho"] + r.uniform(-0.02, 0.02)
        p["crypt_count"] = max(0, base["crypt_count"] + r.randint(-3, 3))
        p["hue_jitter"] = tuple(h * r.uniform(0.97, 1.03) for h in hue)
        out[eye] = p
    return out


def _crypts(g, rho, th, n, strength, zone_lo, zone_hi, device):
    R, T = rho.shape
    field_ = torch.zeros_like(rho)
    if n <= 0:
        return field_
    c_rho = zone_lo + (zone_hi - zone_lo) * torch.rand(n, generator=g, device=device)
    c_th = torch.rand(n, generator=g, device=device) * 2 * math.pi
    s_r = 0.03 + 0.06 * torch.rand(n, generator=g, device=device)  # radial extent
    s_t = 0.025 + 0.05 * torch.rand(n, generator=g, device=device)  # angular extent (radians*rho)
    a = 0.4 + 0.6 * torch.rand(n, generator=g, device=device)
    for i in range(n):
        dr = (rho - c_rho[i]) / s_r[i]
        dt = angdiff(th, c_th[i]) * (0.3 + rho) / s_t[i]
        d2 = dr * dr + dt * dt
        field_ = torch.maximum(field_, a[i] * torch.exp(-d2 * 1.5) * (d2 < 9).float())
    return field_ * strength


def generate_iris(p: IrisParams | dict, R: int = 384, T: int = 2048, device: torch.device | None = None) -> dict:
    """Returns {'albedo': (3,R,T), 'height': (1,R,T), 'params': dict} as CUDA tensors."""
    if isinstance(p, dict):
        p = IrisParams(**{k: v for k, v in p.items() if k in IrisParams.__dataclass_fields__})
    device = device or torch.device("cuda")
    g = gen(p.seed, device)
    gs = gen(p.shared_seed or p.seed, device)
    rho, th = polar_grid(R, T, device)
    fam = p.structure

    def shared(cr, ct, oct_):
        a = fbm(g, R, T, cr, ct, oct_, device)
        b = fbm(gs, R, T, cr, ct, oct_, device)
        return (1 - p.shared_weight) * a + p.shared_weight * b

    # --- radial fibers with wavy trajectories
    wobble = 0.35 * fbm(g, R, T, 5, 20, 2, device) * p.fiber_wave
    fc = p.fiber_count
    if fam == "spoke":
        fc = max(60, fc // 4)
    ridges = torch.zeros_like(rho)
    for k, amp, cr in ((1, 1.0, 3), (2, 0.6, 5), (3, 0.35, 9)):
        ln = lattice_noise(g, R, T, cr, fc * k, device)
        ridges += amp * (1.0 - torch.abs(torch.tanh(ln))) ** 3  # thin radial ridges
    ridges = ridges / 1.95
    # bend fibers along a smooth angular wobble (row-dependent circular shift)
    shift = (wobble / (2 * math.pi) * T).round().long()
    idx = (torch.arange(T, device=device)[None, :] + shift) % T
    ridges = torch.gather(ridges, 1, idx)
    fibers = 2.0 * (ridges - ridges.mean()) / (ridges.std() * 3 + 1e-6)
    fibers = torch.clamp(fibers, -1.0, 1.0)
    if fam == "velvet":
        fibers = 0.5 * fibers + 0.5 * torch.clamp(fbm(g, R, T, 24, 160, 3, device), -1, 1)
    # --- collarette: jagged ring separating pupillary / ciliary zones
    col_r = p.collarette_rho + p.collarette_jag * lattice_noise(g, 1, T, 2, max(8, fc // 20), device)[0][None, :]
    col_ring = torch.exp(-((rho - col_r) / 0.025) ** 2)
    pupil_zone = smoothstep(col_r + 0.02, col_r - 0.02, rho)  # 1 inside collarette
    # --- crypts (dark lacunae), mostly around/beyond the collarette
    crypt_scale = {"crypt_rich": 1.8, "spoke": 1.2, "velvet": 0.3, "silky_fibers": 0.6}.get(fam, 1.0)
    crypts = _crypts(g, rho, th, int(p.crypt_count * crypt_scale), p.crypt_strength, 0.25, 0.75, device)
    if fam == "spoke":
        # held-out family: strong radial spokes of dark crypt channels
        spokes = torch.cos(th * torch.randint(9, 17, (1,), generator=g, device=device).float() + 0.5 * wobble)
        crypts = torch.maximum(crypts, 0.7 * smoothstep(0.75, 0.98, spokes) * smoothstep(0.2, 0.35, rho) * smoothstep(0.85, 0.7, rho))
    # --- contraction furrows (concentric partial arcs, outer iris)
    furrows = torch.zeros_like(rho)
    fn = p.furrow_count * (2 if fam == "furrowed" else 1)
    for i in range(fn):
        rr = 0.55 + 0.38 * (i + 0.5) / max(fn, 1) + 0.02 * lattice_noise(g, 1, T, 2, 12, device)[0][None, :]
        arc = smoothstep(0.1, 0.5, lattice_noise(g, 1, T, 2, 7, device)[0][None, :])
        furrows = torch.maximum(furrows, arc * torch.exp(-((rho - rr) / 0.012) ** 2))
    furrows = furrows * p.furrow_strength * (1.6 if fam == "furrowed" else 1.0)
    # --- pigment spots (nevi)
    spots = torch.zeros_like(rho)
    sc = p.spot_count * (3 if fam == "spotty" else 1)
    for _ in range(sc):
        cr = 0.2 + 0.7 * torch.rand(1, generator=g, device=device)
        ct = torch.rand(1, generator=g, device=device) * 2 * math.pi
        sz = 0.012 + 0.03 * torch.rand(1, generator=g, device=device)
        d2 = ((rho - cr) / sz) ** 2 + (angdiff(th, ct) * (0.3 + rho) / sz) ** 2
        spots = torch.maximum(spots, torch.exp(-d2))
    # --- palette and composition
    base, hi, coll, limb = (torch.tensor(c, device=device).view(3, 1, 1) for c in _PALETTES[p.color])
    hue = torch.tensor(p.hue_jitter, device=device).view(3, 1, 1)
    low = shared(4, 16, 3)
    stroma = base * (1.0 + 0.35 * low) * hue
    fiber_w = p.fiber_contrast * (1.4 if fam == "silky_fibers" else 1.0)
    alb = stroma + (hi * hue - stroma) * (fiber_w * 0.5 * (fibers + 1.0) * 0.6)
    # central heterochromia / pupillary-zone tint
    het = p.heterochromia if p.heterochromia > 0 else 0.25
    alb = alb * (1 - het * pupil_zone) + coll * hue * (1.0 + 0.3 * fibers) * het * pupil_zone
    alb = alb + 0.6 * (hi * hue) * col_ring * 0.5
    alb = alb * (1.0 - 0.75 * crypts)
    alb = alb * (1.0 - 0.35 * furrows)
    spot_col = torch.tensor((0.035, 0.018, 0.008), device=device).view(3, 1, 1)
    alb = alb * (1 - 0.8 * spots) + spot_col * 0.8 * spots
    # limbal ring (darkening toward the root) and pupillary ruff
    lr = smoothstep(1.0 - p.limbal_width - 0.12, 1.0 - 0.06, rho) * p.limbal_ring
    alb = alb * (1 - lr) + limb * lr
    ruff = smoothstep(p.ruff_width, 0.0, rho)
    ruff_col = torch.tensor((0.018, 0.010, 0.006), device=device).view(3, 1, 1)
    alb = alb * (1 - ruff) + ruff_col * ruff
    alb = torch.clamp(alb * p.brightness, 0.003, 0.6)
    height = (0.45 * fibers * (0.5 if fam == "velvet" else 1.0) + 0.9 * col_ring - 1.2 * crypts - 0.6 * furrows
              + 0.15 * low - 0.4 * spots * 0.2)
    height = (height - height.mean()) / (height.std() + 1e-6)
    return {"albedo": alb.float(), "height": height[None].float(), "params": p.as_dict()}


# ----------------------------------------------------------------------- sclera
@dataclass
class ScleraParams:
    seed: int
    base: tuple[float, float, float] = (0.70, 0.65, 0.60)
    vein_density: float = 0.5
    vein_redness: float = 0.5
    limbal_tint: float = 0.3
    yellowing: float = 0.1
    rough_lo: float = 0.03
    rough_hi: float = 0.10

    def as_dict(self) -> dict:
        d = asdict(self)
        d["base"] = list(self.base)
        return d


def sample_sclera(rng_seed: int) -> dict:
    import random

    r = random.Random(rng_seed)
    y = r.uniform(0.0, 0.35)
    b = r.uniform(0.58, 0.78)
    base = (b, b * r.uniform(0.9, 0.97), b * r.uniform(0.82, 0.92) * (1 - 0.3 * y))
    common = dict(base=base, vein_density=r.uniform(0.15, 1.0), vein_redness=r.uniform(0.2, 0.9),
                  limbal_tint=r.uniform(0.1, 0.5), yellowing=y, rough_lo=r.uniform(0.02, 0.05),
                  rough_hi=r.uniform(0.06, 0.14))
    return {eye: dict(common, seed=r.randint(0, 2**31 - 1)) for eye in ("L", "R")}


def generate_sclera(p: ScleraParams | dict, R: int = 256, T: int = 1024, device: torch.device | None = None) -> dict:
    """Polar sclera texture: rows = polar angle from the limbus (0) to the equator (1),
    columns = azimuth (0 = temporal/nasal horizontal).  Returns albedo (3,R,T), roughness (1,R,T)."""
    if isinstance(p, dict):
        p = ScleraParams(**{k: (tuple(v) if k == "base" else v) for k, v in p.items() if k in ScleraParams.__dataclass_fields__})
    device = device or torch.device("cuda")
    g = gen(p.seed, device)
    v, az = polar_grid(R, T, device)
    base = torch.tensor(p.base, device=device).view(3, 1, 1)
    mott = fbm(g, R, T, 5, 20, 4, device)
    alb = base * (1.0 + 0.06 * mott)
    # veins: ridged noise stretched radially + density toward canthi (az ~ 0, pi) and periphery
    canthi = 0.5 + 0.5 * torch.cos(2 * az) ** 2
    periph = smoothstep(0.05, 0.9, v)
    veins = torch.zeros_like(v)
    for k, amp in ((1, 1.0), (2, 0.6), (4, 0.35)):
        n = lattice_noise(g, R, T, 3 * k, 40 * k, device)
        ridge = (1.0 - torch.abs(torch.tanh(1.5 * n))) ** (10 + 4 * k)
        veins = veins + amp * ridge
    veins = veins * canthi * periph * p.vein_density
    vein_col = torch.tensor((0.55, 0.12, 0.10), device=device).view(3, 1, 1)
    w = torch.clamp(veins * (0.5 + p.vein_redness), 0, 0.85)
    alb = alb * (1 - w) + vein_col * base.mean() * w * 1.2
    # diffuse redness near canthi, grey-blue limbal tint near the limbus
    red = torch.tensor((1.0, 0.85, 0.82), device=device).view(3, 1, 1)
    alb = alb * (1 + (red - 1) * 0.35 * canthi * periph * p.vein_redness)
    lt = smoothstep(0.12, 0.0, v) * p.limbal_tint
    limb_col = torch.tensor((0.35, 0.38, 0.42), device=device).view(3, 1, 1)
    alb = alb * (1 - lt) + limb_col * lt
    rough = p.rough_lo + (p.rough_hi - p.rough_lo) * smoothstep(-1.0, 1.0, fbm(g, R, T, 4, 16, 3, device))
    return {"albedo": torch.clamp(alb, 0.02, 0.95).float(), "roughness": rough[None].float(), "params": p.as_dict()}

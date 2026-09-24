"""Persistent per-shot, per-eye-track canonical iris representation (hybrid mode).

A polar atlas (rho from the pupil margin to the iris edge x angle) is built once per shot and
track during a bounded-memory analysis pass: confidence-weighted observations of visible iris
pixels (excluding reflections, clipped pixels and lids/lashes) are unwarped and accumulated.
Unobserved cells take a fixed-seed CUDA procedural prior (eyeclean.procedural.iris) whose three
colour gains are fitted to the observed cells.  The atlas is then FROZEN: every output frame of
the shot renders the same atlas, warped to that frame's gaze/pupil geometry.  It is refit only at
an explicit shot/track boundary, never per frame.  When no usable iris is ever visible the prior
is used with the uncertainty flag set; nothing here recovers the true hidden iris.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

from ..io.exr import write_exr
from ..procedural.iris import IrisParams, generate_iris

R_BINS, T_BINS = 48, 192
# Round 20: the atlas above spans the IRIS ANNULUS only (rho = 0 at the pupil margin), and the pupil
# was represented by a single mean colour - a flat disc, which is exactly the fill the user rejected
# in Round 4 ("you just painted a brown circle in the middle of her eye") and again in Round 20.  The
# pupil now gets its own atlas over rho_p = r / r_pupil, accumulated from real observations the same
# way.  Measured on the plate: the catchlight moves ~5px (std) against the tracked eye, and EVERY
# pupil pixel has at least 117 clean frames somewhere in the 1,200 - so there is real pupil to
# accumulate, which a single frame does not have (26 of 58 eyes keep under 25% of their pupil clean).
P_BINS = 24


@dataclass
class CanonicalState:
    key: str  # "<shot>/<track>"
    seed: int
    sum_w: torch.Tensor  # (R,T)
    sum_wx: torch.Tensor  # (3,R,T)
    pupil_sum: torch.Tensor  # (3,)
    pup_sum_w: torch.Tensor | None = None   # (P,T) weighted observations inside the pupil
    pup_sum_wx: torch.Tensor | None = None  # (3,P,T)
    pupil_atlas: torch.Tensor | None = None  # (3,P,T) frozen pupil texture
    pupil_coverage: torch.Tensor | None = None  # (P,T) in [0,1]
    pupil_w: float = 0.0
    frames_observed: int = 0
    frozen: bool = False
    atlas: torch.Tensor | None = None  # (3,R,T) radiance / exposure
    coverage: torch.Tensor | None = None  # (R,T) in [0,1]
    pupil_color: torch.Tensor | None = None
    gains: list = field(default_factory=list)
    family: str = "velvet"
    uncertain: bool = False
    digest: str = ""

    @classmethod
    def new(cls, key: str, device) -> "CanonicalState":
        seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "little")
        return cls(key, seed, torch.zeros((R_BINS, T_BINS), device=device), torch.zeros((3, R_BINS, T_BINS), device=device),
                   torch.zeros(3, device=device),
                   pup_sum_w=torch.zeros((P_BINS, T_BINS), device=device),
                   pup_sum_wx=torch.zeros((3, P_BINS, T_BINS), device=device))


def polar_coords(S: int, cx: float, cy: float, r_iris: float, r_pupil: float, device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ys, xs = torch.meshgrid(torch.arange(S, device=device, dtype=torch.float32), torch.arange(S, device=device, dtype=torch.float32), indexing="ij")
    dx, dy = xs - cx, ys - cy
    r = torch.sqrt(dx * dx + dy * dy)
    th = torch.remainder(torch.atan2(dy, dx), 2 * math.pi)
    rho = (r - r_pupil) / max(r_iris - r_pupil, 1e-3)
    return r, th, rho


def observe(st: CanonicalState, crop_norm_lin: torch.Tensor, weight: torch.Tensor, pupil_w: torch.Tensor,
            cx: float, cy: float, r_iris: float, r_pupil: float) -> None:
    """crop_norm_lin: (3,S,S) working-space radiance / exposure; weight: (S,S) trust of iris pixels."""
    if st.frozen:
        raise RuntimeError(f"canonical state {st.key} is frozen; refitting is only allowed at shot/track boundaries")
    S = crop_norm_lin.shape[-1]
    r, th, rho = polar_coords(S, cx, cy, r_iris, r_pupil, crop_norm_lin.device)
    ok = (rho >= 0) & (rho < 1) & (weight > 0.05)
    if not bool(ok.any()):
        return
    ri = (rho[ok] * R_BINS).long().clamp(0, R_BINS - 1)
    ti = (th[ok] / (2 * math.pi) * T_BINS).long().clamp(0, T_BINS - 1)
    idx = ri * T_BINS + ti
    w = weight[ok]
    st.sum_w.view(-1).index_add_(0, idx, w)
    for c in range(3):
        st.sum_wx[c].view(-1).index_add_(0, idx, w * crop_norm_lin[c][ok])
    pw = pupil_w * (r < r_pupil * 0.8).float()
    if bool((pw > 0.05).any()):
        st.pupil_sum += (crop_norm_lin * pw).sum((1, 2))
        st.pupil_w += float(pw.sum())
    # Round 20: the pupil's own polar atlas, so the fill is accumulated REAL pupil pixels rather than
    # one mean colour.  Same weights, so reflection/clipped/lid pixels are excluded here too.
    if st.pup_sum_w is not None:
        okp = (r < r_pupil) & (pupil_w > 0.05)
        if bool(okp.any()):
            rp_i = ((r[okp] / max(r_pupil, 1e-3)) * P_BINS).long().clamp(0, P_BINS - 1)
            tp_i = (th[okp] / (2 * math.pi) * T_BINS).long().clamp(0, T_BINS - 1)
            idxp = rp_i * T_BINS + tp_i
            wp = pupil_w[okp]
            st.pup_sum_w.view(-1).index_add_(0, idxp, wp)
            for c in range(3):
                st.pup_sum_wx[c].view(-1).index_add_(0, idxp, wp * crop_norm_lin[c][okp])
    st.frames_observed += 1


def freeze(st: CanonicalState) -> dict:
    """Fit prior gains to observed cells, blend, and freeze. Returns provenance metadata."""
    dev = st.sum_w.device
    obs = st.sum_wx / st.sum_w.clamp(min=1e-6)
    cov = torch.clamp(st.sum_w / 3.0, 0.0, 1.0)  # ~3 weighted samples -> fully observed
    # the family is chosen deterministically from the key seed; the prior is generated at atlas res
    fam = ("silky_fibers", "crypt_rich", "furrowed", "spotty", "velvet")[st.seed % 5]
    pri = generate_iris(IrisParams(seed=st.seed, structure=fam, color="brown", brightness=1.0), R=R_BINS * 4, T=T_BINS * 4, device=dev)["albedo"]
    pri = F.avg_pool2d(pri[None], 4)[0]
    m = cov > 0.3
    if bool(m.sum() > 20):
        gains = [(float((obs[c][m] * pri[c][m]).sum() / (pri[c][m] ** 2).sum().clamp(min=1e-8))) for c in range(3)]
        st.uncertain = bool(m.float().mean() < 0.15)
    else:
        # no usable iris observed: neutral mid-brown prior at the median observed exposure level
        gains = [4.0, 4.0, 4.0]
        st.uncertain = True
    g = torch.tensor(gains, device=dev).view(3, 1, 1)
    prior = pri * g
    st.atlas = cov[None] * obs + (1 - cov[None]) * prior
    st.coverage = cov
    st.pupil_color = st.pupil_sum / max(st.pupil_w, 1e-6) if st.pupil_w > 1 else torch.tensor([0.02, 0.015, 0.012], device=dev)
    if st.pup_sum_w is not None:
        pcov = torch.clamp(st.pup_sum_w / 3.0, 0.0, 1.0)
        pobs = st.pup_sum_wx / st.pup_sum_w.clamp(min=1e-6)
        # uncovered pupil cells fall back to the mean pupil colour - NEVER to the procedural prior,
        # which is an iris texture and has no business inside a pupil
        st.pupil_atlas = pcov[None] * pobs + (1 - pcov[None]) * st.pupil_color.view(3, 1, 1)
        st.pupil_coverage = pcov
    st.gains, st.family, st.frozen = gains, fam, True
    from ..io.rawbytes import cuda_to_bytes

    st.digest = hashlib.sha256(cuda_to_bytes(st.atlas.contiguous().view(torch.uint8).view(-1), "mask_download")).hexdigest()
    return {"key": st.key, "seed": st.seed, "prior_family": fam, "gains": gains, "frames_observed": st.frames_observed,
            "observed_fraction": float((cov > 0.3).float().mean()), "uncertain_prior_only": st.uncertain, "frozen": True}


def render(st: CanonicalState, S: int, cx: float, cy: float, r_iris: float, r_pupil: float, exposure: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Render the frozen atlas into a crop for this frame's geometry. Returns (radiance (3,S,S), mask (S,S))."""
    if not st.frozen:
        raise RuntimeError("canonical state must be frozen before rendering")
    dev = st.atlas.device
    r, th, rho = polar_coords(S, cx, cy, r_iris, r_pupil, dev)
    gx = th / (2 * math.pi) * 2 - 1
    gy = rho.clamp(0, 1) * 2 - 1
    grid = torch.stack([gx, gy], -1)[None]
    a = torch.cat([st.atlas, st.atlas[:, :, :1]], 2)  # wrap seam
    img = F.grid_sample(a[None], grid * torch.tensor([T_BINS / (T_BINS + 1), 1.0], device=dev), mode="bilinear",
                        padding_mode="border", align_corners=False)[0]
    inside_iris = r <= r_iris
    pupil = r < r_pupil
    if st.pupil_atlas is not None:
        pg = torch.stack([th / (2 * math.pi) * 2 - 1, (r / max(r_pupil, 1e-3)).clamp(0, 1) * 2 - 1], -1)[None]
        pa = torch.cat([st.pupil_atlas, st.pupil_atlas[:, :, :1]], 2)  # wrap seam
        pimg = F.grid_sample(pa[None], pg * torch.tensor([T_BINS / (T_BINS + 1), 1.0], device=dev), mode="bilinear",
                             padding_mode="border", align_corners=False)[0]
    else:
        pimg = st.pupil_color.view(3, 1, 1).expand_as(img)
    img = torch.where(pupil[None], pimg, img)
    return img * exposure, inside_iris.float()


def save(st: CanonicalState, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    name = st.key.replace("/", "_")
    p = out_dir / f"{name}_atlas.exr"
    write_exr(p, {"R": st.atlas[0], "G": st.atlas[1], "B": st.atlas[2], "coverage": st.coverage})
    meta = {"key": st.key, "seed": st.seed, "prior_family": st.family, "gains": st.gains, "frames_observed": st.frames_observed,
            "uncertain_prior_only": st.uncertain, "atlas": p.name, "atlas_layout": f"polar rho(0=pupil margin..1=iris edge) x theta, {R_BINS}x{T_BINS}",
            "units": "working-space radiance divided by per-frame crop exposure", "digest": st.digest,
            "provenance": "confidence-weighted visible iris observations over the whole shot; unobserved cells from a fixed-seed "
                          "procedural prior with fitted colour gains; frozen for the shot"}
    (out_dir / f"{name}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta

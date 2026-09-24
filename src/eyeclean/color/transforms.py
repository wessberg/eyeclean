"""CUDA color transforms (FP32).

All functions take/return CUDA tensors laid out (..., 3, H, W) unless noted.  Matrices were
computed offline with colour-science 0.4.7 (Bradford CAT) and are verified against it in
``tests/unit/test_color_reference.py``; colour-science is never used in the production path.

Working space: scene-linear ACEScg (AP1 primaries, ACES white).  Supported input
interpretations (``--input-color``):

* ``linear-acescg``            scene-linear AP1 (identity)
* ``linear-rec709``            scene-linear BT.709 primaries, D65
* ``rec709``                   BT.709-encoded video, *scene-referred* interpretation: inverse
                               BT.709 OETF (camera encoding), BT.709 primaries
* ``rec709-display``           BT.709 primaries with BT.1886 (gamma 2.4) display EOTF
* ``srgb``                     IEC 61966-2-1 piecewise EOTF, BT.709 primaries
* ``panasonic-vlog-vgamut``    Panasonic V-Log curve, V-Gamut primaries (D65)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

# --- primaries matrices (row-major, applied as out = M @ rgb) --------------------------------
M_709_TO_AP1 = (
    (0.613097402401, 0.339523146184, 0.047379451415),
    (0.070193722470, 0.916353879057, 0.013452398473),
    (0.020615592882, 0.109569772938, 0.869814634180),
)
M_AP1_TO_709 = (
    (1.705050992658, -0.621792120657, -0.083258872001),
    (-0.130256417507, 1.140804736575, -0.010548319068),
    (-0.024003356805, -0.128968976065, 1.152972332870),
)
M_VGAMUT_TO_AP1 = (
    (1.048662166908, 0.009552555839, -0.058216322389),
    (-0.029390700595, 1.145805112290, -0.116413795597),
    (-0.003317311881, -0.005607384021, 1.008924915273),
)
M_2020_TO_AP1 = (
    (0.974894977924, 0.019599108637, 0.005505913439),
    (0.002179562798, 0.995535468893, 0.002284968309),
    (0.004797239684, 0.024532016635, 0.970670743682),
)
IDENTITY = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))

# YCbCr luma coefficients (Kr, Kb)
MATRIX_COEFFS = {"bt709": (0.2126, 0.0722), "bt601": (0.299, 0.114), "bt2020": (0.2627, 0.0593)}

INPUT_COLORS = ("linear-acescg", "linear-rec709", "rec709", "rec709-display", "srgb", "panasonic-vlog-vgamut")
WORKING_COLORS = ("linear-acescg", "linear-rec709")


class ColorConfigError(ValueError):
    """Missing/ambiguous color information (exit code 2 with guidance)."""


def _mat(m, dev) -> torch.Tensor:
    return torch.tensor(m, dtype=torch.float32, device=dev)


def apply_matrix(rgb: torch.Tensor, m) -> torch.Tensor:
    """rgb: (..., 3, H, W); m: 3x3 nested tuple or tensor."""
    M = m if isinstance(m, torch.Tensor) else _mat(m, rgb.device)
    return torch.einsum("ij,...jhw->...ihw", M, rgb)


# --- transfer functions -------------------------------------------------------------------
def bt709_oetf(L: torch.Tensor) -> torch.Tensor:
    return torch.where(L < 0.018, 4.5 * L, 1.099 * torch.clamp(L, min=0.018).pow(0.45) - 0.099)


def bt709_inverse_oetf(V: torch.Tensor) -> torch.Tensor:
    return torch.where(V < 0.081, V / 4.5, ((torch.clamp(V, min=0.081) + 0.099) / 1.099).pow(1.0 / 0.45))


def srgb_eotf(V: torch.Tensor) -> torch.Tensor:
    return torch.where(V <= 0.04045, V / 12.92, ((torch.clamp(V, min=0.04045) + 0.055) / 1.055).pow(2.4))


def srgb_inverse_eotf(L: torch.Tensor) -> torch.Tensor:
    return torch.where(L <= 0.0031308, 12.92 * L, 1.055 * torch.clamp(L, min=0.0031308).pow(1.0 / 2.4) - 0.055)


def bt1886_eotf(V: torch.Tensor) -> torch.Tensor:
    return torch.clamp(V, min=0.0).pow(2.4)


def bt1886_inverse_eotf(L: torch.Tensor) -> torch.Tensor:
    return torch.clamp(L, min=0.0).pow(1.0 / 2.4)


# Panasonic V-Log (reference: colour.models.rgb.transfer_functions.panasonic_v_log)
_VLOG_CUT1, _VLOG_CUT2 = 0.01, 0.181
_VLOG_B, _VLOG_C, _VLOG_D = 0.00873, 0.241514, 0.598206


def vlog_encode(L: torch.Tensor) -> torch.Tensor:
    return torch.where(L < _VLOG_CUT1, 5.6 * L + 0.125,
                       _VLOG_C * torch.log10(torch.clamp(L, min=_VLOG_CUT1) + _VLOG_B) + _VLOG_D)


def vlog_decode(V: torch.Tensor) -> torch.Tensor:
    return torch.where(V < _VLOG_CUT2, (V - 0.125) / 5.6,
                       torch.pow(10.0, (torch.clamp(V, min=_VLOG_CUT2) - _VLOG_D) / _VLOG_C) - _VLOG_B)


@dataclass(frozen=True)
class InputInterpretation:
    input_color: str
    input_range: str  # 'full' | 'limited' | 'n/a' (linear float inputs)
    matrix: str  # YCbCr matrix coefficients name or 'n/a'
    working_color: str = "linear-acescg"

    def as_dict(self) -> dict:
        return {"input_color": self.input_color, "input_range": self.input_range,
                "ycbcr_matrix": self.matrix, "working_color": self.working_color}


def encoded_to_linear(rgb_enc: torch.Tensor, input_color: str) -> tuple[torch.Tensor, tuple]:
    """Encoded R'G'B' -> scene-linear in the source primaries. Returns (linear, primaries_to_AP1)."""
    if input_color == "rec709":
        return bt709_inverse_oetf(rgb_enc), M_709_TO_AP1
    if input_color == "rec709-display":
        return bt1886_eotf(rgb_enc), M_709_TO_AP1
    if input_color == "srgb":
        return srgb_eotf(rgb_enc), M_709_TO_AP1
    if input_color == "panasonic-vlog-vgamut":
        return vlog_decode(rgb_enc), M_VGAMUT_TO_AP1
    if input_color == "linear-rec709":
        return rgb_enc, M_709_TO_AP1
    if input_color == "linear-acescg":
        return rgb_enc, IDENTITY
    raise ColorConfigError(f"unsupported input color {input_color!r}; choose one of {INPUT_COLORS}")


def to_working(rgb_enc: torch.Tensor, input_color: str, working_color: str = "linear-acescg") -> torch.Tensor:
    """Single recorded GPU transform from the decoded source to the FP32 working space."""
    lin, m = encoded_to_linear(rgb_enc.float(), input_color)
    if working_color == "linear-acescg":
        return apply_matrix(lin, m)
    if working_color == "linear-rec709":
        # source primaries -> AP1 -> 709 (exact identity when the source is already 709)
        if m is M_709_TO_AP1:
            return lin
        return apply_matrix(apply_matrix(lin, m), M_AP1_TO_709)
    raise ColorConfigError(f"unsupported working color {working_color!r}")


def working_to_linear709(w: torch.Tensor, working_color: str = "linear-acescg") -> torch.Tensor:
    return apply_matrix(w, M_AP1_TO_709) if working_color == "linear-acescg" else w


def linear709_to_working(l709: torch.Tensor, working_color: str = "linear-acescg") -> torch.Tensor:
    return apply_matrix(l709, M_709_TO_AP1) if working_color == "linear-acescg" else l709


# --- YCbCr ----------------------------------------------------------------------------------
def ycbcr_to_rgb(y: torch.Tensor, cb: torch.Tensor, cr: torch.Tensor, bit_depth: int, full_range: bool,
                 matrix: str = "bt709") -> torch.Tensor:
    """Integer-code Y/Cb/Cr planes (same resolution, float or int) -> R'G'B' in [0,1] nominal."""
    kr, kb = MATRIX_COEFFS[matrix]
    kg = 1.0 - kr - kb
    s = float(1 << (bit_depth - 8))
    y = y.float()
    cb = cb.float()
    cr = cr.float()
    if full_range:
        mx = float((1 << bit_depth) - 1)
        Y = y / mx
        Pb = (cb - float(1 << (bit_depth - 1))) / mx
        Pr = (cr - float(1 << (bit_depth - 1))) / mx
    else:
        Y = (y - 16.0 * s) / (219.0 * s)
        Pb = (cb - 128.0 * s) / (224.0 * s)
        Pr = (cr - 128.0 * s) / (224.0 * s)
    R = Y + 2.0 * (1.0 - kr) * Pr
    B = Y + 2.0 * (1.0 - kb) * Pb
    G = (Y - kr * R - kb * B) / kg
    return torch.stack([R, G, B], dim=-3)


def rgb_to_ycbcr(rgb: torch.Tensor, bit_depth: int, full_range: bool, matrix: str = "bt709") -> torch.Tensor:
    """R'G'B' in [0,1] -> integer-valued float Y, Cb, Cr codes (not yet rounded)."""
    kr, kb = MATRIX_COEFFS[matrix]
    kg = 1.0 - kr - kb
    R, G, B = rgb.unbind(-3)
    Y = kr * R + kg * G + kb * B
    Pb = (B - Y) / (2.0 * (1.0 - kb))
    Pr = (R - Y) / (2.0 * (1.0 - kr))
    s = float(1 << (bit_depth - 8))
    if full_range:
        mx = float((1 << bit_depth) - 1)
        c0 = float(1 << (bit_depth - 1))
        return torch.stack([Y * mx, Pb * mx + c0, Pr * mx + c0], dim=-3)
    return torch.stack([16.0 * s + 219.0 * s * Y, 128.0 * s + 224.0 * s * Pb, 128.0 * s + 224.0 * s * Pr], dim=-3)


def upsample_chroma_420(c: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """Bilinear 4:2:0 -> 4:4:4 with MPEG-2/H.264 default siting (horizontally co-sited with
    even luma columns, vertically centred between luma rows).  c: (N, Hc, Wc) float CUDA."""
    N, Hc, Wc = c.shape
    dev = c.device
    # chroma-sample coordinate for each luma position
    xs = torch.arange(W, device=dev, dtype=torch.float32) / 2.0
    ys = (torch.arange(H, device=dev, dtype=torch.float32) - 0.5) / 2.0
    # normalized for grid_sample(align_corners=True): -1..1 maps to sample centres 0..Wc-1
    gx = (xs / max(Wc - 1, 1)) * 2.0 - 1.0
    gy = (ys / max(Hc - 1, 1)) * 2.0 - 1.0
    grid = torch.stack(torch.meshgrid(gy, gx, indexing="ij")[::-1], dim=-1).unsqueeze(0).expand(N, H, W, 2)
    out = torch.nn.functional.grid_sample(c.unsqueeze(1), grid, mode="bilinear", padding_mode="border", align_corners=True)
    return out[:, 0]


def downsample_chroma_420(c: torch.Tensor) -> torch.Tensor:
    """4:4:4 -> 4:2:0 matching the siting above: vertical 2-tap average, horizontal [1,2,1]/4
    centred on even columns.  c: (N, H, W) with even H, W."""
    v = 0.5 * (c[:, 0::2, :] + c[:, 1::2, :])
    centre = v[:, :, 0::2]
    right = v[:, :, 1::2]
    left = torch.cat([v[:, :, :1], v[:, :, 1::2][:, :, :-1]], dim=2)  # column 2k-1 (clamped at k=0)
    return 0.25 * left + 0.5 * centre + 0.25 * right


# --- model mapping: invertible exposure-aware asinh -----------------------------------------
ASINH_ALPHA = 8.0


def exposure_reference(rgb_work: torch.Tensor, weight: torch.Tensor | None = None, q: float = 0.75) -> torch.Tensor:
    """Robust exposure reference e (per batch item) from luminance: the q-quantile of positive
    luminance (optionally weighted region). rgb_work: (B, 3, H, W). Returns (B,)."""
    lum = 0.2722 * rgb_work[:, 0] + 0.6741 * rgb_work[:, 1] + 0.0537 * rgb_work[:, 2]  # AP1 luminance weights
    B = lum.shape[0]
    flat = lum.reshape(B, -1)
    if weight is not None:
        w = weight.reshape(B, -1) > 0.5
        big = torch.where(w, flat, torch.full_like(flat, float("nan")))
        e = torch.nanquantile(big, q, dim=1)
        e = torch.where(torch.isfinite(e), e, torch.quantile(flat, q, dim=1))
    else:
        e = torch.quantile(flat, q, dim=1)
    return torch.clamp(e, min=1e-4)


def asinh_forward(x: torch.Tensor, e: torch.Tensor, alpha: float = ASINH_ALPHA) -> torch.Tensor:
    """x: (B, C, H, W) linear; e: (B,) exposure reference. y = asinh(alpha x/e) / asinh(alpha)."""
    s = e.view(-1, *([1] * (x.dim() - 1)))
    return torch.asinh(alpha * x / s) / math.asinh(alpha)


def asinh_inverse(y: torch.Tensor, e: torch.Tensor, alpha: float = ASINH_ALPHA) -> torch.Tensor:
    s = e.view(-1, *([1] * (y.dim() - 1)))
    return s * torch.sinh(y * math.asinh(alpha)) / alpha


# --- detector proxy branch (display-referred; never written back to the reference) ----------
def detector_proxy(rgb_work: torch.Tensor, working_color: str = "linear-acescg", exposure: float | torch.Tensor | None = None) -> torch.Tensor:
    """Display-referred [0,1] sRGB-like proxy for face detection/landmarks (separate branch)."""
    l709 = working_to_linear709(rgb_work, working_color)
    if exposure is None:
        # mid-grey on the median, but never push the brightest 3% of the frame (typically a lit
        # face against a darker set) beyond ~0.85 display-linear
        lum = 0.2126 * l709[..., 0, :, :] + 0.7152 * l709[..., 1, :, :] + 0.0722 * l709[..., 2, :, :]
        sub = lum.reshape(-1)[:: max(1, lum.numel() // 262144)]
        q = torch.quantile(sub, torch.tensor([0.5, 0.97], device=sub.device))
        exposure = torch.minimum(0.18 / torch.clamp(q[0], min=1e-4), 0.85 / torch.clamp(q[1], min=1e-4))
    x = torch.clamp(l709 * exposure, 0.0, 1.0)
    return srgb_inverse_eotf(x)

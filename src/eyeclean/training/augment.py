"""Paired GPU camera simulation (training augmentation and demo-input construction).

Physically ordered chain applied identically to source and target radiance (scene-linear
Rec.709), with the *same* noise realization for both:
  optics (defocus PSF, linear motion integration) -> exposure -> white balance -> sensor shot/read
  noise -> highlight clipping at the white level -> encoding (BT.709 OETF / sRGB / linear) ->
  quantisation (8/10-bit) -> chroma subsampling (4:2:0 / 4:2:2) -> mild sharpening/ringing ->
  decode back to the scene-linear ACEScg working space (what the inference decoder produces).
Radiance spread by optics beyond the authorised ocular support (e.g. bloom onto skin) is kept in
the target: the target equals the source outside the recomputed support.
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, replace

import torch
import torch.nn.functional as F

from ..color import transforms as CT
from ..gpu.imgops import gaussian_blur


@dataclass
class CamParams:
    defocus_sigma: float = 0.0
    motion_len: float = 0.0
    motion_angle: float = 0.0
    exposure: float = 1.0
    wb: tuple = (1.0, 1.0, 1.0)
    shot_a: float = 0.0
    read_b: float = 0.0
    white: float = 1.0
    encoding: str = "rec709"  # rec709 | srgb | linear
    bits: int = 10
    chroma: str = "420"  # 420 | 422 | 444
    sharpen: float = 0.0
    saturate_forced: bool = False
    # v2 (data.camera_sim): camera knee instead of a hard clip, and block-DCT codec quantisation
    knee: float = 0.0  # 0 = hard clip at white; else knee point (linear fraction of white)
    clip_in: float = 1.0  # input level (fraction of white) reaching the top code value when knee > 0
    codec_q: float = 0.0  # 0 = off; else scale of the JPEG-style quantisation tables (1.0 ~ quality 50)
    codec_off: tuple = (0, 0)  # block-grid offset (rows, cols)
    mono: tuple | None = None  # monochrome: per-channel luma weights, applied to source and target alike
    # Round 22 (data.camera_sim.flare_prob): veiling glare / a lifted black floor, per channel, as a
    # fraction of white.  Added to source and target ALIKE after exposure, so it is never part of
    # what the model is asked to remove.  Both banks had true-zero blacks; real footage never does
    # (the plate's pupil sits at 0.05-0.08 display), and a uniform lift of 0.012 linear on held-out
    # NEGATIVES was enough to make the Round 21 model darken 66% of the eye.
    flare: tuple = (0.0, 0.0, 0.0)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["wb"] = list(self.wb)
        d["codec_off"] = list(self.codec_off)
        d["flare"] = list(self.flare)
        return d

    @staticmethod
    def from_dict(d: dict) -> "CamParams":
        return CamParams(**{**d, "wb": tuple(d["wb"]), "codec_off": tuple(d.get("codec_off", (0, 0))),
                            "flare": tuple(d.get("flare", (0.0, 0.0, 0.0)))})


def sample_params(r: random.Random, force_saturate: bool = False, sim: dict | None = None) -> CamParams:
    """sim = config data.camera_sim (v2) or None (the original prototype draws, unchanged)."""
    enc = r.choices(["rec709", "srgb", "linear"], weights=[0.6, 0.15, 0.25])[0]
    p = CamParams(
        defocus_sigma=r.uniform(0.0, 1.4) if r.random() < 0.5 else 0.0,
        motion_len=r.uniform(0.0, 5.0) if r.random() < 0.25 else 0.0,
        motion_angle=r.uniform(0, math.pi),
        exposure=1.0, wb=(r.uniform(0.85, 1.15), 1.0, r.uniform(0.85, 1.15)),
        shot_a=r.uniform(0.0, 0.004) if r.random() < 0.8 else 0.0, read_b=r.uniform(0.0, 6e-5),
        white=1.0, encoding=enc, bits=r.choice([8, 10]) if enc != "linear" else 32,
        chroma=r.choices(["420", "422", "444"], weights=[0.7, 0.1, 0.2])[0] if enc != "linear" else "444",
        sharpen=r.uniform(0.2, 0.8) if (enc != "linear" and r.random() < 0.2) else 0.0,
        saturate_forced=force_saturate)
    if sim:
        if r.random() < sim["highlight_rolloff_prob"]:
            p.knee, p.clip_in = r.uniform(0.35, 0.8), r.uniform(2.0, 8.0)
        if force_saturate:  # a forced-saturation sample must actually clip: hard clip at white
            p.knee, p.clip_in = 0.0, 1.0
        if r.random() < sim.get("monochrome_prob", 0.0):
            # black-and-white delivery: a fixed channel mix, not necessarily Rec.709 luma - stills
            # and broadcast cameras use a range of mixes, and some sensors are natively mono
            w = [r.uniform(0.15, 0.35), r.uniform(0.45, 0.75), r.uniform(0.03, 0.20)]
            t = sum(w)
            p.mono = tuple(x / t for x in w)
        if enc != "linear" and r.random() < sim["codec_prob"]:
            p.codec_q = math.exp(r.uniform(math.log(0.25), math.log(1.6)))
            p.codec_off = (r.randrange(8), r.randrange(8))
        if "flare_prob" in sim and r.random() < float(sim["flare_prob"]):
            # log-uniform 0.002-0.04 of white: +0.01 to +0.18 display on a black pupil, bracketing
            # the plate's 0.05-0.08.  Mildly tinted - flare takes the colour of the scene, not grey.
            b = math.exp(r.uniform(math.log(0.002), math.log(0.04)))
            p.flare = (b * r.uniform(0.85, 1.15), b, b * r.uniform(0.85, 1.15))
    return p


def _motion_kernel(length: float, angle: float, dev) -> torch.Tensor:
    L = max(1, int(math.ceil(length)))
    k = 2 * L + 1
    ys, xs = torch.meshgrid(torch.arange(k, device=dev, dtype=torch.float32) - L,
                            torch.arange(k, device=dev, dtype=torch.float32) - L, indexing="ij")
    d = (xs * math.sin(angle) - ys * math.cos(angle)).abs()
    along = (xs * math.cos(angle) + ys * math.sin(angle)).abs()
    ker = torch.clamp(1 - d, 0, 1) * (along <= length / 2 + 0.5).float()
    return ker / ker.sum()


def optics(x: torch.Tensor, p: CamParams) -> torch.Tensor:
    """x: (C,H,W) linear. Linear operators: blur(clean + refl) = blur(clean) + blur(refl)."""
    if p.defocus_sigma > 0.05:
        x = gaussian_blur(x, p.defocus_sigma)
    if p.motion_len > 0.5:
        k = _motion_kernel(p.motion_len, p.motion_angle, x.device)
        r = k.shape[0] // 2
        C = x.shape[0]
        x = F.conv2d(F.pad(x[None], (r, r, r, r), mode="replicate"), k[None, None].expand(C, 1, -1, -1), groups=C)[0]
    return x


def choose_exposure(clean: torch.Tensor, refl: torch.Tensor, ocular: torch.Tensor, r: random.Random, p: CamParams,
                    valid: torch.Tensor | None = None, sim: dict | None = None, labels: torch.Tensor | None = None) -> float:
    """Exposure from the median luminance of the *rendered* pixels.  valid (H,W) bool excludes
    zero padding outside the frame: close-ups near the frame edge can have most of the native
    crop outside the render, where a median over padding would be 0 and blow the crop out.
    v2 (sim): the crop median lands log-uniformly in [exposure_midgray_min, _max] of white
    (real graded footage puts skin near 0.3-0.5 linear, brighter than the v1 range)."""
    lum = 0.2126 * clean[0] + 0.7152 * clean[1] + 0.0722 * clean[2]
    vals = lum[valid] if valid is not None and int(valid.sum()) >= 64 else lum.flatten()
    med = float(torch.quantile(vals[:: max(1, vals.numel() // 65536)], 0.5).clamp(min=1e-5))
    k = None
    if sim and labels is not None and r.random() < float(sim.get("sclera_exposure_prob", 0.0)):
        # expose for the scene, not for the skin: the visible sclera (near-white for everyone) lands at
        # 25-75% of white, so darker skin stays darker instead of being normalised to light-skin levels
        scl = (labels[0] > 0.5) & (labels[3] < 0.2) & (labels[4] < 0.2)
        if valid is not None:
            scl = scl & valid
        if int(scl.sum()) >= 30:
            smed = float(lum[scl].median().clamp(min=1e-5))
            k = math.exp(r.uniform(math.log(0.25), math.log(0.75))) / smed
            # a camera operator never lets the face median clip or vanish: crop median in [4%, 55%] of white
            k = min(max(k, 0.04 / med), 0.55 / med)
    if k is None and sim:
        k = math.exp(r.uniform(math.log(sim["exposure_midgray_min"]), math.log(sim["exposure_midgray_max"]))) / med
    elif k is None:
        k = r.uniform(0.07, 0.3) / med
    if p.saturate_forced and ocular.any():
        src_max = float(((clean + refl) * k).amax(0)[ocular].max())
        if src_max < 1.3:
            k = min(k * 4.0, k * r.uniform(1.3, 3.5) / max(src_max, 1e-6))
    return k


def _encode(v: torch.Tensor, enc: str) -> torch.Tensor:
    if enc == "rec709":
        return CT.bt709_oetf(v)
    if enc == "srgb":
        return CT.srgb_inverse_eotf(v)
    return v


def _decode(v: torch.Tensor, enc: str) -> torch.Tensor:
    if enc == "rec709":
        return CT.bt709_inverse_oetf(v)
    if enc == "srgb":
        return CT.srgb_eotf(v)
    return v


_JPEG_LUMA = ((16, 11, 10, 16, 24, 40, 51, 61), (12, 12, 14, 19, 26, 58, 60, 55), (14, 13, 16, 24, 40, 57, 69, 56),
              (14, 17, 22, 29, 51, 87, 80, 62), (18, 22, 37, 56, 68, 109, 103, 77), (24, 35, 55, 64, 81, 104, 113, 92),
              (49, 64, 78, 87, 103, 121, 120, 101), (72, 92, 95, 98, 112, 100, 103, 99))
_JPEG_CHROMA = ((17, 18, 24, 47, 99, 99, 99, 99), (18, 21, 26, 66, 99, 99, 99, 99), (24, 26, 56, 99, 99, 99, 99, 99),
                (47, 66, 99, 99, 99, 99, 99, 99)) + ((99,) * 8,) * 4


def _dct8(dev) -> torch.Tensor:
    k = torch.arange(8, device=dev, dtype=torch.float32)[:, None]
    n = torch.arange(8, device=dev, dtype=torch.float32)[None, :]
    d = torch.cos(math.pi * (2 * n + 1) * k / 16) * math.sqrt(2 / 8)
    d[0] = math.sqrt(1 / 8)
    return d


def block_dct_quantize(plane: torch.Tensor, table, scale: float, bits: int, off: tuple[int, int]) -> torch.Tensor:
    """Intra-codec style degradation of one code-value plane (H,W): 8x8 DCT on a grid shifted by
    ``off``, coefficients quantised with a JPEG table (x scale, in code units of ``bits``), inverse
    DCT.  Approximates H.264/JPEG blocking and the loss of fine detail at small eye sizes."""
    H, W = plane.shape
    oy, ox = off
    ph, pw = (-(H + oy)) % 8, (-(W + ox)) % 8
    x = F.pad(plane[None, None], (ox, pw, oy, ph), mode="replicate")[0, 0]
    Hb, Wb = x.shape[0] // 8, x.shape[1] // 8
    blk = x.view(Hb, 8, Wb, 8).permute(0, 2, 1, 3)
    D = _dct8(plane.device)
    q = torch.tensor(table, device=plane.device, dtype=torch.float32) * (scale * (1 << (bits - 8)))
    coef = D @ blk @ D.T
    coef = torch.round(coef / q) * q
    rec = (D.T @ coef @ D).permute(0, 2, 1, 3).reshape(Hb * 8, Wb * 8)
    return rec[oy:oy + H, ox:ox + W]


def _quant_chroma(e: torch.Tensor, p: CamParams) -> torch.Tensor:
    """e: (3,H,W) encoded R'G'B' in [0,1] -> YCbCr codes, chroma subsampling, (v2: block-DCT codec)
    quantisation, back."""
    if p.encoding == "linear":
        return e
    ycc = CT.rgb_to_ycbcr(e, p.bits, full_range=False, matrix="bt709")  # (3,H,W) codes
    H, W = e.shape[-2:]
    codec = (lambda pl, tab: block_dct_quantize(pl, tab, p.codec_q, p.bits, tuple(p.codec_off))) if p.codec_q > 0 else None
    y = codec(ycc[0], _JPEG_LUMA) if codec else ycc[0]
    if p.chroma in ("420", "422") and H % 2 == 0 and W % 2 == 0:
        c = ycc[1:3]
        if p.chroma == "420":
            cs = CT.downsample_chroma_420(c)
            if codec:
                cs = torch.stack([codec(cs[0], _JPEG_CHROMA), codec(cs[1], _JPEG_CHROMA)])
            cs = torch.round(cs)
            up = CT.upsample_chroma_420(cs, H, W)
        else:
            cs = 0.5 * (c[:, :, 0::2] + c[:, :, 1::2])
            if codec:
                cs = torch.stack([codec(cs[0], _JPEG_CHROMA), codec(cs[1], _JPEG_CHROMA)])
            cs = torch.round(cs)
            up = F.interpolate(cs[None], size=(H, W), mode="bilinear", align_corners=False)[0]
        ycc = torch.cat([torch.round(y[None]), up], 0)
    else:
        c = ycc[1:3]
        if codec:
            c = torch.stack([codec(c[0], _JPEG_CHROMA), codec(c[1], _JPEG_CHROMA)])
        ycc = torch.round(torch.cat([y[None], c], 0))
    mx = float((1 << p.bits) - 1)
    ycc = ycc.clamp(0, mx)
    return CT.ycbcr_to_rgb(ycc[0], ycc[1], ycc[2], p.bits, full_range=False, matrix="bt709")


def rolloff(x: torch.Tensor, k0: float, c: float) -> torch.Tensor:
    """Monotone highlight shoulder on x (fraction of white, >= 0): identity up to k0, then an
    exponential approach that reaches exactly 1 at x = c (clip beyond).  Slope ~1 at the knee."""
    w = 1.0 - k0
    norm = 1.0 - math.exp(-(c - k0) / w)
    y = k0 + w * (1.0 - torch.exp(-(x - k0).clamp(min=0) / w)) / norm
    return torch.where(x <= k0, x, torch.where(x >= c, torch.ones_like(x), y))


def _sharpen(e: torch.Tensor, amount: float) -> torch.Tensor:
    if amount <= 0:
        return e
    lum = 0.2126 * e[0] + 0.7152 * e[1] + 0.0722 * e[2]
    detail = lum - gaussian_blur(lum, 1.0)
    return e + amount * detail[None]


def simulate_pair(clean: torch.Tensor, refl: torch.Tensor, ocular: torch.Tensor, p: CamParams, gen: torch.Generator | None,
                  negative: bool = False, working_color: str = "linear-acescg") -> dict:
    """clean/refl: (3,H,W) scene-linear Rec.709; ocular: (H,W) coverage. Returns working-space
    source/target (3,H,W), clipped mask (H,W) and the authorised support after optics."""
    src_rad = clean if negative else clean + refl
    src = optics(src_rad, p)
    tgt = optics(clean, p)
    oc_blur = optics(ocular[None].float(), p)[0]
    wb = torch.tensor(p.wb, device=clean.device).view(3, 1, 1)
    src = src * p.exposure * wb
    tgt = tgt * p.exposure * wb
    if any(p.flare):
        fl = torch.tensor(p.flare, device=clean.device, dtype=src.dtype).view(3, 1, 1) * p.white
        src, tgt = src + fl, tgt + fl
    if p.shot_a > 0 or p.read_b > 0:
        n = torch.randn(src.shape, device=src.device, generator=gen)  # one realization for both
        src = src + torch.sqrt(torch.clamp(p.shot_a * src, min=0) + p.read_b) * n
        tgt = tgt + torch.sqrt(torch.clamp(p.shot_a * tgt, min=0) + p.read_b) * n
    if p.knee > 0:
        # camera knee: linear below the knee, exponential shoulder reaching white at clip_in (paired)
        clipped = (src >= p.white * p.clip_in).any(0)
        src = p.white * rolloff(src.clamp(min=0.0) / p.white, p.knee, p.clip_in)
        tgt = p.white * rolloff(tgt.clamp(min=0.0) / p.white, p.knee, p.clip_in)
    else:
        clipped = (src >= p.white).any(0)
        src = src.clamp(0.0, p.white)
        tgt = tgt.clamp(0.0, p.white)
    if p.mono is not None:
        # identical mix for source and target, so the pair still differs only by the reflection
        mw = torch.tensor(p.mono, device=src.device, dtype=src.dtype).view(3, 1, 1)
        src = (src * mw).sum(0, keepdim=True).expand(3, -1, -1)
        tgt = (tgt * mw).sum(0, keepdim=True).expand(3, -1, -1)
    se, te = _encode(src / p.white, p.encoding), _encode(tgt / p.white, p.encoding)
    se, te = _sharpen(se, p.sharpen), _sharpen(te, p.sharpen)
    if p.codec_q > 0:
        # codec error is shared by source and target (like the noise realization): the pair then
        # differs only by the reflection instead of by whole re-quantised 8x8 blocks around it
        p0 = replace(p, codec_q=0.0)
        s0, t0 = _quant_chroma(se, p0), _quant_chroma(te, p0)
        sq = _quant_chroma(se, p)
        se, te = sq, sq + (t0 - s0)
    else:
        se, te = _quant_chroma(se, p), _quant_chroma(te, p)
    src_lin = _decode(se, p.encoding) * p.white
    tgt_lin = _decode(te, p.encoding) * p.white
    src_w = CT.linear709_to_working(src_lin, working_color)
    tgt_w = CT.linear709_to_working(tgt_lin, working_color)
    auth = oc_blur > 0.02
    tgt_w = torch.where(auth[None], tgt_w, src_w)  # target == source outside the authorised support
    return {"src": src_w, "tgt": tgt_w, "clipped": clipped, "authorised": auth}

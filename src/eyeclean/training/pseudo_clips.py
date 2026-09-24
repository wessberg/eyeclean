"""Round 35: temporal sequences with exact targets, built from single-frame CG stills and rendered clips.

Why (development round 34)
--------------------------
The boil is the single-frame restorer re-synthesising the fill's VALUES every frame from slightly
different input (noise, sub-pixel motion, a catchlight that slides), and only temporal information
can pin those values down.  The main CG bank is all stills (Round 33), so the temporal mechanism is
tested first on PSEUDO-CLIPS made from them, before any clip bank is rendered: the cache stores the
clean eye and the reflection layer separately, so a sequence of frames from one still can move the
eye, move the catchlight independently of it and draw fresh sensor noise, with an exact target for
every frame.

Geometry of a frame
-------------------
Two transforms per frame, as in `data.build_crop`:

  tf1  render -> native: the CAMERA.  For a pseudo-clip it moves relative to the static rendered eye
       along the eye-motion trajectory, so the native pixel grid slides over the eye by sub-pixel
       amounts every frame - which is what a real camera does to a moving eye (aliasing, noise and
       codec phase all change).  For a RENDERED clip it is fixed for the whole clip and the eye moves
       inside it (`build_crop` instead re-centres the native grid on the eye every frame, which locks
       the sampling phase to the eye and understates exactly this variation).
  tf2  native -> crop: the TRACKER.  Centred on the eye's true native position plus the fixed framing
       offset plus tracking noise; the noise is smoothed over +-1 frame and the size is held per
       sequence, as `inference.pipeline._smooth_tracks` does (3-frame mean centre, 7-frame median size).

The catchlight (the `key` part of the reflection layer, see `data.split_reflection`) of a pseudo-clip
is sampled through its own render -> native transform, offset from the eye's by an independent
trajectory: a gaze change slides the iris under a light that stays put.  It is masked by the eye's
ocular coverage, because a reflection cannot sit on a lid.  The target (clean + sheen) moves rigidly
with the eye, so for a pseudo-clip the exact crop -> crop correspondence of the eye's content is
known (`content_grid`): a temporal loss or metric can compare two frames' errors at the same point
of the eye rather than at the same crop pixel.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..gpu.audit import boundary
from ..gpu.imgops import CropTransform, _crop_grid, extract_crop, gaussian_blur
from .augment import CamParams, choose_exposure, sample_params, simulate_pair
from .data import GUARD_VS_LABEL, CacheStore, eye_geometry, pick_side, sample_geo, split_reflection

# --------------------------------------------------------------------------------- motion model ---
# Per-frame statistics in units of the iris diameter D (render px for the eye and the catchlight,
# native px for the tracker).  At the plate's 18.8 px native iris, 0.01 D is 0.19 native px per frame.
#   static  nothing moves; only the sensor noise and the tracker noise change.  The purest case of the
#           boil the user sees: a still eye whose fill re-draws itself.
#   slow    a talking head: smooth AR(1) drift of the eye in the camera and of the catchlight on the
#           eye, sub-degree roll, sub-percent zoom.
#   fast    faster drift plus saccades - jumps of the catchlight relative to the iris.
REGIMES = (("static", 0.30), ("slow", 0.45), ("fast", 0.25))
_REG = {
    "static": dict(v=0.0, key=0.0, rot_deg=0.0, zoom=0.0, sacc_p=0.0),
    "slow": dict(v=0.004, key=0.003, rot_deg=0.10, zoom=0.001, sacc_p=0.0),
    "fast": dict(v=0.012, key=0.008, rot_deg=0.25, zoom=0.002, sacc_p=0.12),
}
AR = 0.8               # velocity persistence of the drifts
EYE_MAX = 0.35         # |eye offset| cap over a sequence, in D
KEY_MAX = 0.25         # |catchlight offset| cap, in D: it stays on the iris region
SACC = (0.04, 0.15)    # saccade jump of the catchlight relative to the iris, in D
TRK_SIGMA = (0.004, 0.025)   # tracker centre noise, log-uniform per sequence, in native D


@dataclass
class Motion:
    regime: str
    eye: list          # (x, y) render px: where the eye sits relative to the camera's start
    rot: list          # radians: the eye's roll in the camera
    zoom: list         # scale of the eye in the camera
    key: list          # (x, y) render px: catchlight offset relative to the eye
    trk: list          # (x, y) native px: tracker centre noise, already smoothed over +-1 frame
    trk_sigma: float   # native px


def _walk(r: random.Random, T: int, sigma: float, cap: float) -> list:
    pos, vel, out = [0.0, 0.0], [0.0, 0.0], []
    for t in range(T):
        if t > 0 and sigma > 0:
            vel = [AR * vel[0] + r.gauss(0.0, sigma), AR * vel[1] + r.gauss(0.0, sigma)]
            pos = [pos[0] + vel[0], pos[1] + vel[1]]
            m = math.hypot(*pos)
            if m > cap:
                pos = [pos[0] * cap / m, pos[1] * cap / m]
                vel = [0.0, 0.0]
        out.append((pos[0], pos[1]))
    return out


def sample_motion(r: random.Random, T: int, D_r: float, D_n: float, regime: str | None = None) -> Motion:
    if regime is None:
        u, acc = r.random(), 0.0
        regime = REGIMES[-1][0]
        for name, p in REGIMES:
            acc += p
            if u < acc:
                regime = name
                break
    c = _REG[regime]
    eye = [(x * D_r, y * D_r) for x, y in _walk(r, T, c["v"], EYE_MAX)]
    key = _walk(r, T, c["key"], KEY_MAX)
    if c["sacc_p"] > 0:
        jump = [0.0, 0.0]
        for t in range(1, T):
            if r.random() < c["sacc_p"]:
                a, m = r.uniform(0, 2 * math.pi), r.uniform(*SACC)
                jump = [jump[0] + m * math.cos(a), jump[1] + m * math.sin(a)]
            k = [key[t][0] + jump[0], key[t][1] + jump[1]]
            n = math.hypot(*k)
            if n > KEY_MAX:
                k = [k[0] * KEY_MAX / n, k[1] * KEY_MAX / n]
            key[t] = (k[0], k[1])
    key = [(x * D_r, y * D_r) for x, y in key]
    rot = [0.0]
    zoom = [1.0]
    for _ in range(1, T):
        rot.append(rot[-1] + math.radians(r.gauss(0.0, c["rot_deg"])) if c["rot_deg"] > 0 else rot[-1])
        zoom.append(zoom[-1] * (1.0 + r.gauss(0.0, c["zoom"])) if c["zoom"] > 0 else zoom[-1])
    sig = math.exp(r.uniform(math.log(TRK_SIGMA[0]), math.log(TRK_SIGMA[1]))) * D_n
    raw = [(r.gauss(0.0, sig), r.gauss(0.0, sig)) for _ in range(T + 2)]
    trk = [tuple(sum(raw[t + j][i] for j in range(3)) / 3.0 for i in range(2)) for t in range(T)]
    return Motion(regime, eye, rot, zoom, key, trk, sig)


# ------------------------------------------------------------------------ transform algebra ---
def tf_fwd(tf: CropTransform, x, y):
    """frame -> crop coordinates (pixel centres), the mapping `CropTransform` documents."""
    c, s = math.cos(tf.theta), math.sin(tf.theta)
    dx, dy = x - tf.cx, y - tf.cy
    return (tf.scale * (c * dx + s * dy) + tf.size / 2.0 - 0.5,
            tf.scale * (-s * dx + c * dy) + tf.size / 2.0 - 0.5)


def tf_inv(tf: CropTransform, u, v):
    """crop -> frame coordinates; the exact inverse of `tf_fwd` (and of `imgops._crop_grid`)."""
    c, s = math.cos(tf.theta), math.sin(tf.theta)
    du, dv = (u - tf.size / 2.0 + 0.5) / tf.scale, (v - tf.size / 2.0 + 0.5) / tf.scale
    return tf.cx + c * du - s * dv, tf.cy + s * du + c * dv


def content_grid(a: dict, b: dict, S: int, dev) -> tuple[torch.Tensor, torch.Tensor]:
    """Where each crop-b pixel's EYE CONTENT sits in crop a, for two frames of one pseudo-clip.

    a, b: frame dicts with "tf1" (render -> native) and "tf2" (native -> crop).  The rendered eye is
    static, so crop_b -> native_b -> render -> native_a -> crop_a is exact.  Returns the grid_sample
    grid (1,S,S,2) that resamples a crop-a tensor into crop b, and the (1,S,S) mask of crop-b pixels
    whose preimage lies inside crop a."""
    v, u = torch.meshgrid(torch.arange(S, device=dev, dtype=torch.float32),
                          torch.arange(S, device=dev, dtype=torch.float32), indexing="ij")
    x, y = tf_inv(b["tf1"], *tf_inv(b["tf2"], u, v))
    ua, va = tf_fwd(a["tf2"], *tf_fwd(a["tf1"], x, y))
    grid = torch.stack([(ua + 0.5) / S * 2.0 - 1.0, (va + 0.5) / S * 2.0 - 1.0], -1)[None]
    inside = ((ua >= 0) & (ua <= S - 1) & (va >= 0) & (va <= S - 1)).float()[None]
    return grid, inside


# ----------------------------------------------------------------------------------- frames ---
def build_frame(sd, fi: int, side: str, geo, p: CamParams, r: random.Random, gen: torch.Generator,
                dev: torch.device, S: int, tf1: CropTransform, eye_xy: tuple[float, float],
                rot: float = 0.0, tf1_key: CropTransform | None = None, trk=(0.0, 0.0),
                unit: float | None = None, exposure: float | None = None, sim: dict | None = None,
                key_only: bool = True, still: "Still | None" = None) -> dict:
    """One frame of a sequence.  Mirrors `data.build_crop` (key_only split, camera simulation,
    frame-by-visible sizing, the same output keys) with the two transforms given explicitly.

    tf1      render -> native camera transform for this frame
    eye_xy   the eye centre in RENDER coordinates for this frame (the tracker's truth)
    rot      the eye's roll in the camera (radians); the tracker follows it
    tf1_key  pseudo-clips: the catchlight's own render -> native transform (None: the rendered one)
    trk      tracker centre noise in native px
    unit     the tracker's size unit (visible iris, native px), held per sequence; None on the first
             frame, which measures it the way `build_crop` does and returns it
    exposure None on the first frame (chosen as `build_crop` chooses it), then passed back in
    still    the frame on the GPU (`Still`); a pseudo-clip builds it ONCE for all its frames
    """
    fm = sd.frames_meta[fi]
    if still is None:
        still = Still(sd, fi, dev)
    N = tf1.size
    nat = still.sample(tf1)
    clean_n, refl_n, lab_n, valid_n = nat[:3], nat[3:6], nat[6:12], nat[12:13]
    negative = bool(sd.negative)
    if key_only and not negative:
        if exposure is None:
            exposure = choose_exposure(clean_n, refl_n, lab_n[0] > 0.5, r, p, valid=valid_n[0] > 0.999,
                                       sim=sim, labels=lab_n)
        sheen, key = split_reflection(refl_n, exposure, geo.native_iris, lab_n[0])
        if tf1_key is not None:
            # the catchlight sampled where the light puts it, split on its own (moved) ocular domain,
            # then kept only where the eye's cornea is now
            kn = still.sample(tf1_key, slice(3, 7))       # reflection + ocular label
            _, key = split_reflection(kn[:3], exposure, geo.native_iris, kn[3])
            key = key * lab_n[0:1].clamp(0, 1)
        clean_n = clean_n + sheen
        refl_n = key
        negative = not bool((refl_n > 0).any())
    elif tf1_key is not None and not negative:
        refl_n = still.sample(tf1_key, slice(3, 6)) * lab_n[0:1].clamp(0, 1)
    if exposure is None:
        exposure = choose_exposure(clean_n, refl_n, lab_n[0] > 0.5, r, p, valid=valid_n[0] > 0.999,
                                   sim=sim, labels=lab_n)
    p.exposure = exposure
    sp = simulate_pair(clean_n, refl_n, lab_n[0], p, gen, negative=negative)
    if unit is None:
        vis = float(((lab_n[3] + lab_n[4]).clamp(0, 1) * (valid_n[0] > 0.999).float()).sum())
        d_vis = 2.0 * math.sqrt(max(vis, 0.0) / math.pi) * GUARD_VS_LABEL
        unit = max(d_vis, 0.35 * geo.native_iris)
    C_n = unit * geo.k
    u0, v0 = tf_fwd(tf1, eye_xy[0], eye_xy[1])
    tf2 = CropTransform(u0 + geo.jx * C_n + trk[0], v0 + geo.jy * C_n + trk[1], S / C_n, geo.theta - rot, S)
    stk2 = torch.cat([sp["src"], sp["tgt"], lab_n, sp["clipped"][None].float(), sp["authorised"][None].float(), valid_n], 0)
    out = extract_crop(stk2, tf2)
    e_meta = fm["eyes"][side]
    valid = (out[14:15] > 0.999).float()
    return {"src": out[0:3], "tgt": out[3:6], "labels": out[6:12], "clipped": out[12:13] * valid,
            "auth": out[13:14] * valid, "valid": valid, "openness": e_meta["openness"],
            "native_iris": geo.native_iris, "exposure": exposure, "pupil_frac": e_meta["pupil_frac"],
            "negative": negative, "cam": p.as_dict(), "tf1": tf1, "tf2": tf2, "unit": unit, "N": N}


class Still:
    """One rendered frame on the GPU as the stack [clean 3, reflection 3, labels 6, ones 1].

    `extract_crop` antialiases a downsampling crop by blurring the frame region it reads (sigma
    0.45 / scale).  A pseudo-clip samples the SAME still at (nearly) the same scale for every frame
    and twice per frame (camera and catchlight), and that blur of a 1536 px render was most of a
    training step.  With `scale` given, the whole still is blurred once here and each sample is a
    plain bilinear read of it: the same result (the blur's support stays inside the region
    `extract_crop` would have padded), without repeating it.  scale=None samples through
    `extract_crop` unchanged (rendered clips: every frame is a different render)."""

    def __init__(self, sd, fi: int, dev: torch.device, scale: float | None = None):
        with boundary("training_cache_upload"):
            clean = sd.clean[fi].to(dev).float()
            refl = sd.refl[fi].to(dev).float()
            lab = sd.labels[fi].to(dev).float() / 255.0
        self.stack = torch.cat([clean, refl, lab, torch.ones_like(lab[:1])], 0)
        self.H, self.W = self.stack.shape[-2:]
        self.pre = scale is not None and scale < 0.75
        if self.pre:
            self.stack = gaussian_blur(self.stack, sigma=0.45 / scale)
            self._ones = torch.ones((1, 1, self.H, self.W), device=dev)

    def sample(self, tf: CropTransform, ch: slice = slice(None)) -> torch.Tensor:
        if not self.pre:
            return extract_crop(self.stack[ch], tf)
        grid = _crop_grid(tf, self.H, self.W, self.stack.device)
        valid = F.grid_sample(self._ones, grid, mode="nearest", padding_mode="zeros", align_corners=False)[0]
        return F.grid_sample(self.stack[ch][None], grid, mode="bilinear", padding_mode="border", align_corners=False)[0] * valid


def camera_size(D_r: float, s1: float, travel: float, zmax: float = 1.0) -> int:
    """Native camera window: `build_crop`'s 3.8 D plus room for the eye to travel inside it."""
    return int(math.ceil((D_r * 3.8 + 2.0 * travel) * s1 * zmax)) + 2


def item(fr: dict) -> dict:
    """A frame dict in the shape `train_tx.make_batch` consumes."""
    return dict(src=fr["src"], tgt=fr["tgt"], valid=fr["valid"], auth=fr["auth"], labels=fr["labels"],
                clipped=fr["clipped"], openness=fr["openness"], negative=float(fr["negative"]), kind="cg")


class PseudoClipSampler:
    """Pseudo-clips from the CG stills: one still per sequence, shared camera parameters and exposure,
    fresh noise per frame, the motion model above.  Checkpointable like `TrainSampler`."""

    def __init__(self, store: CacheStore, dev: torch.device, seed: int, S: int = 256, min_iris: float = 8.0,
                 sim: dict | None = None, neg_keep: float = 0.34, key_only: bool = True):
        self.store, self.dev, self.S, self.min_iris = store, dev, S, min_iris
        self.sim, self.neg_keep, self.key_only = sim, neg_keep, key_only
        self.p_sat = float(sim["forced_saturation_prob"]) if sim else 0.2
        self.r = random.Random(seed)
        self.gen = torch.Generator(device=dev)
        self.gen.manual_seed(seed)

    def state_dict(self) -> dict:
        return {"py_random": self.r.getstate(), "cuda_gen": self.gen.get_state()}

    def load_state_dict(self, st: dict) -> None:
        self.r.setstate(st["py_random"])
        self.gen.set_state(st["cuda_gen"].cpu())

    def _native_range(self, sd) -> tuple[float, float] | None:
        opt = (self.sim or {}).get("portrait_native_iris")
        if not opt or not sd.info["shot"].get("portrait"):
            return None
        if self.r.random() >= float(opt["prob"]):
            return None
        return float(opt["min_px"]), float(opt["max_px"])

    def clip(self, T: int, regime: str | None = None, native_range: tuple[float, float] | None = None,
             positive_only: bool = False) -> tuple[list[dict], Motion]:
        r = self.r
        for _ in range(50):
            si, fi = r.choice(self.store.frames)
            sd = self.store.shots[si]
            fm = sd.frames_meta[fi]
            side = pick_side(fm, sd.info["shot"], r)
            if side is None:
                continue
            cx, cy, D_r = eye_geometry(fm, side)
            geo = sample_geo(r, D_r, side, self.min_iris,
                             native_range=native_range if native_range is not None else self._native_range(sd))
            p = sample_params(r, force_saturate=r.random() < self.p_sat, sim=self.sim)
            mot = sample_motion(r, T, D_r, geo.native_iris, regime)
            travel = max(math.hypot(*e) for e in mot.eye) + max(math.hypot(*k) for k in mot.key)
            N = camera_size(D_r, geo.s1, travel, max(mot.zoom))
            frames, exposure, unit = [], None, None
            still = Still(sd, fi, self.dev, scale=geo.s1)
            for t in range(T):
                ex, ey = mot.eye[t]
                kx, ky = mot.key[t]
                tf1 = CropTransform(cx - ex, cy - ey, geo.s1 * mot.zoom[t], mot.rot[t], N)
                tf1k = CropTransform(cx - ex - kx, cy - ey - ky, geo.s1 * mot.zoom[t], mot.rot[t], N)
                fr = build_frame(sd, fi, side, geo, CamParams.from_dict(p.as_dict()), r, self.gen, self.dev,
                                 self.S, tf1, (cx, cy), rot=mot.rot[t], tf1_key=tf1k, trk=mot.trk[t],
                                 unit=None if unit is None else unit * mot.zoom[t] / mot.zoom[0],
                                 exposure=exposure, sim=self.sim, key_only=self.key_only, still=still)
                if unit is None:
                    unit = fr["unit"]
                exposure = fr["exposure"]
                fr["shot_id"], fr["frame_index"] = sd.shot_id, fi
                frames.append(fr)
            if frames[0]["negative"] and (positive_only or r.random() >= self.neg_keep):
                continue
            return frames, mot
        raise RuntimeError("could not sample a pseudo-clip")


# --------------------------------------------------------------------------- rendered clips ---
def rendered_clip(sd, native: float, r: random.Random, gen: torch.Generator, dev: torch.device, S: int,
                  sim: dict | None, key_only: bool = True) -> list[dict] | None:
    """A rendered clip (Rounds 29-31) through a FIXED camera: one render -> native transform for the
    whole clip, centred on the eye's mean position and large enough to hold its travel, so the eye
    moves across the native pixel grid.  The tracker is the oracle (true limbus centre per frame, no
    noise) with its size held for the clip at the median visible iris.  One camera parameter set,
    one exposure (frame 0's), fresh noise per frame.  None if the eye is smaller than `native`."""
    F_ = len(sd.frames_meta)
    ref = sd.frames_meta[F_ // 2]
    side = pick_side(ref, sd.info["shot"], r) or "L"
    cs = [eye_geometry(sd.frames_meta[f], side) for f in range(F_)]
    D_r = cs[F_ // 2][2]
    if D_r < native:
        return None
    geo = sample_geo(r, D_r, side, native_range=(native, native))
    p = sample_params(r, sim=sim)
    mx = sum(c[0] for c in cs) / F_
    my = sum(c[1] for c in cs) / F_
    travel = max(math.hypot(c[0] - mx, c[1] - my) for c in cs)
    tf1 = CropTransform(mx, my, geo.s1, 0.0, camera_size(D_r, geo.s1, travel))
    # the tracker's size: the median of the per-frame visible-iris units (the pipeline's 7-frame median)
    units = []
    for f in range(F_):
        lab = sd.labels[f].to(dev).float() / 255.0
        ln = extract_crop(torch.cat([lab[3:5], torch.ones_like(lab[:1])], 0), tf1)
        vis = float(((ln[0] + ln[1]).clamp(0, 1) * (ln[2] > 0.999).float()).sum())
        units.append(max(2.0 * math.sqrt(max(vis, 0.0) / math.pi) * GUARD_VS_LABEL, 0.35 * geo.native_iris))
    unit = sorted(units)[F_ // 2]
    frames, exposure = [], None
    for f in range(F_):
        fr = build_frame(sd, f, side, geo, CamParams.from_dict(p.as_dict()), r, gen, dev, S, tf1,
                         (cs[f][0], cs[f][1]), unit=unit, exposure=exposure, sim=sim, key_only=key_only)
        exposure = fr["exposure"]
        frames.append(fr)
    return frames

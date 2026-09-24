"""Training sample construction on the GPU from the float16 render cache.

Render once at high resolution, then create lower apparent eye resolution on CUDA: the eye
region is resampled to a sampled *native* iris diameter, degraded by the paired camera
simulation at that native pixel grid, and finally resampled into the model crop exactly as the
inference crop extractor does.  Labels follow the same geometry (no photometric changes).
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

from ..gpu.audit import boundary
from ..gpu.imgops import CropTransform, dilate, extract_crop, gaussian_blur
from ..models.inputs import normalize
from .augment import CamParams, choose_exposure, sample_params, simulate_pair

LABELS = ("ocular", "skin", "lash", "iris", "pupil", "tear")


_ST_DTYPES = {"F16": torch.float16, "F32": torch.float32, "U8": torch.uint8}


class CacheFile:
    """Read-only memory map of one safetensors cache file (header parsed here, no copy).

    The render cache (~11 GB for the prototype train split) must stay in the OS file cache:
    loading it (or safetensors' copy-on-write maps) is charged to the Windows commit limit
    and exhausted this host's headroom during training.  ACCESS_READ maps are not."""

    def __init__(self, path: Path):
        import mmap
        import struct

        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            self.header = json.loads(f.read(n))
            self._mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        self.data_start = 8 + n

    def frames(self, key: str) -> "LazyFrames":
        h = self.header[key]
        return LazyFrames(self, _ST_DTYPES[h["dtype"]], torch.Size(h["shape"]), self.data_start + h["data_offsets"][0])


class LazyFrames:
    """(F,C,H,W) tensor in a CacheFile; `[i]` is a zero-copy CPU view of frame i (read-only:
    only ever used as the source of a host-to-device copy)."""

    def __init__(self, cf: CacheFile, dtype: torch.dtype, shape: torch.Size, offset: int):
        self._cf, self.dtype, self.shape, self._off = cf, dtype, shape, offset
        self._n = math.prod(shape[1:])

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, i: int) -> torch.Tensor:
        import warnings

        if not 0 <= i < self.shape[0]:
            raise IndexError(i)
        isz = torch.tensor([], dtype=self.dtype).element_size()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # read-only buffer: never written
            t = torch.frombuffer(self._cf._mm, dtype=self.dtype, count=self._n, offset=self._off + i * self._n * isz)
        return t.view(self.shape[1:])


@dataclass
class ShotData:
    shot_id: str
    split: str
    kind: str
    negative: bool
    frames_meta: list
    clean: LazyFrames  # (F,3,H,W) fp16, host file cache
    refl: LazyFrames
    labels: LazyFrames  # (F,6,H,W) uint8
    info: dict = field(default_factory=dict)


class CacheStore:
    def __init__(self, run_dir: Path, split: str, kinds: tuple[str, ...] = ("still", "clip")):
        man = json.loads((run_dir / "manifests" / "dataset_manifest.json").read_text(encoding="utf-8"))
        self.shots: list[ShotData] = []
        cache = run_dir / "data" / "cache"
        for s in man["shots"]:
            if s["split"] != split or s["kind"] not in kinds:
                continue
            info = json.loads((cache / split / f"{s['shot_id']}.json").read_text(encoding="utf-8"))
            cf = CacheFile(cache / s["cache"])
            self.shots.append(ShotData(s["shot_id"], split, s["kind"], bool(s["negative"]), info["frames"],
                                       cf.frames("clean"), cf.frames("reflection"), cf.frames("labels"),
                                       {"manifest": s, "shot": info["shot"]}))
        if not self.shots:
            raise RuntimeError(f"no cached shots for split {split} kinds {kinds} under {cache}")
        self.frames = [(i, f) for i, sd in enumerate(self.shots) for f in range(len(sd.frames_meta))]
        self.clips = [i for i, sd in enumerate(self.shots) if sd.kind == "clip"]


def eye_geometry(fm: dict, side: str) -> tuple[float, float, float]:
    e = fm["eyes"][side]
    lx = [p[0] for p in e["limbus_px"]]
    ly = [p[1] for p in e["limbus_px"]]
    cx = sum(lx) / len(lx) - 0.5  # continuous -> pixel-centre index coordinates
    cy = sum(ly) / len(ly) - 0.5
    return cx, cy, max(4.0, e["iris_diameter_px"])


def pick_side(fm: dict, shot: dict, r: random.Random) -> str | None:
    cands = [s for s in ("L", "R") if fm["eyes"][s]["in_frame"] and fm["eyes"][s]["facing_cos"] > 0.15]
    tgt = shot.get("camera", {}).get("target", "")
    if tgt in ("eye_L", "eye_R") and tgt[-1] in cands:
        return tgt[-1]
    return r.choice(cands) if cands else None


@dataclass
class Geo:
    side: str
    s1: float  # render -> native scale
    k: float  # crop size in iris diameters
    theta: float
    jx: float
    jy: float
    native_iris: float


def sample_geo(r: random.Random, D_r: float, side: str, min_iris: float = 8.0, max_native: float = 120.0,
               native_range: tuple[float, float] | None = None) -> Geo:
    """native_range (Round 2, optional): draw the native iris diameter log-uniformly inside this range
    instead of the default 8 px..rendered-size law.  Never upsamples: the range is clipped to the
    rendered size, and a rendered eye smaller than the range falls back to the default law.  With
    ``None`` the random stream is consumed exactly as before."""
    hi = min(D_r, max_native)
    if native_range is not None and D_r >= native_range[0]:
        lo_n, hi_n = max(min_iris, native_range[0]), min(native_range[1], D_r)
        D_n = math.exp(r.uniform(math.log(lo_n), math.log(max(lo_n * 1.01, hi_n))))
    elif r.random() < 0.3 and D_r <= max_native:
        D_n = D_r
    else:
        D_n = math.exp(r.uniform(math.log(min_iris), math.log(max(min_iris * 1.01, hi))))
    return Geo(side, D_n / D_r, r.uniform(2.3, 3.2), math.radians(r.uniform(-12, 12)), r.uniform(-0.12, 0.12), r.uniform(-0.12, 0.12), D_n)


def shift_zero(x: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
    """Translate (C,H,W) by integer pixels; vacated area is zero (no wrap-around)."""
    C, H, W = x.shape
    out = torch.zeros_like(x)
    ys, yd = (slice(0, H - dy), slice(dy, H)) if dy >= 0 else (slice(-dy, H), slice(0, H + dy))
    xs, xd = (slice(0, W - dx), slice(dx, W)) if dx >= 0 else (slice(-dx, W), slice(0, W + dx))
    out[:, yd, xd] = x[:, ys, xs]
    return out


# The label-derived visible diameter and the guard-derived one are not the same number: the guard
# thresholds its probability at 0.5, which erodes the anti-aliased limbus edge the coverage label
# keeps.  Measured by running `iris_geometry` (the estimator inference actually frames with) over
# 151 open CG crops sized from the labels: the crop came out 3.06 visible diameters wide where the
# plate's own recorded transforms give 2.75, a systematic 1.11x.  Correcting here makes the network
# see the eye at the size it is deployed at.  Not circular: all three numbers come from that one
# estimator, so only the sizing input differs.
GUARD_VS_LABEL = 2.75 / 3.06


KEY_T_HI = 0.05     # exposed scene-linear luminance (white = 1.0) the key must reach, or the shot is a negative
KEY_SHEEN_CAP = 0.03   # the most a broad reflection may leave behind as sheen
KEY_DEAD = 0.004    # key below this is folded back into the sheen (no confetti support)
KEY_OPEN_FRAC = 0.2    # opening radius, in native iris diameters: structures narrower than 0.4 D are catchlight


def split_reflection(refl: torch.Tensor, exposure: float, native_iris: float,
                     ocular: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(3,H,W) reflection layer, (H,W) ocular coverage -> (sheen, key), sheen + key == refl, both >= 0.

    The renderer writes ONE reflection layer - the whole ocular F*glossy closure (blender/materials.py
    mode 1) - so the broad environment SHEEN on the cornea and the KEY-LIGHT catchlight arrive
    together, and a target of `clean` removes both.  Measured in Round 22's review: 35% of CG
    positives had no catchlight at all, the support covered a median 43% of the iris+pupil, and the
    target darkened a median 22% of the eye away from the catchlight - while the user's R39 changes
    nothing outside the blob (-0.0006).  The model learned veil removal, and on real footage, which
    always carries a lifted floor, that is the whole-eye darkening.

    The two do not separate by LEVEL (the exposed luminance histogram is a continuum), and a
    first attempt that grew each catchlight down to a low threshold swallowed the sheen connected
    to it (support still 31% of the eye).  They separate by SIZE: a catchlight is compact, the
    sheen is broad.  A grey-scale opening at 0.2 iris diameters keeps only what is broader than
    0.4 D - skirt included, so no ring of catchlight is left in the target - and that, capped at
    `KEY_SHEEN_CAP` so a broad BRIGHT reflection (a window) still counts as something to remove,
    is the sheen.  The colour of the layer is kept: the split is a per-pixel ratio.

    Two details that a second attempt got wrong and the audit caught (support went UP, to 36%):
      * the opening must run INSIDE the ocular surface only.  The layer is zero on lids and lashes,
        so an unmasked erosion ate the sheen away for 0.2 D along every lid edge and behind every
        lash, and all of that came out as "catchlight";
      * the tear film sparkles.  Compact specks far below catchlight level survive the opening's
        logic as key, so a component is kept only if it actually reaches `KEY_T_HI` somewhere.
    """
    w = torch.tensor([0.2126, 0.7152, 0.0722], device=refl.device, dtype=refl.dtype).view(3, 1, 1)
    L = ((refl * w).sum(0) * float(exposure))[None, None]
    dom = (ocular > 0.95)[None, None]
    rho = max(1, int(round(KEY_OPEN_FRAC * native_iris)))
    k = 2 * rho + 1
    big = 1e4
    eroded = -F.max_pool2d(-torch.where(dom, L, torch.full_like(L, big)), k, stride=1, padding=rho)
    opened = F.max_pool2d(torch.where(dom & (eroded < big / 2), eroded, torch.zeros_like(L)), k, stride=1, padding=rho)
    df = dom.float()
    smooth = gaussian_blur((opened * df)[0], rho / 2.0)[None] / gaussian_blur(df[0], rho / 2.0)[None].clamp(min=1e-3)
    Ls = torch.where(dom, torch.minimum(smooth, L).clamp(max=KEY_SHEEN_CAP), L)
    key_L = (L - Ls - KEY_DEAD).clamp(min=0.0)
    seed, body = key_L >= KEY_T_HI, key_L > 0
    if not bool(seed.any()):
        return refl, torch.zeros_like(refl)
    for _ in range(max(refl.shape[-2:])):
        new = (F.max_pool2d(seed.float(), 3, stride=1, padding=1) > 0.5) & body
        if bool((new == seed).all()):
            break
        seed = new
    key_L = key_L * seed.float()
    ratio = ((L - key_L) / L.clamp(min=1e-8))[0]
    sheen = refl * ratio
    return sheen, refl - sheen


def build_crop(sd: ShotData, fi: int, geo: Geo, p: CamParams, r: random.Random, gen: torch.Generator, dev: torch.device,
               S: int, exposure: float | None = None, center_override: tuple[float, float] | None = None,
               jitter_px: float = 0.0, sim: dict | None = None, return_native: bool = False,
               k_clamp: bool = True, frame_by_visible: bool = False, key_only: bool = False) -> dict:
    """return_native adds the native-resolution simulated pair, labels and the native->crop
    transform (used to reproduce the full-resolution inference gate on validation data)."""
    fm = sd.frames_meta[fi]
    with boundary("training_cache_upload"):
        clean = sd.clean[fi].to(dev).float()
        refl = sd.refl[fi].to(dev).float()
        lab = sd.labels[fi].to(dev).float() / 255.0
    cx, cy, D_r = eye_geometry(fm, geo.side)
    if center_override is not None:
        cx, cy = center_override
    H, W = clean.shape[-2:]
    # prefer crops that fit inside the rendered frame (worst-case rotation and jitter)
    d_edge = max(1.0, min(cx, W - 1 - cx, cy, H - 1 - cy))
    k_fit = 2.0 * d_edge / (D_r * 1.42 * (1 + 2 * max(abs(geo.jx), abs(geo.jy))))
    if k_clamp:
        geo.k = max(2.0, min(geo.k, k_fit))
    # k_clamp=False (Round 21): the new CG bank is framed close - median rendered iris 534 px in a
    # 1536 px frame - so k_fit is under 2.3 for 87% of it and the clamp pinned 76% of crops at the
    # 2.0 floor, training at 2.0 iris diameters against inference's K_CROP = 2.75.  The crop may
    # instead extend past the frame: `valid` already marks those pixels and every loss term is
    # weighted by it, so they are excluded rather than fabricated.
    N = int(math.ceil(D_r * 3.8 * geo.s1)) + 2
    tf1 = CropTransform(cx, cy, geo.s1, 0.0, N)
    stack = torch.cat([clean, refl, lab, torch.ones_like(lab[:1])], 0)
    nat = extract_crop(stack, tf1)
    clean_n, refl_n, lab_n, valid_n = nat[:3], nat[3:6], nat[6:12], nat[12:13]
    negative = bool(sd.negative)
    if key_only and not negative:
        # Round 22: the target keeps the corneal sheen and loses only the catchlight (see
        # `split_reflection`).  The split needs the exposure, so it is chosen first - before the
        # reflection shift rather than after it, which also means only the KEY moves when the light
        # is re-aimed and the sheen stays where the environment put it.  A shot whose reflection
        # never reaches catchlight level becomes a NEGATIVE: sheen in, the same sheen out.
        if exposure is None:
            exposure = choose_exposure(clean_n, refl_n, lab_n[0] > 0.5, r, p, valid=valid_n[0] > 0.999, sim=sim, labels=lab_n)
        sheen, refl_n = split_reflection(refl_n, exposure, geo.native_iris, lab_n[0])
        clean_n = clean_n + sheen
        negative = not bool((refl_n > 0).any())
    shift = (sim or {}).get("reflection_shift") if not negative else None
    if shift and center_override is None and r.random() < float(shift["prob"]):
        # Round 4 (config camera_sim.reflection_shift): move the reflection layer within the eye - a
        # light from another direction reflects elsewhere on the cornea.  Integer native pixels,
        # zero fill (no wrap-around); target/labels follow automatically because they are derived
        # from the simulated pair.  Absent option = identical random stream.
        d_max = float(shift["max_frac"]) * geo.native_iris
        ang = r.uniform(0.0, 2.0 * math.pi)
        d = r.uniform(0.0, d_max)
        dx, dy = int(round(d * math.cos(ang))), int(round(d * math.sin(ang)))
        if dx or dy:
            refl_n = shift_zero(refl_n, dx, dy)
    if exposure is None:
        exposure = choose_exposure(clean_n, refl_n, lab_n[0] > 0.5, r, p, valid=valid_n[0] > 0.999, sim=sim, labels=lab_n)
    p.exposure = exposure
    sp = simulate_pair(clean_n, refl_n, lab_n[0], p, gen, negative=negative)
    unit = geo.native_iris
    if frame_by_visible:
        # Inference sizes the crop by the VISIBLE iris+pupil disk: pipeline._refine feeds
        # models.inputs.iris_geometry, whose radius is sqrt(area / pi) of the guard's iris+pupil
        # probability - so the lid-occluded part is NOT counted.  `geo.native_iris` is the opposite
        # quantity, the major axis of the fully projected limbus (blender/generate.py: max of the
        # limbus bbox), which stays the same size as the lid closes over it.  Sizing training by
        # the second and inference by the first framed the training eye 1.59x smaller than the
        # deployed one, measured against the plate's own recorded crop transforms.
        vis = float(((lab_n[3] + lab_n[4]).clamp(0, 1) * (valid_n[0] > 0.999).float()).sum())
        d_vis = 2.0 * math.sqrt(max(vis, 0.0) / math.pi) * GUARD_VS_LABEL
        
        # A nearly-shut eye has a sliver of visible iris and would zoom without bound; inference
        # never reaches such an eye (it fails the presence gate), so fall back to the limbus there.
        unit = max(d_vis, 0.35 * geo.native_iris)
    C_n = unit * geo.k
    c = (N / 2 - 0.5) + geo.jx * C_n + r.gauss(0, jitter_px), (N / 2 - 0.5) + geo.jy * C_n + r.gauss(0, jitter_px)
    tf2 = CropTransform(c[0], c[1], S / C_n, geo.theta, S)
    stk2 = torch.cat([sp["src"], sp["tgt"], lab_n, sp["clipped"][None].float(), sp["authorised"][None].float(), valid_n], 0)
    out = extract_crop(stk2, tf2)
    e_meta = fm["eyes"][geo.side]
    valid = (out[14:15] > 0.999).float()
    res = {"src": out[0:3], "tgt": out[3:6], "labels": out[6:12], "clipped": out[12:13] * valid, "auth": out[13:14] * valid,
           "valid": valid,
           "openness": e_meta["openness"], "native_iris": geo.native_iris, "exposure": exposure,
           "pupil_frac": e_meta["pupil_frac"], "negative": negative, "cam": p.as_dict()}
    if return_native:
        res["native"] = {"src": sp["src"], "tgt": sp["tgt"], "labels": lab_n, "valid": valid_n[0] > 0.999, "tf": tf2, "N": N,
                         "clipped": sp["clipped"]}
    return res


class TrainSampler:
    def __init__(self, store: CacheStore, dev: torch.device, seed: int, S: int = 256, min_iris: float = 8.0,
                 sim: dict | None = None, k_clamp: bool = True, frame_by_visible: bool = False,
                 key_only: bool = False):
        """sim: config data.camera_sim (v2 camera simulation) or None (original prototype behaviour)."""
        self.store, self.dev, self.S = store, dev, S
        self.k_clamp = k_clamp
        self.frame_by_visible = frame_by_visible
        self.key_only = key_only
        self.sim = sim or None
        self.p_sat = float(self.sim["forced_saturation_prob"]) if self.sim else 0.2
        self.r = random.Random(seed)
        self.gen = torch.Generator(device=dev)
        self.gen.manual_seed(seed)
        self.min_iris = min_iris

    def state_dict(self) -> dict:
        return {"py_random": self.r.getstate(), "cuda_gen": self.gen.get_state()}

    def load_state_dict(self, st: dict) -> None:
        self.r.setstate(st["py_random"])
        self.gen.set_state(st["cuda_gen"].cpu())  # generator state must be a CPU ByteTensor

    def _one(self, return_native: bool = False) -> dict:
        for _ in range(20):
            si, fi = self.r.choice(self.store.frames)
            sd = self.store.shots[si]
            fm = sd.frames_meta[fi]
            side = pick_side(fm, sd.info["shot"], self.r)
            if side is None:
                continue
            _, _, D_r = eye_geometry(fm, side)
            geo = sample_geo(self.r, D_r, side, self.min_iris, native_range=self.native_range(sd))
            p = sample_params(self.r, force_saturate=self.r.random() < self.p_sat, sim=self.sim)
            return build_crop(sd, fi, geo, p, self.r, self.gen, self.dev, self.S, sim=self.sim,
                              return_native=return_native, k_clamp=self.k_clamp,
                              frame_by_visible=self.frame_by_visible, key_only=self.key_only)
        raise RuntimeError("could not sample a valid eye crop")

    def native_range(self, sd: ShotData) -> tuple[float, float] | None:
        """Round 2 (config camera_sim.portrait_native_iris): portrait shots are drawn at deployment
        native iris sizes with probability ``prob``.  Absent option = no extra random draws."""
        opt = (self.sim or {}).get("portrait_native_iris")
        if not opt or not sd.info["shot"].get("portrait"):
            return None
        if self.r.random() >= float(opt["prob"]):
            return None
        return float(opt["min_px"]), float(opt["max_px"])

    def sample_native(self) -> dict:
        """One eye crop plus its native-resolution pair/labels/transform (validation analyses)."""
        return self._one(return_native=True)

    def negative_crop(self) -> dict:
        """Crop away from both eyes (EyeGuard 'no eye' examples)."""
        for _ in range(30):
            si, fi = self.r.choice(self.store.frames)
            sd = self.store.shots[si]
            fm = sd.frames_meta[fi]
            H, W = sd.clean.shape[-2:]
            side = self.r.choice(["L", "R"])
            _, _, D_r = eye_geometry(fm, side)
            cx, cy = self.r.uniform(0, W - 1), self.r.uniform(0, H - 1)
            far = all(math.hypot(cx - eye_geometry(fm, s)[0], cy - eye_geometry(fm, s)[1]) > 1.6 * D_r for s in ("L", "R"))
            if not far:
                continue
            geo = sample_geo(self.r, D_r, side, self.min_iris)
            geo.jx = geo.jy = 0.0
            p = sample_params(self.r, sim=self.sim)
            out = build_crop(sd, fi, geo, p, self.r, self.gen, self.dev, self.S, center_override=(cx, cy), sim=self.sim)
            out["force_absent"] = True
            return out
        return self._one()

    def batch(self, n: int, negative_frac: float = 0.0) -> dict:
        items = [self.negative_crop() if self.r.random() < negative_frac else self._one() for _ in range(n)]
        return collate(items, self.dev)

    def clip_window(self, T: int = 6) -> dict:
        """T consecutive frames (stabilised per frame on the eye), shared camera params,
        per-frame noise.  Frames beyond the clip are marked invalid (replicated content)."""
        ci = self.r.choice(self.store.clips)
        sd = self.store.shots[ci]
        F_ = len(sd.frames_meta)
        start = self.r.randint(-2, F_ - T + 2)
        idx = [min(F_ - 1, max(0, start + k)) for k in range(T)]
        inside = [0 <= start + k < F_ for k in range(T)]
        ref = sd.frames_meta[idx[T // 2]]
        side = pick_side(ref, sd.info["shot"], self.r) or "L"
        _, _, D_r = eye_geometry(ref, side)
        geo = sample_geo(self.r, D_r, side, self.min_iris)
        p = sample_params(self.r, force_saturate=self.r.random() < self.p_sat, sim=self.sim)
        exposure = None
        items = []
        for k, fi in enumerate(idx):
            it = build_crop(sd, fi, geo, CamParams.from_dict(p.as_dict()), self.r, self.gen, self.dev, self.S,
                            exposure=exposure, jitter_px=0.6, sim=self.sim)
            exposure = it["exposure"]
            items.append(it)
        b = collate(items, self.dev)
        b["inside"] = torch.tensor(inside, device=self.dev)
        return b


def collate(items: list[dict], dev: torch.device) -> dict:
    out = {k: torch.stack([it[k] for it in items]) for k in ("src", "tgt", "labels", "clipped", "auth", "valid")}
    out["openness"] = torch.tensor([it["openness"] for it in items], device=dev)
    out["native_iris"] = torch.tensor([it["native_iris"] for it in items], device=dev)
    out["negative"] = torch.tensor([it["negative"] for it in items], device=dev)
    out["force_absent"] = torch.tensor([it.get("force_absent", False) for it in items], device=dev)
    out["meta"] = [{"cam": it["cam"], "pupil_frac": it["pupil_frac"]} for it in items]
    return out


def targets_from_batch(b: dict, e: torch.Tensor, vis: float | None = None) -> dict:
    """Ground-truth maps in the model's normalised space (computed after the camera sim).

    vis (v2, config model.reflection_visibility): the guard's reflection channel marks only *visible*
    reflection (normalised change > vis); fainter changed pixels get zero loss weight ("don't care"),
    and a per-eye presence label is 1 with visible reflection, 0 on reflection-free eyes / no-eye
    crops, don't-care for faint-only eyes.  The restorer's support/synthesis targets are unchanged."""
    lab = b["labels"]
    ocular, skin, lash, iris, pupil, tear = (lab[:, i:i + 1] for i in range(6))
    src_n = normalize(b["src"], e)
    tgt_n = normalize(b["tgt"], e)
    diff = (tgt_n - src_n).abs().amax(1, keepdim=True)
    refl = (diff > 0.015).float() * (ocular > 0.2).float()
    support = dilate(refl[:, 0] > 0.5, 1)[:, None].float() * (b["auth"] > 0.5).float()
    synth = (b["clipped"] > 0.5).float() * support
    protected = torch.clamp(skin + lash + tear, 0, 1) * (1 - ocular).clamp(0, 1) + torch.clamp(lash + tear, 0, 1) * ocular
    present = ((b["openness"] > 0.15) & (~b["force_absent"]) & ((iris + pupil).sum((1, 2, 3)) > 30)).float()
    out = {"src_n": src_n, "tgt_n": tgt_n, "reflection": refl, "support": support, "synth": synth,
           "masks": torch.cat([ocular, iris, pupil, protected.clamp(0, 1), refl, b["clipped"]], 1), "present": present}
    if vis is not None:
        eye = (ocular > 0.2).float()
        refl_vis = (diff > vis).float() * eye
        faint = ((diff > 0.015) & ~(diff > vis)).float() * eye
        mw = torch.ones_like(out["masks"])
        mw[:, 4:5] = 1.0 - faint
        out["masks"] = torch.cat([ocular, iris, pupil, protected.clamp(0, 1), refl_vis, b["clipped"]], 1)
        out["mask_weight"] = mw
        n_vis = (refl_vis * b["valid"]).sum((1, 2, 3))
        neg = b["negative"].bool() | b["force_absent"].bool()
        out["refl_present"] = ((n_vis >= 6) & ~neg).float()
        out["refl_present_w"] = ((n_vis >= 6) | neg).float()
    return out

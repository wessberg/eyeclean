"""Convert Blender multipart EXR renders into validated float16 training caches (GPU).

Per shot: ``<cache>/<shot_id>.safetensors`` with
  clean       (F, 3, H, W) float16  scene-linear Rec.709 (renderer space), target radiance
  reflection  (F, 3, H, W) float16  designated camera-visible ocular reflection F*glossy
  labels      (F, 6, H, W) uint8    coverage x255: ocular, skin, lash/brow, iris, pupil, tear
and ``<shot_id>.json`` with the shot spec and per-frame generator metadata.  The source image is
assembled at training time as clean + reflection (or clean alone for negatives).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from ..gpu.audit import boundary
from ..gpu.imgops import dilate
from ..io.exr import ExrError, read_render_exr

LABEL_NAMES = ("ocular", "skin", "lash", "iris", "pupil", "tear")
FP16_MAX = 65000.0


def _rgb(d: dict, layer: str) -> torch.Tensor:
    return torch.stack([d[f"{layer}.Combined.{c}"] for c in "RGB"])


def convert_frame(exr_path: Path, dev: torch.device) -> tuple[dict, dict]:
    d = read_render_exr(exr_path, dev)
    clean, refl = _rgb(d, "clean"), _rgb(d, "reflection")
    la, lb = _rgb(d, "lab_a"), _rgb(d, "lab_b")
    stats = {}
    finite = bool(torch.isfinite(clean).all() & torch.isfinite(refl).all())
    if not finite:
        raise ExrError(f"{exr_path}: non-finite radiance")
    stats["refl_negative_px"] = int((refl < -1e-6).any(0).sum())
    away = ~dilate(la[0] > 0, 2)
    stats["refl_offeye_nonzero_px"] = int(((refl.abs().sum(0) > 0) & away).sum())
    stats["fp16_clamped_px"] = int(((clean > FP16_MAX) | (refl > FP16_MAX)).any(0).sum())
    stats["ocular_px"] = int((la[0] > 0.5).sum())
    stats["iris_px"] = int((lb[0] > 0.5).sum())
    stats["pupil_px"] = int((lb[1] > 0.5).sum())
    oc = la[0] > 0.5
    stats["refl_max"] = float(refl.max())
    stats["refl_mean_ocular"] = float(refl.mean(0)[oc].mean()) if stats["ocular_px"] else 0.0
    stats["clean_median_ocular"] = float(clean.mean(0)[oc].median()) if stats["ocular_px"] else 0.0
    labels = torch.cat([la, lb], 0).clamp(0, 1).mul(255).round().to(torch.uint8)
    return {"clean": clean.clamp(-FP16_MAX, FP16_MAX).half(), "reflection": refl.clamp(0, FP16_MAX).half(), "labels": labels}, stats


def convert_shot(raw_dir: Path, cache_dir: Path, shot: dict, dev: torch.device, keep_raw: bool = False) -> dict:
    meta = json.loads((raw_dir / "meta.json").read_text(encoding="utf-8"))
    frames = meta["frames"]
    tens = {"clean": [], "reflection": [], "labels": []}
    fstats = []
    for fm in frames:
        t, st = convert_frame(raw_dir / fm["exr"], dev)
        for k in tens:
            tens[k].append(t[k])
        fstats.append(st)
    out = {k: torch.stack(v) for k, v in tens.items()}
    cache_dir.mkdir(parents=True, exist_ok=True)
    tp = cache_dir / f"{shot['shot_id']}.safetensors"
    tmp = tp.with_suffix(".part")
    with boundary("training_cache_upload"):
        host = {k: v.contiguous().cpu() for k, v in out.items()}
    save_file(host, str(tmp), metadata={"schema": "eyeclean.cache/1", "shot_id": shot["shot_id"]})
    with open(tmp, "rb+") as f:
        os.fsync(f.fileno())
    os.replace(tmp, tp)
    sha = hashlib.sha256(tp.read_bytes()).hexdigest()
    info = {"shot": shot, "render_meta": {k: v for k, v in meta.items() if k != "frames"}, "frames": frames, "frame_stats": fstats,
            "tensors": {k: list(v.shape) for k, v in out.items()}, "cache_file": tp.name, "sha256": sha,
            "label_names": LABEL_NAMES, "radiance_space": "scene-linear Rec.709 (Blender default)"}
    jp = cache_dir / f"{shot['shot_id']}.json"
    jp.write_text(json.dumps(info), encoding="utf-8")
    if not keep_raw:
        for fm in frames:
            try:
                (raw_dir / fm["exr"]).unlink()
            except FileNotFoundError:
                pass
    return info


def cache_valid(cache_dir: Path, shot_id: str) -> bool:
    tp = cache_dir / f"{shot_id}.safetensors"
    jp = cache_dir / f"{shot_id}.json"
    if not (tp.exists() and jp.exists()):
        return False
    try:
        info = json.loads(jp.read_text(encoding="utf-8"))
        return info.get("sha256") == hashlib.sha256(tp.read_bytes()).hexdigest()
    except Exception:
        return False


def load_shot(cache_dir: Path, shot_id: str) -> tuple[dict, dict]:
    """CPU tensors (declared cache boundary) + metadata."""
    info = json.loads((cache_dir / f"{shot_id}.json").read_text(encoding="utf-8"))
    return load_file(str(cache_dir / f"{shot_id}.safetensors")), info

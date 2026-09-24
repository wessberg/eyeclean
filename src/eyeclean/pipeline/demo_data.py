"""Build held-out demo inputs (visible to prediction) and targets (evaluation-only) from the cache.

For each demo case the rendered clean/reflection radiance goes through ONE fixed paired camera
simulation (exposure from the face, mild sensor noise, clipping at the white level, linear
encoding) and is written as an uncompressed HALF EXR sequence in scene-linear Rec.709 primaries
(declared to `eyeclean process` as ``--input-color linear-rec709``).  Targets, labels and
generator metadata go to ``<run>/eval_private/<case>/`` which the prediction subprocess is
forbidden to read (enforced by an audit hook and logged).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch

from ..data.sampling import stable_seed
from ..io.exr import PIXEL_HALF, write_exr
from ..training.augment import CamParams, simulate_pair


def demo_cases(run_dir: Path) -> dict[str, list[str]]:
    man = json.loads((run_dir / "manifests" / "dataset_manifest.json").read_text(encoding="utf-8"))
    cases: dict[str, list[str]] = {}
    for s in man["shots"]:
        if s["kind"] == "demo":
            base = s["shot_id"][:-2] if s["shot_id"].endswith(("_a", "_b")) else s["shot_id"]
            cases.setdefault(base, []).append(s["shot_id"])
    return {k: sorted(v) for k, v in sorted(cases.items())}


def _cam_linear(case: str, seed: int) -> CamParams:
    r = random.Random(seed)
    return CamParams(defocus_sigma=0.0, motion_len=0.0, exposure=1.0, wb=(1.0, 1.0, 1.0), shot_a=r.uniform(2e-4, 8e-4),
                     read_b=1e-6, white=1.0, encoding="linear", bits=32, chroma="444", sharpen=0.0,
                     saturate_forced="saturated" in case)


def build_case(run_dir: Path, case: str, shot_ids: list[str], dev: torch.device, seed: int) -> dict:
    from safetensors.torch import load_file

    ex_dir = run_dir / "examples" / case
    ev_dir = run_dir / "eval_private" / case
    inp_dir, tgt_dir = ex_dir / "input_exr", ev_dir / "target_exr"
    for d in (inp_dir, tgt_dir, ev_dir / "labels"):
        d.mkdir(parents=True, exist_ok=True)
    cam = _cam_linear(case, stable_seed(seed, "democam", case))
    gen = torch.Generator(device=dev)
    gen.manual_seed(stable_seed(seed, "demonoise", case))
    frames_meta = []
    k = 0
    exposure = None
    cache = run_dir / "data" / "cache" / "test"
    for part, sid in enumerate(shot_ids):
        info = json.loads((cache / f"{sid}.json").read_text(encoding="utf-8"))
        t = load_file(str(cache / f"{sid}.safetensors"))
        negative = bool(info["shot"].get("negative"))
        for fi, fm in enumerate(info["frames"]):
            clean = t["clean"][fi].to(dev).float()
            refl = t["reflection"][fi].to(dev).float()
            lab = t["labels"][fi].to(dev).float() / 255.0
            if exposure is None or (part > 0 and fi == 0):
                skin = lab[1] > 0.5
                lum = 0.2126 * clean[0] + 0.7152 * clean[1] + 0.0722 * clean[2]
                med = float(lum[skin].median()) if bool(skin.sum() > 100) else float(lum.median())
                exposure = 0.22 / max(med, 1e-5)
                if cam.saturate_forced:
                    oc = lab[0] > 0.5
                    if bool(oc.any()):
                        mx = float((refl + clean).amax(0)[oc].max()) * exposure
                        if mx < 1.5:
                            exposure *= 1.5 / max(mx, 1e-6)
            cam.exposure = exposure
            sim = simulate_pair(clean, refl, lab[0], cam, gen, negative=negative, working_color="linear-rec709")
            src, tgt = sim["src"], sim["tgt"]
            write_exr(inp_dir / f"frame_{k:06d}.exr", {"R": src[0], "G": src[1], "B": src[2]}, pixel_type=PIXEL_HALF)
            write_exr(tgt_dir / f"frame_{k:06d}.exr", {"R": tgt[0], "G": tgt[1], "B": tgt[2]}, pixel_type=PIXEL_HALF)
            write_exr(ev_dir / "labels" / f"frame_{k:06d}.exr",
                      {n: lab[i] for i, n in enumerate(("ocular", "skin", "lash", "iris", "pupil", "tear"))} |
                      {"clipped": sim["clipped"].float(), "authorised": sim["authorised"].float()}, pixel_type=PIXEL_HALF)
            frames_meta.append({"index": k, "shot_id": sid, "cut_part": part, "time_s": k / 24.0, "negative": negative,
                                "eyes": fm["eyes"], "head_deg": fm.get("head_deg")})
            k += 1
    meta = {"case": case, "shots": shot_ids, "frames": frames_meta, "fps": 24.0, "input_color": "linear-rec709",
            "camera": cam.as_dict(), "input_dir": str(inp_dir), "target_dir": str(tgt_dir),
            "note": "input = camera-simulated source (clean + designated reflection); target = same simulation of clean"}
    (ev_dir / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    (ex_dir / "case.json").write_text(json.dumps({"case": case, "frames": k, "fps": 24.0, "input_color": "linear-rec709",
                                                   "input_pattern": str(inp_dir / "frame_%06d.exr")}, indent=1), encoding="utf-8")
    return meta


def build_all(run_dir: Path, seed: int, dev: torch.device) -> dict:
    out = {}
    for case, sids in demo_cases(run_dir).items():
        out[case] = build_case(run_dir, case, sids, dev, seed)
    return out

"""Verify the paired-target construction before training (GPU statistics on Cycles renders).

Checks (isolated scene, high spp, no adaptive sampling, undenoised FP32):
  E1 beauty == clean + reflection in expectation on the ocular surface
  E2 clean == (1-F) * unit_transmission on the central cornea (transmission weight preserved,
     reflected energy absorbed rather than redistributed)
  E3 refraction active: apparent iris/pupil areas through the cornea exceed the IOR-1 geometry
  E4 off-eye pixels unchanged between beauty and clean (non-camera transport untouched)
  E5 reflection layer exactly zero away from the camera-visible ocular surface
  E6 dark pupil in the clean render coincides with the through-refraction pupil label
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from ..gpu.imgops import dilate, erode
from ..io.exr import read_render_exr
from ..render.blender import run_script

DEFAULT_EYE = dict(limbus_ratio=0.49, cornea_R_ratio=0.65, asphericity=-0.26, iris_recess_ratio=0.045, iris_bulge_ratio=0.02,
                   pupil_min_ratio=0.2, pupil_max_ratio=0.6, n_eff=1.3375, ior_surface=1.376, ior_tear=1.336,
                   cornea_roughness=0.01, sclera_sss=0.5, sclera_sss_scale=0.002, iris_bump=0.25)


def fresnel_dielectric(cos_i: float, n: float) -> float:
    """Unpolarized Fresnel reflectance (air -> n)."""
    sin_t = math.sqrt(max(0.0, 1 - cos_i * cos_i)) / n
    if sin_t >= 1:
        return 1.0
    cos_t = math.sqrt(1 - sin_t * sin_t)
    rs = (cos_i - n * cos_t) / (cos_i + n * cos_t)
    rp = (n * cos_i - cos_t) / (n * cos_i + cos_t)
    return 0.5 * (rs * rs + rp * rp)


def _rgb(d: dict, layer: str) -> torch.Tensor:
    return torch.stack([d[f"{layer}.Combined.{c}"] for c in "RGB"])


def run(out_dir: Path, textures: dict, samples: int = 1024, size: int = 192, seed: int = 20260910, clean_iris_render: bool = False) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    job = {
        "eye": DEFAULT_EYE, "textures": textures, "samples": samples, "width": size, "height": size, "seed": seed,
        "camera_distance_m": 0.14, "out_dir": str(out_dir),
        "lights": [
            {"family": "rect_softbox", "azimuth_deg": 18, "elevation_deg": 15, "distance_m": 0.9, "size": [0.4, 0.6], "power": 60, "color": [1, 0.97, 0.93]},
            {"family": "bulb", "azimuth_deg": -25, "elevation_deg": 8, "distance_m": 0.6, "size": [0.01, 0.01], "power": 4, "color": [1, 0.85, 0.7]},
        ],
        "world": {"seed": 5, "strength": 0.3, "panel_density": 0.2, "panel_gain": 2.0},
        "clean_iris_render": bool(clean_iris_render),
    }
    jp = out_dir / "job.json"
    jp.write_text(json.dumps(job, indent=1), encoding="utf-8")
    blender_run = run_script("pair_validation.py", ["--job", str(jp)], out_dir / "blender.log")
    scene = json.loads((out_dir / "validation_scene.json").read_text(encoding="utf-8"))
    return analyze(out_dir, scene, blender_run)


def analyze(out_dir: Path, scene: dict, blender_run: dict | None = None) -> dict:
    dev = torch.device("cuda")
    d = read_render_exr(out_dir / "main.exr", dev)
    d1 = read_render_exr(out_dir / "ior1_labb.exr", dev)
    beauty, clean, refl = _rgb(d, "beauty"), _rgb(d, "clean"), _rgb(d, "reflection")
    unit = _rgb(d, "unit_transmission")
    la, lb = _rgb(d, "lab_a"), _rgb(d, "lab_b")
    lb1 = _rgb(d1, "lab_b")
    H, W = beauty.shape[1:]
    ocular = erode(la[0] > 0.99, 1)
    res: dict = {"scene": {k: scene[k] for k in ("samples", "width", "height", "ior_surface", "n_eff")}, "checks": {}}
    # E1
    diff = (beauty - (clean + refl)).mean(0)[ocular]
    mb = beauty.mean(0)[ocular].mean()
    n = diff.numel()
    mean_d = float(diff.mean())
    se = float(diff.std() / math.sqrt(max(n, 1)))
    rel = abs(mean_d) / float(mb)
    res["checks"]["E1_beauty_equals_clean_plus_reflection"] = {
        "pixels": n, "mean_diff": mean_d, "standard_error": se, "relative_mean_diff": rel,
        "passed": bool(n > 500 and abs(mean_d) <= 4 * se + 0.002 * float(mb))}
    # E2 (central cornea over visible iris, small incidence angles)
    ys, xs = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float32), torch.arange(W, device=dev, dtype=torch.float32), indexing="ij")
    ax, ay = scene["apex_px"]
    limb = torch.tensor(scene["limbus_px"], device=dev)
    r_limb = float(((limb[:, 0] - ax) ** 2 + (limb[:, 1] - ay) ** 2).sqrt().mean())
    rpx = ((xs + 0.5 - ax) ** 2 + (ys + 0.5 - ay) ** 2).sqrt()
    iris_c = (rpx < 0.8 * r_limb) & (lb[0] > 0.99) & ocular  # visible iris seen through the cornea
    g = scene["geom"]
    # expected per-pixel (1-F): incidence angle from the conic surface normal at corneal height h
    # (near-axial camera; h estimated from the projected radius), averaged over the same pixels
    h = (rpx[iris_c] / r_limb * g["rl"]).clamp(max=0.999 * g["rl"])
    Rc, Q = g["Rc"], g["Q"]
    dzdh = h / (Rc * torch.sqrt(torch.clamp(1 - (1 + Q) * h * h / (Rc * Rc), min=1e-9)))
    cos_i = (1 / torch.sqrt(1 + dzdh * dzdh)).tolist() if iris_c.any() else []
    w_unit = unit.mean(0)[iris_c]
    exp_w = [(1 - fresnel_dielectric(ci, scene["ior_surface"])) for ci in cos_i]
    expected = float((torch.tensor(exp_w, device=dev) * w_unit).sum() / w_unit.sum()) if cos_i else float("nan")
    ratio = float(clean.mean(0)[iris_c].sum() / unit.mean(0)[iris_c].sum()) if iris_c.any() else float("nan")
    res["checks"]["E2_clean_transmission_weight"] = {
        "pixels": int(iris_c.sum()), "measured_clean_over_unit": ratio, "expected_one_minus_F": expected,
        "passed": bool(iris_c.sum() > 200 and abs(ratio - expected) < 0.01 and ratio < 0.99)}
    # E3 refraction active: the pupil seen through the cornea is magnified relative to IOR 1
    # (iris+pupil together are bounded by the corneal silhouette, so only the pupil is informative)
    p_ref, p_1 = float(lb[1].sum()), float(lb1[1].sum())
    res["checks"]["E3_refraction_active"] = {
        "apparent_pupil_area_px": p_ref, "ior1_pupil_area_px": p_1, "pupil_area_magnification": p_ref / max(p_1, 1),
        "passed": bool(p_ref / max(p_1, 1) > 1.05)}
    # E4 off-eye unchanged
    anylab = (la.sum(0) > 0) | (lb.sum(0) > 0)
    offeye = ~dilate(anylab | (refl.abs().sum(0) > 0), 3)
    d4 = (beauty - clean).mean(0)[offeye]
    m4 = float(beauty.mean(0)[offeye].mean())
    se4 = float(d4.std() / math.sqrt(max(d4.numel(), 1)))
    res["checks"]["E4_offeye_unchanged"] = {
        "pixels": int(offeye.sum()), "mean_diff": float(d4.mean()), "standard_error": se4, "mean_level": m4,
        "passed": bool(offeye.sum() > 1000 and abs(float(d4.mean())) <= 4 * se4 + 0.002 * m4)}
    # E5 reflection exactly zero away from the ocular surface
    away = ~dilate(la[0] > 0, 2)
    nz = int(((refl.abs().sum(0) > 0) & away).sum())
    res["checks"]["E5_reflection_zero_offeye"] = {"nonzero_pixels": nz, "pixels_checked": int(away.sum()), "passed": nz == 0}
    # E6 dark pupil vs label
    lum = clean.mean(0)
    inside = (lb[0] + lb[1]) > 0.99
    iris_med = float(lum[(lb[0] > 0.99) & ocular].median())
    dark = inside & (lum < 0.2 * iris_med)
    lab_p = lb[1] > 0.5
    inter = float((dark & lab_p).sum())
    union = float((dark | lab_p).sum())
    iou = inter / max(union, 1)
    res["checks"]["E6_pupil_label_consistency"] = {"iou": iou, "passed": bool(iou > 0.85)}
    res["passed"] = all(c["passed"] for c in res["checks"].values())
    res["blender"] = blender_run
    res["evidence"] = [str(out_dir / "main.exr"), str(out_dir / "ior1_labb.exr"), str(out_dir / "validation_scene.json")]
    (out_dir / "pair_validation_report.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res

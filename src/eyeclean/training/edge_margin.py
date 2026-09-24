"""Validation-based choice of the full-resolution gate parameters (config ``inference.gate_selection``).

The final gate keeps edits on the visible ocular surface: ``ocular & ~protected`` (protected =
lashes, tear line, lids), an optional full-resolution band around protected pixels, and an erosion
by ``max(erode_min_px, erode_frac * iris_px)``; reflection pixels are those selected by the guard's
reflection mask or the restorer's support head.  On validation data the dominant off-eye errors are
thin lash strands and the tear line inside the eye opening (missed by the guard), while at small
eye sizes a 2 px erosion removes most of a catchlight lying against the upper lid (seen on real
footage), and the default mask thresholds leave most faint reflection unselected.

This analysis reproduces the inference gate exactly at each validation eye's *native* resolution
(guard + restorer on the crop, inverse warp, protected band, erosion) for every candidate setting,
scored against the rendered labels: off-eye edited pixels (ocular coverage < 0.004, the evaluation's
definition), edited pixels on no-reflection negatives, and coverage of the recoverable reflection.
Selection: the highest masked-MAE reduction over the recoverable reflection (the primary quality
gate, computed with the pipeline's own warped and feathered correction) among settings with **zero**
off-eye edits **and** zero negative edits; if none qualifies, the fewest violations (reported as such).  Test data is never used; the
frozen gates are still measured on the held-out test cases.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import torch

from ..data.sampling import stable_seed
from ..gpu.runtime import require_cuda
from ..gpu.imgops import inverse_warp
from ..inference.gate import crop_support, full_res_gate
from ..models.eyerestore import build_restore
from ..models.inputs import clipped_mask, crop_exposure, denormalize, guard_input, normalize, restore_input
from .data import CacheStore, TrainSampler
from .trainer import load_guard

# Lid-margin/feather settings, as one axis: the floor is meaningless in the other space.  "native"
# quantises the margin to whole frame pixels (legacy); "crop" computes the same fraction of the iris
# at the masks' own resolution.  See inference.gate.full_res_gate.
MARGINS = ({"margin_space": "native", "erode_min_px": 2},  # legacy default
           {"margin_space": "crop", "erode_min_crop_px": 4, "feather_min_crop_px": 4},
           # a margin fixed in crop pixels only: the iris is 80-111 crop px wide by construction, so
           # these are ~1-2% of the iris rather than the 5% that erode_frac asks for
           {"margin_space": "crop", "erode_min_crop_px": 2, "feather_min_crop_px": 2, "erode_frac": 0.0, "feather_frac": 0.0},
           {"margin_space": "crop", "erode_min_crop_px": 1, "feather_min_crop_px": 1, "erode_frac": 0.0, "feather_frac": 0.0},
           # Round 4: the 5% margin (index 1) leaked 1-3 px on one nearly closed 110 px eye under the full-pool guard;
           # 6-7% of the iris is legal there and still only ~1.1-1.2 native px at a 17.7 px iris
           {"margin_space": "crop", "erode_min_crop_px": 4, "feather_min_crop_px": 4, "erode_frac": 0.06, "feather_frac": 0.06},
           {"margin_space": "crop", "erode_min_crop_px": 4, "feather_min_crop_px": 4, "erode_frac": 0.07, "feather_frac": 0.07},
           # Round 4: 7% for large eyes, a fixed 3 crop px (~0.6 native px at a 17.7 px iris) below 48 px - every
           # lid-edge leak measured under the full-pool guard was on irises >= 77 px (0/0 on 500 eyes, measured)
           {"margin_space": "crop", "erode_min_crop_px": 4, "feather_min_crop_px": 4, "erode_frac": 0.07, "feather_frac": 0.07,
            "small_iris_px": 48.0, "erode_min_crop_px_small": 3, "erode_frac_small": 0.0})

# ocular_thresh is the *confidence-shaped* margin and matters more than the geometric one: a uniform
# erosion shrinks the support everywhere, including deep inside the iris where the guard is certain,
# while raising this threshold pulls the boundary in only where the guard is unsure - which is where
# lid and lash pixels leak in.  Measured off-eye edits come from the eye-opening boundary, not the
# interior, so this buys safety at almost no cost in reflection coverage.
# reflection/support thresholds also go below 0.35: `selected = (support | reflection) & surface` is
# where coverage drops from 64% to 41% on small eyes, and a false positive there is cheap - the
# restorer is trained towards a zero residual off the reflection - while a miss is unrecoverable.
# ocular_thresh is kept at two values: 0.99 reached zero off-eye edits but cost 40% of overall
# quality (measured), so only the useful end of that axis is searched.
GRID = {"ocular_thresh": (0.5, 0.9), "protected_thresh": (0.5, 0.2), "protected_dilate_px": (0, 1),
        "edge_margin": tuple(range(len(MARGINS))), "reflection_thresh": (0.5, 0.35, 0.2), "support_thresh": (0.5, 0.35, 0.2),
        "conf_thresh": (None, 0.1), "feather_mode": ("surface",)}
# NOTE: every gate key that changes crop_support's masks must be listed here - the harness caches one
# crop_support result per tuple of these values, so a key left out would silently share another
# setting's masks.  iris_only_thresh (Round 19) reshapes `surface`, hence its presence.
_CROP_KEYS = ("ocular_thresh", "protected_thresh", "reflection_thresh", "support_thresh", "conf_thresh",
              "iris_only_thresh")
# the full-resolution gate's geometry; together with _CROP_KEYS these decide `hard`/`feather`
_GEOM_KEYS = ("margin_space", "erode_min_px", "erode_min_crop_px", "feather_min_crop_px", "erode_frac",
              "feather_frac", "feather_mode", "protected_dilate_px", "small_iris_px", "erode_min_crop_px_small", "erode_frac_small")
# these only decide whether the eye is edited at all, so they are applied to the finished gate
_PRESENCE_KEYS = ("refl_present_thresh", "min_mean_correction", "min_reflection_area_crop_px")

# Eye-size cohorts (native iris diameter in pixels).  The erosion floor only binds on small eyes -
# at 18 px a 2 px margin is 11% of the iris from every side, at 100 px it is 2% - so a score summed
# over pixels is dominated by large eyes and cannot see the choice at all.  Cohorts are reported and
# the small cohort breaks ties.  Real footage at 1080p sits in the small cohort.
COHORTS = ((0.0, 24.0, "iris_lt_24px"), (24.0, 48.0, "iris_24_48px"), (48.0, 1e9, "iris_ge_48px"))


def _cohort(iris_px: float) -> int:
    return next(i for i, (lo, hi, _) in enumerate(COHORTS) if lo <= iris_px < hi)


@torch.no_grad()
def select_gate_params(run_dir: Path, cfg: dict, restore_path: Path, gate: dict, n: int = 400, temporal: bool = True,
                       grid: dict | None = None) -> dict:
    """temporal=True scores the five-frame model (default inference path, centre frame valid only)."""
    from safetensors.torch import load_file

    dev = require_cuda()
    guard = load_guard(run_dir, dev)
    presence = None
    pp = Path(run_dir) / "checkpoints" / "presence" / "best.safetensors"
    if pp.exists():
        from ..models.presence import ReflectionPresence

        presence = ReflectionPresence().to(dev).eval()
        presence.load_state_dict(load_file(str(pp), device=str(dev)))
    if grid is None:
        # refl_present_thresh 0.0 disables the classifier, min_mean_correction 0.0 disables the
        # residual test; the measurement decides which piece of evidence (or both) the gate uses
        mmc = (0.0, 0.02, 0.045, 0.08)
        grid = (dict(GRID, refl_present_thresh=(0.0, 0.5, 0.8), min_mean_correction=mmc)
                if (guard.presence_head or presence is not None) else dict(GRID, min_mean_correction=mmc))
    model = build_restore(cfg["model"], checkpoint=False).to(dev).eval()
    model.load_state_dict(load_file(str(restore_path), device=str(dev)))
    base = dict(gate) if temporal else dict(gate, conf_thresh=gate.get("conf_thresh_single", gate["conf_thresh"]))
    store = CacheStore(run_dir, "validation", kinds=("still", "clip"))
    smp = TrainSampler(store, dev, stable_seed(cfg["seed"], "gate_selection"), cfg["model"]["crop_size"],
                       float(cfg["inference"]["minimum_iris_diameter_pixels"]), sim=cfg["data"].get("camera_sim"))
    keys = list(grid)
    combos = [dict(zip(keys, v)) for v in itertools.product(*(grid[k] for k in keys))]
    for c in combos:  # None = keep the calibrated confidence threshold
        if c.get("conf_thresh", 0) is None:
            c["conf_thresh"] = base["conf_thresh"]
        if "edge_margin" in c:
            c.update(MARGINS[c.pop("edge_margin")])
    combos = [dict(t) for t in {tuple(sorted(c.items())) for c in combos}]
    zero = {"offeye_px": 0, "eyes_with_offeye": 0, "negative_edited_px": 0, "negatives_edited": 0, "edited_px": 0,
            "recoverable_px": 0, "covered_px": 0, "err_in": 0.0, "err_pred": 0.0, "eye_red_sum": 0.0, "eye_n": 0}
    st = [dict(zero, params=c, coh_in=[0.0] * len(COHORTS), coh_pred=[0.0] * len(COHORTS), coh_n=[0] * len(COHORTS),
               coh_cov=[0] * len(COHORTS), coh_rec=[0] * len(COHORTS)) for c in combos]
    coh_seen = [0] * len(COHORTS)
    used = negs = 0
    for _ in range(n):
        it = smp.sample_native()
        x = it["src"][None]
        e = crop_exposure(x)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            go = guard(guard_input(x, e))
        probs = torch.sigmoid(go["masks"].float())
        if presence is not None:
            from ..models.presence import presence_input

            with torch.autocast("cuda", dtype=torch.bfloat16):
                rp = float(torch.sigmoid(presence(presence_input(guard_input(x, e), probs)).float())[0])
        else:
            rp = float(torch.sigmoid(go["refl_present"].float())[0]) if "refl_present" in go else None
        inp = restore_input(x, e, probs, clipped_mask(x, 1.0))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if temporal:
                valid = torch.zeros(1, 5, dtype=torch.bool, device=dev)
                valid[:, 2] = True
                out = model.forward_video(inp[None].expand(1, 5, *inp.shape[1:]).contiguous(), valid)
            else:
                out = model(inp)
        o1 = {k: v[0] for k, v in out.items()}
        # the correction exactly as the pipeline builds it: crop-space delta, inverse-warped, feathered
        delta_c = (denormalize(normalize(x, e) + out["residual"].float(), e) - x)[0]
        nat = it["native"]
        N = nat["N"]
        ocular = nat["labels"][0]
        is_neg = bool(it["negative"])
        diff = (normalize(nat["tgt"][None], e) - normalize(nat["src"][None], e)).abs().amax(1)[0]
        rec = (diff > 0.01) & ~nat["clipped"] & nat["valid"]
        n_rec = 0 if is_neg else int(rec.sum())
        tgt_nn = normalize(nat["tgt"][None], e)[0]
        src_nn = normalize(nat["src"][None], e)[0]
        delta_n = inverse_warp(delta_c, nat["tf"], (0, 0, N, N))
        e_in = float((src_nn - tgt_nn).abs().mean(0)[rec].sum()) if n_rec else 0.0
        used += 1
        negs += int(is_neg)
        ci = _cohort(float(it["native_iris"]))
        if not is_neg:
            coh_seen[ci] += 1
        # The presence tests only ever zero the whole eye, so the masks are built once per
        # (crop thresholds, gate geometry) with presence forced on, and each presence setting is
        # applied afterwards as a per-eye boolean.  Exact, and ~12x fewer full_res_gate calls.
        comps, frs = {}, {}
        empty = torch.zeros_like(nat["valid"])
        for s in st:
            g = dict(base, **s["params"])
            ck = tuple(g.get(k) for k in _CROP_KEYS)
            if ck not in comps:
                comps[ck] = crop_support(probs[0], o1, "conservative",
                                         dict(g, refl_present_thresh=0.0, min_mean_correction=0.0,
                                              min_reflection_area_crop_px=0.0), refl_present=rp)
            comp = comps[ck]
            gk = ck + tuple(g.get(k) for k in _GEOM_KEYS)
            if gk not in frs:
                frs[gk] = full_res_gate(comp, nat["tf"], (0, 0, N, N), float(it["native_iris"]), g, None)
            present = (comp["area"] >= g["min_reflection_area_crop_px"]
                       and comp["mean_corr"] >= g.get("min_mean_correction", 0.0)
                       and (rp is None or rp > g.get("refl_present_thresh", 0.5)))
            fr = frs[gk] if present else {"hard": empty, "feather": empty.float()}
            H = fr["hard"] & nat["valid"]
            h = int(H.sum())
            if is_neg:
                s["negative_edited_px"] += h
                s["negatives_edited"] += int(h > 0)
                continue
            off = int((H & (ocular < 0.004)).sum())
            s["offeye_px"] += off
            s["eyes_with_offeye"] += int(off > 0)
            s["edited_px"] += h
            s["recoverable_px"] += n_rec
            cov = int((H & rec).sum())
            s["covered_px"] += cov
            if n_rec:
                A = fr["feather"] * H.float()
                pred = nat["src"] + delta_n * A[None]
                e_pred = float((normalize(pred[None], e)[0] - tgt_nn).abs().mean(0)[rec].sum())
                s["err_in"] += e_in
                s["err_pred"] += e_pred
                s["coh_in"][ci] += e_in
                s["coh_pred"][ci] += e_pred
                s["coh_n"][ci] += 1
                s["coh_cov"][ci] += cov
                s["coh_rec"][ci] += n_rec
                if e_in > 0:  # per-eye reduction: every eye counts once, regardless of its pixel area
                    s["eye_red_sum"] += 1 - e_pred / e_in
                    s["eye_n"] += 1
    for s in st:
        s["coverage"] = s["covered_px"] / s["recoverable_px"] if s["recoverable_px"] else 0.0
        s["masked_mae_reduction"] = 1 - s["err_pred"] / s["err_in"] if s["err_in"] > 0 else 0.0
        s["per_eye_mae_reduction"] = s["eye_red_sum"] / s["eye_n"] if s["eye_n"] else 0.0
        s["cohorts"] = {name: {"eyes": s["coh_n"][i],
                               "masked_mae_reduction": (1 - s["coh_pred"][i] / s["coh_in"][i]) if s["coh_in"][i] > 0 else 0.0,
                               "coverage": (s["coh_cov"][i] / s["coh_rec"][i]) if s["coh_rec"][i] else 0.0}
                        for i, (_, _, name) in enumerate(COHORTS)}
        for k in ("coh_in", "coh_pred", "coh_n", "coh_cov", "coh_rec", "eye_red_sum", "eye_n"):
            s.pop(k)
    small = COHORTS[0][2]
    ok = [s for s in st if s["offeye_px"] == 0 and s["negative_edited_px"] == 0]
    if ok:
        # per-eye (not per-pixel) reduction so small eyes count; ties go to the small cohort, which is
        # the regime the erosion floor actually decides and where real 1080p footage sits
        best = max(ok, key=lambda s: (s["per_eye_mae_reduction"], s["cohorts"][small]["masked_mae_reduction"], s["coverage"]))
        verdict = "zero off-eye and zero negative edits on validation"
    else:
        best = min(st, key=lambda s: (s["offeye_px"] + s["negative_edited_px"], -s["per_eye_mae_reduction"]))
        verdict = "no setting reached zero off-eye and negative edits on validation; fewest violations selected"
    dflt = {"protected_thresh": 0.5, "protected_dilate_px": 0, "margin_space": "native", "erode_min_px": 2,
            "reflection_thresh": 0.5, "support_thresh": 0.5, "feather_mode": "support",
            "conf_thresh": base["conf_thresh"]}  # the untuned gate, for comparison
    default = next((s for s in st if all(s["params"].get(k) == v for k, v in dflt.items())), None)
    pick = ("offeye_px", "eyes_with_offeye", "negative_edited_px", "negatives_edited", "coverage", "masked_mae_reduction",
            "per_eye_mae_reduction", "edited_px", "cohorts")
    return {"method": "full-resolution gate reproduced at native eye resolution on validation crops (oracle geometry), "
                      + ("five-frame model, centre frame" if temporal else "single-frame model")
                      + "; selected on the per-eye (not per-pixel) masked-MAE reduction, ties to the smallest eye cohort",
            "grid": {k: list(v) for k, v in grid.items()}, "samples": used, "negative_samples": negs, "results": st,
            "cohort_eyes": {name: coh_seen[i] for i, (_, _, name) in enumerate(COHORTS)},
            "chosen": best["params"], "chosen_stats": {k: best[k] for k in pick},
            "default_stats": None if default is None else {k: default[k] for k in pick},
            "verdict": verdict, "note": "test data is never used; the frozen gates are still measured on the test cases"}

"""Assemble the deliverable checkpoint bundle and calibrate reconstruction confidence.

Calibration (held-out validation crops, never test): raw confidence is histogram-binned against
the measured event |prediction - target| < tolerance in normalised space over the correction
support.  The gate's confidence threshold is the lowest raw value whose bin accuracy reaches the
target precision.  The exported uncertainty remains "1 - raw confidence" and is documented as a
raw score; the reliability table is stored for interpretation.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from ..data.sampling import stable_seed
from ..gpu.runtime import require_cuda
from ..inference.gate import DEFAULT_GATE
from ..models.bundle import write_bundle
from ..models.eyerestore import build_restore
from ..models.inputs import crop_exposure
from ..validation.acceptance import acceptance_sha256
from .data import CacheStore, TrainSampler, targets_from_batch
from .trainer import _restore_forward, load_guard

TOL = 0.02


@torch.no_grad()
def calibrate(run_dir: Path, cfg: dict, restore_path: Path, n: int = 96, bins: int = 10, precision: float = 0.8,
              temporal: bool = False) -> dict:
    """temporal=True calibrates the five-frame model on consecutive validation clip windows
    (centre frame), i.e. the same forward path conservative inference uses by default."""
    from safetensors.torch import load_file

    from .trainer import _window_inputs

    dev = require_cuda()
    guard = load_guard(run_dir, dev)
    model = build_restore(cfg["model"], checkpoint=False).to(dev).eval()
    model.load_state_dict(load_file(str(restore_path), device=str(dev)))
    store = CacheStore(run_dir, "validation", kinds=("still", "clip"))
    smp = TrainSampler(store, dev, stable_seed(cfg["seed"], "calibration", "temporal" if temporal else "single"), cfg["model"]["crop_size"],
                       float(cfg["inference"]["minimum_iris_diameter_pixels"]), sim=cfg["data"].get("camera_sim"))
    hist_n = torch.zeros(bins, device=dev)
    hist_ok = torch.zeros(bins, device=dev)
    used = 0
    for _ in range(n):
        if temporal:
            wb = smp.clip_window(5)
            xs, e5, valid = _window_inputs(guard, wb)
            if not bool(valid[0, 2]):
                continue
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model.forward_video(xs, valid)
            b = {k: (v[2:3] if torch.is_tensor(v) and v.shape[:1] == (5,) else v) for k, v in wb.items()}
            e = e5[2:3]
        else:
            b = smp.batch(1)
            out, e, _, _, _ = _restore_forward(model, guard, b)
        gt = targets_from_batch(b, e)
        w = (gt["support"] * b["valid"])[0, 0] > 0.5
        if not bool(w.any()):
            continue
        used += 1
        conf = torch.sigmoid(out["conf"].float())[0, 0][w]
        err_p = (gt["src_n"] + out["residual"].float() - gt["tgt_n"]).abs().amax(1)[0][w]
        if cfg["model"].get("conf_target") == "improvement":  # same event the conf head was trained on
            err_s = (gt["src_n"] - gt["tgt_n"]).abs().amax(1)[0][w]
            ok = ((err_p <= 0.5 * err_s) | (err_p < TOL)).float()
        else:
            ok = (err_p < TOL).float()
        idx = (conf * bins).long().clamp(0, bins - 1)
        hist_n.index_add_(0, idx, torch.ones_like(conf))
        hist_ok.index_add_(0, idx, ok)
    acc = (hist_ok / hist_n.clamp(min=1)).tolist()
    cnt = hist_n.tolist()
    thr, reached = 0.5, False
    for k in range(bins):
        if cnt[k] >= 50 and all(acc[j] >= precision for j in range(k, bins) if cnt[j] >= 50):
            thr, reached = k / bins, True
            break
    return {"method": "histogram binning on validation " + ("clip windows (five-frame model)" if temporal else "crops (single-frame model)"),
            "tolerance_normalised": TOL, "bins": bins, "bin_accuracy": acc, "bin_count": cnt, "target_precision": precision,
            "target_reached": reached, "conf_thresh": thr, "n_samples": n, "n_with_support": used,
            "note": ("uncertainty layers export 1 - raw confidence; use this table to interpret it"
                     + ("" if reached else "; no confidence bin reached the target precision, so the default 0.5 threshold is kept "
                        "(correction quality is then governed by the other gates and is reported by the evaluation)"))}


def assemble(cfg: dict, calibrate_conf: bool = True) -> dict:
    run_dir = Path(cfg["run_dir"]).resolve()
    ck = run_dir / "checkpoints"
    parts_best = {"guard": ck / "guard" / "best_ema.safetensors", "restore": ck / "temporal" / "best_ema.safetensors",
                  "restore_single": ck / "image" / "best_ema.safetensors"}
    parts_last = {"guard": ck / "guard" / "last_ema.safetensors", "restore": ck / "temporal" / "last_ema.safetensors",
                  "restore_single": ck / "image" / "last_ema.safetensors"}
    for p in list(parts_best.values()) + list(parts_last.values()):
        if not p.exists():
            raise RuntimeError(f"missing stage checkpoint {p}")
    if (ck / "presence" / "best.safetensors").exists():  # optional v2 presence classifier
        parts_best["presence"] = ck / "presence" / "best.safetensors"
        parts_last["presence"] = ck / "presence" / "last.safetensors"
    if calibrate_conf:
        cal = {"temporal": calibrate(run_dir, cfg, parts_best["restore"], temporal=True),
               "single": calibrate(run_dir, cfg, parts_best["restore_single"])}
    else:
        cal = {"temporal": {"conf_thresh": 0.5, "method": "none"}, "single": {"conf_thresh": 0.5, "method": "none"}}
    gate = dict(DEFAULT_GATE, conf_thresh=float(cal["temporal"]["conf_thresh"]), conf_thresh_single=float(cal["single"]["conf_thresh"]))
    gate_sel = None
    if cfg["inference"].get("gate_selection") == "validation":  # optional (v2): validation-selected gate thresholds
        from .edge_margin import select_gate_params

        gate_sel = select_gate_params(run_dir, cfg, parts_best["restore"], gate, temporal=True)
        gate.update(gate_sel["chosen"])
    plan = json.loads((run_dir / "manifests" / "dataset_plan.json").read_text(encoding="utf-8"))
    summaries = {}
    for st in ("guard", "image", "temporal"):
        sp = ck / st / "summary.json"
        if sp.exists():
            s = json.loads(sp.read_text(encoding="utf-8"))
            summaries[st] = {k: s.get(k) for k in ("steps", "parameters", "best", "adaptations", "train_seconds", "memory", "sample_mix")}
    manifest = {"profile": cfg["profile"], "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "model": cfg["model"], "gate": gate, "calibration": cal, "training": summaries, "gate_selection": gate_sel,
                "dataset_plan_digest": plan["digest"], "acceptance_sha256": acceptance_sha256(),
                "working_color": cfg["color"]["working_color"], "crop_size": cfg["model"]["crop_size"]}
    best = write_bundle(ck / "best.safetensors", parts_best, dict(manifest, selection="best validation score per stage (EMA)"))
    last = write_bundle(ck / "last.safetensors", parts_last, dict(manifest, selection="last EMA weights per stage"))
    return {"best": best["path"], "best_sha256": best["sha256"], "last": last["path"], "last_sha256": last["sha256"], "calibration": cal}

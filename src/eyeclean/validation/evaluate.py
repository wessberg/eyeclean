"""Held-out evaluation of full-frame pipeline runs on the demo sequences (targets read AFTER prediction).

Predictions are reconstructed from the exported authoritative layers only (signed corrections +
hard masks), exactly as `apply-patches` would, so the evaluation measures what a user receives.
Metrics (all on held-out test identities/scenes; buckets reported, failures kept visible):
  * recoverable masked MAE reduction vs the unprocessed input (normalised space, target-differs
    pixels that are not sensor-clipped), aggregate and per case
  * recoverable reflection coverage by the final support; abstention and synthesis fractions
  * changed pixels on the controlled no-reflection case and off-eye (label ocular == 0) pixels
  * pupil area relative error (eyes with iris >= 24 px) from an independent evaluator that is
    first validated against render labels on the targets
  * temporal error (prediction error differences between consecutive eye-aligned frames) of the
    five-frame model divided by that of the trained single-frame model
  * independent full-frame detection recall and centre error against generator eye positions
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from ..color import transforms as CT
from ..gpu.imgops import CropTransform, dilate, extract_crop
from ..inference.compositor import apply_signed_correction
from ..io.exr import read_exr
from .anatomy_eval import pupil_area

ALPHA = CT.ASINH_ALPHA


def _read_rgb(p: Path, dev) -> torch.Tensor:
    img = read_exr(p, dev)
    return torch.stack([img.channels[c].float() for c in "RGB"])


def _labels(p: Path, dev) -> dict:
    img = read_exr(p, dev)
    return {k: v.float() for k, v in img.channels.items()}


def reconstruct(proc: Path, index: dict, k: int, reference: torch.Tensor) -> dict:
    """Prediction and masks for frame k from the exported layers."""
    H, W = reference.shape[1:]
    dev = reference.device
    pred = reference.clone()
    sup = torch.zeros((H, W), dtype=torch.bool, device=dev)
    syn = torch.zeros_like(sup)
    abst = torch.zeros_like(sup)
    sel = torch.zeros_like(sup)
    for eye in index.get(k, {}).get("eyes", []):
        if "patch" not in eye:
            continue
        p = read_exr(proc / eye["patch"], dev)
        m = read_exr(proc / eye["mask"], dev)
        dw = p.header.data_window
        box = (dw[0], dw[1], dw[2] + 1, dw[3] + 1)
        delta = torch.stack([p.channels[c].float() for c in "RGB"])
        hard = m.channels["hard"].float() > 0.5
        apply_signed_correction(reference, delta, hard, box, pred)
        sl = (slice(box[1], box[3]), slice(box[0], box[2]))
        sup[sl] |= hard
        syn[sl] |= m.channels["synth"].float() > 0.5
        abst[sl] |= m.channels["abstained"].float() > 0.5
        sel[sl] |= m.channels["selected"].float() > 0.5
    return {"pred": pred, "support": sup, "synth": syn, "abstained": abst, "selected": sel}


def _n(x: torch.Tensor, e: float) -> torch.Tensor:
    return torch.asinh(ALPHA * x / e) / math.asinh(ALPHA)


def _eye_center(eye: dict) -> tuple[float, float, float]:
    lx = [p[0] for p in eye["limbus_px"]]
    ly = [p[1] for p in eye["limbus_px"]]
    return sum(lx) / len(lx) - 0.5, sum(ly) / len(ly) - 0.5, eye["iris_diameter_px"]


def evaluate_case(run_dir: Path, case: str, procs: dict[str, Path], dev) -> dict:
    ev = run_dir / "eval_private" / case
    meta = json.loads((ev / "meta.json").read_text(encoding="utf-8"))
    idx = {m: {e["frame"]: e for e in json.loads((p / "patches" / "index.json").read_text(encoding="utf-8"))["frames"]} for m, p in procs.items()}
    inp_dir = Path(meta["input_dir"])
    acc = {m: {"err_pred": 0.0, "err_in": 0.0, "rec_px": 0, "covered": 0, "support_px": 0, "offeye_px": 0, "abstained_px": 0,
               "selected_px": 0, "synth_px": 0, "pupil_rel": [], "temporal": [], "temporal_px": 0} for m in procs}
    eval_valid = []
    prev = {m: {} for m in procs}
    e_case = None
    negative_case = all(f["negative"] for f in meta["frames"])
    for f in meta["frames"]:
        k = f["index"]
        inp = CT.to_working(_read_rgb(inp_dir / f"frame_{k:06d}.exr", dev), meta["input_color"])
        tgt = CT.to_working(_read_rgb(ev / "target_exr" / f"frame_{k:06d}.exr", dev), meta["input_color"])
        lab = _labels(ev / "labels" / f"frame_{k:06d}.exr", dev)
        if e_case is None:
            lum = 0.2722 * tgt[0] + 0.6741 * tgt[1] + 0.0537 * tgt[2]
            skin = lab["skin"] > 0.5
            e_case = float(2 * (lum[skin].median() if bool(skin.sum() > 100) else lum.median()).clamp(min=1e-4))
        n_in, n_tg = _n(inp, e_case), _n(tgt, e_case)
        differs = (n_tg - n_in).abs().amax(0) > 0.01
        rec = differs & (lab["clipped"] < 0.5)
        lumt = 0.2722 * tgt[0] + 0.6741 * tgt[1] + 0.0537 * tgt[2]
        for s, eye in f["eyes"].items():
            cx, cy, D = _eye_center(eye)
            if D >= 24 and eye["in_frame"] and eye["openness"] > 0.5:
                S = int(3 * D)
                tf = CropTransform(cx, cy, 1.0, 0.0, S)
                ir = extract_crop(lab["iris"][None], tf, mode="nearest")[0]
                pu = extract_crop(lab["pupil"][None], tf, mode="nearest")[0]
                a_lab = float((pu > 0.5).sum())
                a_eval = pupil_area(extract_crop(lumt[None], tf, mode="nearest")[0], ir, pu)
                if a_eval is not None and a_lab > 4:
                    eval_valid.append(abs(a_eval - a_lab) / a_lab)
        for m, proc in procs.items():
            r = reconstruct(proc, idx[m], k, inp)
            n_pr = _n(r["pred"], e_case)
            a = acc[m]
            if not negative_case:
                a["err_pred"] += float((n_pr - n_tg).abs().mean(0)[rec].sum())
                a["err_in"] += float((n_in - n_tg).abs().mean(0)[rec].sum())
                a["rec_px"] += int(rec.sum())
                a["covered"] += int((rec & r["support"]).sum())
            a["support_px"] += int(r["support"].sum())
            a["offeye_px"] += int((r["support"] & (lab["ocular"] < 0.004)).sum())
            a["abstained_px"] += int(r["abstained"].sum())
            a["selected_px"] += int(r["selected"].sum())
            a["synth_px"] += int(r["synth"].sum())
            lump = 0.2722 * r["pred"][0] + 0.6741 * r["pred"][1] + 0.0537 * r["pred"][2]
            for s, eye in f["eyes"].items():
                cx, cy, D = _eye_center(eye)
                if not eye["in_frame"]:
                    continue
                S = max(16, int(3 * D))
                tf = CropTransform(cx, cy, 1.0, 0.0, S)
                if D >= 24 and eye["openness"] > 0.5:
                    ir = extract_crop(lab["iris"][None], tf, mode="nearest")[0]
                    pu = extract_crop(lab["pupil"][None], tf, mode="nearest")[0]
                    a_lab = float((pu > 0.5).sum())
                    a_pr = pupil_area(extract_crop(lump[None], tf, mode="nearest")[0], ir, pu)
                    if a_pr is not None and a_lab > 4:
                        a["pupil_rel"].append(abs(a_pr - a_lab) / a_lab)
                err = extract_crop(n_pr - n_tg, tf, mode="nearest")
                rr = extract_crop(rec[None].float(), tf, mode="nearest")[0] > 0.5
                pv = prev[m].get(s)
                if pv is not None and pv["part"] == f["cut_part"] and pv["open"] and eye["openness"] > 0.5 and pv["err"].shape == err.shape:
                    w = rr | pv["rec"]
                    if int(w.sum()) > 0:
                        a["temporal"].append(float((err - pv["err"]).abs().mean(0)[w].mean()))
                        a["temporal_px"] += int(w.sum())
                prev[m][s] = {"err": err, "rec": rr, "part": f["cut_part"], "open": eye["openness"] > 0.5}
    out = {"case": case, "frames": len(meta["frames"]), "negative_case": negative_case, "exposure_ref": e_case,
           "anatomy_evaluator_validation": eval_valid, "models": {}}
    for m, a in acc.items():
        red = 1 - a["err_pred"] / a["err_in"] if a["err_in"] > 0 else None
        out["models"][m] = {
            "masked_mae_pred": a["err_pred"] / a["rec_px"] if a["rec_px"] else None,
            "masked_mae_input": a["err_in"] / a["rec_px"] if a["rec_px"] else None,
            "masked_mae_reduction": red, "recoverable_px": a["rec_px"],
            "coverage": a["covered"] / a["rec_px"] if a["rec_px"] else None,
            "support_px": a["support_px"], "offeye_px": a["offeye_px"], "abstained_px": a["abstained_px"],
            "selected_px": a["selected_px"], "synth_px": a["synth_px"], "pupil_rel": a["pupil_rel"],
            "temporal_errors": a["temporal"], "temporal_mean": sum(a["temporal"]) / len(a["temporal"]) if a["temporal"] else None,
            "_err_pred": a["err_pred"], "_err_in": a["err_in"], "_covered": a["covered"]}
    return out


def detection_metrics(run_dir: Path, case: str, proc: Path) -> dict:
    meta = json.loads((run_dir / "eval_private" / case / "meta.json").read_text(encoding="utf-8"))
    tr = {f["frame"]: f for f in json.loads((proc / "tracks.json").read_text(encoding="utf-8"))["frames"]}
    hits = total = 0
    errs = []
    for f in meta["frames"]:
        det = tr.get(f["index"], {"eyes": []})["eyes"]
        for s, eye in f["eyes"].items():
            cx, cy, D = _eye_center(eye)
            if not eye["in_frame"] or eye["openness"] < 0.2 or D < 6:
                continue
            total += 1
            best = min((math.hypot(d["cx"] - cx, d["cy"] - cy) for d in det), default=1e9)
            if best <= max(3.0, 0.5 * D):
                hits += 1
                errs.append(best)
    return {"gt_eyes": total, "detected": hits, "recall": hits / total if total else None,
            "center_error_px_mean": sum(errs) / len(errs) if errs else None,
            "note": "detection ran on full frames with no render coordinates; GT read only here, after prediction"}


def _q(vals: list[float], q: float) -> float | None:
    if not vals:
        return None
    v = sorted(vals)
    return v[min(len(v) - 1, max(0, int(round(q * (len(v) - 1)))))]


def aggregate(cases: list[dict], dets: dict) -> dict:
    models = sorted({m for c in cases for m in c["models"]})
    agg = {}
    for m in models:
        ep = sum(c["models"][m]["_err_pred"] for c in cases if m in c["models"])
        ei = sum(c["models"][m]["_err_in"] for c in cases if m in c["models"])
        rp = sum(c["models"][m]["recoverable_px"] for c in cases if m in c["models"])
        cov = sum(c["models"][m]["_covered"] for c in cases if m in c["models"])
        pr = [x for c in cases if m in c["models"] for x in c["models"][m]["pupil_rel"]]
        te = [x for c in cases if m in c["models"] for x in c["models"][m]["temporal_errors"]]
        sel = sum(c["models"][m]["selected_px"] for c in cases if m in c["models"])
        agg[m] = {"masked_mae_reduction": (1 - ep / ei) if ei > 0 else None, "recoverable_px": rp,
                  "coverage": cov / rp if rp else None,
                  "negative_case_changed_px": sum(c["models"][m]["support_px"] for c in cases if c["negative_case"] and m in c["models"]),
                  "offeye_changed_px": sum(c["models"][m]["offeye_px"] for c in cases if m in c["models"]),
                  "pupil_rel_median": _q(pr, 0.5), "pupil_rel_p95": _q(pr, 0.95), "pupil_samples": len(pr),
                  "temporal_error_mean": sum(te) / len(te) if te else None, "temporal_samples": len(te),
                  "abstention_fraction": (sum(c["models"][m]["abstained_px"] for c in cases if m in c["models"]) / sel) if sel else None,
                  "synthesis_px": sum(c["models"][m]["synth_px"] for c in cases if m in c["models"])}
    ev = [x for c in cases for x in c["anatomy_evaluator_validation"]]
    ev_med, ev_p95 = _q(ev, 0.5), _q(ev, 0.95)
    # the pupil metric needs a separately validated evaluator: if the evaluator cannot measure the
    # clean targets within the gate tolerances, the metric is unavailable (null), never a pass/fail
    import yaml

    acc_q = yaml.safe_load((Path(__file__).resolve().parents[3] / "contracts" / "acceptance.yaml").read_text(encoding="utf-8"))["synthetic_quality"]
    ev_ok = ev_med is not None and ev_med <= acc_q["pupil_area_relative_error_median_max"] and ev_p95 <= acc_q["pupil_area_relative_error_p95_max"]
    for a in agg.values():
        a["pupil_metric_status"] = "valid" if ev_ok else (
            f"unavailable: the independent pupil evaluator failed its validation on the clean targets "
            f"(median {ev_med}, p95 {ev_p95} relative error vs render labels)")
        if not ev_ok:
            a["pupil_rel_median_unvalidated"], a["pupil_rel_p95_unvalidated"] = a["pupil_rel_median"], a["pupil_rel_p95"]
            a["pupil_rel_median"] = a["pupil_rel_p95"] = None
    out = {"models": agg, "anatomy_evaluator_validation": {"median": ev_med, "p95": ev_p95, "samples": len(ev), "passed": ev_ok},
           "detection": dets}
    if "temporal" in agg and "single" in agg and agg["temporal"]["temporal_error_mean"] and agg["single"]["temporal_error_mean"]:
        out["temporal_error_ratio_vs_single_frame"] = agg["temporal"]["temporal_error_mean"] / agg["single"]["temporal_error_mean"]
    else:
        out["temporal_error_ratio_vs_single_frame"] = None
    tot_gt = sum(d["gt_eyes"] for d in dets.values())
    tot_hit = sum(d["detected"] for d in dets.values())
    out["detection_recall"] = tot_hit / tot_gt if tot_gt else None
    return out


def evaluate_run(run_dir: Path, split: str = "test", models: tuple[str, ...] = ("temporal", "single")) -> dict:
    dev = torch.device("cuda")
    run_dir = Path(run_dir).resolve()
    res_cases = []
    dets = {}
    for cdir in sorted((run_dir / "examples").iterdir()):
        if not (cdir / "case.json").exists():
            continue
        case = cdir.name
        procs = {m: cdir / f"process_{m}" for m in models if (cdir / f"process_{m}" / "patches" / "index.json").exists()}
        if not procs:
            continue
        r = evaluate_case(run_dir, case, procs, dev)
        res_cases.append(r)
        if "temporal" in procs:
            dets[case] = detection_metrics(run_dir, case, procs["temporal"])
    summary = aggregate(res_cases, dets)
    for c in res_cases:
        for m in c["models"].values():
            for k in ("_err_pred", "_err_in", "_covered"):
                m.pop(k, None)
    out = {"split": split, "summary": summary, "cases": res_cases}
    (run_dir / "evaluation").mkdir(exist_ok=True)
    (run_dir / "evaluation" / "evaluation.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out

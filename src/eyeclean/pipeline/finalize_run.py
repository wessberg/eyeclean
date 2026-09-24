"""Gallery, RUN_REPORT.json, MODEL_CARD.md, DATASET_CARD.md and verify-run for a build."""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import torch

from ..validation.gallery import build_case_gallery, write_index
from ..validation.report import build_report, rel
from ..validation.verify_run import verify_run


def _load(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def finalize(cfg: dict, run_dir: Path, st: dict, best: Path) -> dict:
    dev = torch.device("cuda")
    S = st["stages"]
    res = lambda n: (S.get(n) or {}).get("result") or {}
    ok = lambda n: (S.get(n) or {}).get("status") == "passed"
    evaluation = _load(run_dir / "evaluation" / "evaluation.json") or {}
    # gallery (actual checkpoint outputs)
    gals = []
    for cdir in sorted((run_dir / "examples").iterdir()):
        if (cdir / "process_temporal" / "patches" / "index.json").exists() and (run_dir / "eval_private" / cdir.name / "meta.json").exists():
            gals.append(build_case_gallery(run_dir, cdir, dev))
    ck_sha = json.loads((best.with_suffix(".manifest.json")).read_text(encoding="utf-8"))["sha256"]
    ck_in_runs = all(_load(run_dir / "examples" / g["case"] / "process_temporal" / "source_manifest.json")["checkpoint"]["sha256"] == ck_sha for g in gals)
    ev: dict = {"warnings": []}
    ev["doctor_path"] = run_dir / "doctor.json"
    ev["doctor"] = _load(run_dir / "doctor.json") or {}
    ev["render_log"] = _load(run_dir / "manifests" / "render_log.json") or []
    ev["split_integrity"] = _load(run_dir / "manifests" / "split_integrity.json") or {}
    procs = []
    for cdir in sorted((run_dir / "examples").iterdir()):
        for sub in ("process_temporal", "process_single", "process_hybrid"):
            rp = cdir / sub / "RUN_REPORT.json"
            if rp.exists():
                procs.append({"dir": cdir / sub, "report": _load(rp)})
    ev["process_runs"] = procs
    ev["train_audits"] = {s: (_load(run_dir / "checkpoints" / s / "summary.json") or {}).get("gpu_audit") for s in ("guard", "image", "temporal")}
    ev["preservation_suite_path"] = run_dir / "validation" / "preservation_suite.json"
    ev["preservation_suite"] = _load(ev["preservation_suite_path"]) or {}
    ev["evaluation"] = evaluation
    own = res("own_footage")
    ev["own_footage"] = own
    ev["special_runs"] = {
        "strength_zero": res("special_strength_zero"), "empty_support": _empty_support(run_dir, evaluation),
        "no_eye": res("special_no_eye"), "alpha": res("special_alpha"), "long_stream": res("special_long_stream"),
        "resume": res("special_resume"), "canonical": res("special_canonical"), "blink_cut": res("special_blink_cut"),
        "timestamps": _timestamps(run_dir, procs, own),
    }
    ev["gallery"] = {"ok": bool(gals) and ck_in_runs,
                     "evidence": [str(run_dir / "examples" / "index.html")] +
                                 [str(run_dir / "examples" / g["case"] / "process_temporal" / "source_manifest.json") for g in gals[:4]]}
    rf = res("real_footage")
    ev["real_quality_status"] = rf.get("status", "not_tested")
    # training summary
    summaries = {s: _load(run_dir / "checkpoints" / s / "summary.json") or {} for s in ("guard", "image", "temporal")}
    led = _load(run_dir / "manifests" / "optimizer_steps.json") or {}
    steps_total = sum(sum(v) for v in led.values()) if led else None
    params = sum(summaries[s].get("parameters") or 0 for s in ("guard", "temporal")) or None
    peak = max((summaries[s].get("memory", {}).get("peak_device_used_gib") or 0) for s in summaries) or None
    ev["training"] = {"steps": steps_total, "checkpoint_path": rel(best, run_dir), "checkpoint_sha256": ck_sha, "parameter_count": params,
                      "peak_gpu_gib": peak, "data_manifest_path": rel(run_dir / "manifests" / "dataset_manifest.json", run_dir)}
    # stages
    stage_rows = []
    for name, rec in S.items():
        stage_rows.append({"name": name, "status": "passed" if rec.get("status") == "passed" else "failed",
                           "commands": [f"uv run eyeclean build-demo --config configs/{cfg['profile']}.yaml --resume auto"],
                           "evidence": [rel(run_dir / "manifests" / "build_state.json", run_dir)],
                           "notes": (rec.get("error") or f"{rec.get('seconds', 0)} s")[:400]})
    ev["stages"] = stage_rows
    # extra metrics: timings, memory, disk
    extra = {}
    from ..validation.report import metric

    tr_s = [p["report"] for p in procs if p["dir"].name == "process_temporal"]
    if tr_s:
        fr = sum(r["render"]["frames"] for r in tr_s)
        sec = sum(r["seconds"] for r in tr_s)
        extra["process_throughput_fps"] = metric(fr / sec if sec else None, "frames/s", "held-out 1080p demo sequences, both passes, conservative, incl. exports", [rel(p["dir"] / "RUN_REPORT.json", run_dir) for p in procs[:2]])
        extra["process_peak_vram_gib"] = metric(max(r["memory"]["peak_device_used_gib"] for r in tr_s), "GiB", "whole-device used memory measured by the driver during processing", [rel(p["dir"] / "RUN_REPORT.json", run_dir) for p in procs[:2]])
    from ..data.generate import dir_size_gib

    extra["workspace_disk_gib"] = metric(dir_size_gib(run_dir), "GiB", "measured size of the run directory", [rel(run_dir / "manifests" / "build_state.json", run_dir)], split="all")
    orc = _load(run_dir / "evaluation" / "oracle_crop_diagnostic.json")
    if orc:
        for bk, v in orc["buckets"].items():
            if v.get("masked_mae_reduction") is not None:
                extra[f"diagnostic_oracle_masked_mae_reduction_{bk}"] = metric(v["masked_mae_reduction"], "fraction",
                    "DIAGNOSTIC ONLY: generator (oracle) crops of test stills, single-frame network, no detection or gate",
                    [rel(run_dir / "evaluation" / "oracle_crop_diagnostic.json", run_dir)], bucket=bk)
    ev["extra_metrics"] = extra
    # artifacts
    arts = [(best, "checkpoint_bundle", "trained EyeGuard + five-frame EyeRestore + single-frame EyeRestore (EMA), this run"),
            (run_dir / "checkpoints" / "last.safetensors", "checkpoint_bundle_last", "last EMA weights per stage, this run"),
            (run_dir / "manifests" / "dataset_manifest.json", "dataset_manifest", "render cache manifest with per-shot SHA-256"),
            (run_dir / "manifests" / "dataset_plan.json", "dataset_plan", "deterministic plan (identities, scenes, splits)"),
            (run_dir / "manifests" / "split_integrity.json", "split_integrity", "split disjointness verification"),
            (run_dir / "manifests" / "acceptance_frozen.json", "acceptance_freeze", "acceptance hash frozen before training"),
            (run_dir / "evaluation" / "evaluation.json", "evaluation", "held-out full-frame evaluation"),
            (run_dir / "examples" / "index.html", "gallery", "held-out example gallery"),
            (run_dir / "validation" / "pair" / "pair_validation_report.json", "pair_validation", "paired-target construction verification"),
            (run_dir / "validation" / "preservation_suite.json", "preservation_suite", "4K compositor fixture and preservation tests"),
            (run_dir / "doctor.json", "doctor", "workstation inspection and smoke tests"),
            (run_dir / "manifests" / "dependencies.json", "dependency_manifest", "exact versions, uv.lock hash, asset hashes/licenses"),
            (run_dir / "manifests" / "licenses.json", "license_manifest", "third-party license statements")]
    if (run_dir / "manifests" / "discarded_runs.json").exists():
        arts.append((run_dir / "manifests" / "discarded_runs.json", "discarded_runs", "training runs discarded after a bug fix (steps still counted)"))
    for p in sorted((run_dir / "validation" / "inspection").glob("*.jpg")) if (run_dir / "validation" / "inspection").exists() else []:
        arts.append((p, "inspection_sheet", "human-inspection contact sheet of rendered pairs and labels"))
    for p in sorted((run_dir / "scenes").glob("*.blend")) if (run_dir / "scenes").exists() else []:
        arts.append((p, "blend_scene", "representative generator scene saved by Blender"))
    for p in sorted((run_dir / "profiles").glob("*.trace.json")) if (run_dir / "profiles").exists() else []:
        arts.append((p, "profiler_trace", "torch.profiler CPU+CUDA trace"))
    ev["artifacts"] = arts
    if not ev["doctor"]:
        ev["warnings"].append("doctor report missing")
    report = build_report(run_dir, cfg, ev)
    write_index(run_dir, gals, evaluation, report)
    # the gallery page is an artifact: refresh its hash after writing
    report = build_report(run_dir, cfg, ev)
    (run_dir / "RUN_REPORT.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    write_cards(cfg, run_dir, report, summaries, evaluation, best)
    ver = verify_run(run_dir)
    (run_dir / "verify_run.json").write_text(json.dumps(ver, indent=1), encoding="utf-8")
    return {"engineering_status": report["engineering_status"], "synthetic_quality_status": report["synthetic_quality_status"],
            "real_quality_status": report["real_quality_status"], "verify_run_valid": ver["valid"], "verify_problems": ver["problems"][:20],
            "failures": report["failures"], "gallery": str(run_dir / "examples" / "index.html")}


def _empty_support(run_dir: Path, evaluation: dict) -> dict:
    """Compositor property (spec check zero_changed_bits_empty_support): whenever the final support is
    empty the output is the reference bit-for-bit.  Evidence: held-out frames with tracked eyes whose
    support was empty (each verified on the GPU by the pipeline) and the 4K preservation fixture's
    explicit empty-support test.  Whether the model *should* have produced an empty support (e.g. the
    no-reflection case) is a quality question, scored by controlled_no_reflection_changed_pixels and
    reported here only for information.  (Until 2026-09-11 this check also required zero support on
    the negative case, double-counting that quality gate.)"""
    ev, frames = [], 0
    neg_support = None
    for c in evaluation.get("cases", []):
        proc = run_dir / "examples" / c["case"] / "process_temporal"
        idx = proc / "patches" / "index.json"
        if not idx.exists():
            continue
        n0 = frames
        for f in json.loads(idx.read_text(encoding="utf-8"))["frames"]:
            if f["eyes"] and f["bypass"] == "empty_support" and f["assigned_px"] == 0:
                frames += 1
        if c.get("negative_case"):
            neg_support = c["models"].get("temporal", {}).get("support_px")
        if frames > n0 and len(ev) < 3:
            ev.append(str(idx))
    sp = run_dir / "validation" / "preservation_suite.json"
    suite = json.loads(sp.read_text(encoding="utf-8")) if sp.exists() else {}
    suite_ok = suite.get("empty_support_changed_bits") == 0
    ok = frames > 0 and suite_ok
    return {"ok": ok, "empty_support_frames_with_eyes_verified": frames, "fixture_empty_support_changed_bits": suite.get("empty_support_changed_bits"),
            "negative_case_support_px_info": neg_support, "evidence": ev + ([str(sp)] if sp.exists() else [])}


def _timestamps(run_dir: Path, procs: list, own: dict) -> dict:
    ok = True
    ev = []
    for p in procs:
        if p["dir"].name != "process_temporal":
            continue
        man = _load(p["dir"] / "source_manifest.json")
        prev = p["report"].get("previews", {}).get("preview", {})
        if not man or not prev:
            continue
        ok &= prev.get("pts") == man["pts"] and man["frame_count"] == prev.get("frames")
        ev.append(str(p["dir"] / "source_manifest.json"))
    ok &= bool(own.get("vfr_ok"))
    return {"ok": ok and bool(ev), "evidence": ev[:4] + [e for e in own.get("evidence", []) if "out_vfr" in e]}


def write_cards(cfg: dict, run_dir: Path, report: dict, summaries: dict, evaluation: dict, best: Path) -> None:
    bman = json.loads(best.with_suffix(".manifest.json").read_text(encoding="utf-8"))["manifest"]
    m = report["metrics"]
    fmt = lambda k: "null" if m.get(k, {}).get("value") is None else f"{m[k]['value']:.4g}"
    model_card = f"""# EyeClean model card ({cfg['profile']} profile)

Generated {report['generated_at']} from measured run data. **Research prototype.** Engineering status:
`{report['engineering_status']}`; synthetic-quality status: `{report['synthetic_quality_status']}`; real-footage status:
`{report['real_quality_status']}`. Synthetic results do not establish real-footage quality.

## Components (checkpoint `{rel(best, run_dir)}`, SHA-256 `{report['training']['checkpoint_sha256']}`)
| Component | Role | Parameters |
|---|---|---|
| EyeGuard | CUDA U-Net: ocular surface, iris, pupil, protected lids/lashes/tear, reflection, clipped; eye-present score | {summaries.get('guard', {}).get('parameters')} |
| EyeRestore (five-frame) | NAF-style residual encoder-decoder, widths {cfg['model']['widths']}, blocks {cfg['model']['encoder_blocks']}, 512-wide bottleneck, zero-initialised residual head, local SDPA temporal fusion at 1/4 and 1/8 resolution | {summaries.get('temporal', {}).get('parameters')} |
| EyeRestore (single-frame) | same network, image stage only; kept for the temporal comparison | {summaries.get('image', {}).get('parameters')} |
| Face detector / landmarks | S3FD + 2DFAN4 (face-alignment, BSD-3 code; weights have no explicit license, local research use only, converted to safetensors once with `weights_only=True`) | public |
| Canonical iris | per-shot, per-track frozen polar atlas: confidence-weighted visible observations + fixed-seed CUDA procedural prior with fitted colour gains | no training |

## Training (actual runs)
Optimizer AdamW (lr {cfg['training']['learning_rate']}, wd {cfg['training']['weight_decay']}), warmup {cfg['training']['warmup_steps']}, cosine decay,
grad clip {cfg['training']['gradient_clip']}, EMA {cfg['training']['ema_decay']}, BF16 network / FP32 losses. Steps actually executed (all runs, incl.
repairs): {report['training']['optimizer_steps']}. Stage summaries:
```json
{json.dumps({s: {k: v for k, v in (summaries.get(s) or {}).items() if k in ('steps', 'best', 'train_seconds', 'adaptations', 'sample_mix', 'memory')} for s in summaries}, indent=1, default=str)[:6000]}
```
Discarded runs (their steps are included in the total above): {json.dumps((_load(run_dir / 'manifests' / 'discarded_runs.json') or {}).get('runs', []), indent=1)}

## Held-out evaluation (test identities, full-frame detection, conservative mode)
| Metric | Value | Frozen threshold |
|---|---|---|
| recoverable masked MAE reduction (five-frame) | {fmt('recoverable_masked_mae_reduction')} | >= 0.20 |
| same, single-frame model | {fmt('recoverable_masked_mae_reduction_single_frame')} | — |
| recoverable reflection coverage | {fmt('recoverable_reflection_coverage')} | >= 0.30 |
| changed pixels, no-reflection case | {fmt('controlled_no_reflection_changed_pixels')} | 0 |
| changed off-eye pixels | {fmt('controlled_off_eye_changed_pixels')} | 0 |
| pupil area rel. error median / p95 | {fmt('pupil_area_relative_error_median')} / {fmt('pupil_area_relative_error_p95')} | 0.05 / 0.15 |
| temporal error ratio vs single-frame | {fmt('temporal_error_ratio_vs_single_frame')} | <= 0.95 |
| detection recall (independent full-frame) | {fmt('detection_recall')} | engineering check >= 0.8 |
| abstention fraction | {fmt('abstention_fraction')} | reported |

Failed gates and checks (kept visible): {json.dumps(report['failures'], indent=1)}

## Confidence calibration
{json.dumps(bman.get('calibration', {}), indent=1)}
The exported uncertainty layer is `1 - raw confidence`, **not** a calibrated probability; use the table above.

## Intended use and limits
Removes the designated camera-visible ocular (tear-film/cornea/sclera) reflection under unchanged remaining illumination.
Conservative mode abstains on saturated/low-confidence regions; hybrid mode (`--allow-synthesis`) fills them from a
per-shot canonical iris representation and labels them as synthesized. It cannot recover a true hidden iris. Glasses,
contact lenses, extreme gazes and eyes below {cfg['inference']['minimum_iris_diameter_pixels']} px are passed through or unsupported.
Trained only on procedural CG (MPFB/MakeHuman heads, project eyes); expect a domain gap on real footage.
"""
    (run_dir / "MODEL_CARD.md").write_text(model_card, encoding="utf-8")
    man = json.loads((run_dir / "manifests" / "dataset_manifest.json").read_text(encoding="utf-8"))
    plan = json.loads((run_dir / "manifests" / "dataset_plan.json").read_text(encoding="utf-8"))
    si = json.loads((run_dir / "manifests" / "split_integrity.json").read_text(encoding="utf-8"))
    fam = Counter((x["split"], x["iris_family"]) for x in man["shots"])
    lights = Counter((x["split"], lf) for x in man["shots"] for lf in x["light_families"])
    blink = sum(1 for x in man["shots"] if x.get("has_blink"))
    dataset_card = f"""# EyeClean dataset card ({cfg['profile']} profile)

Procedurally generated with Blender {report['versions'].get('blender', '5.2.1')} / Cycles on an OptiX RTX 5080, MPFB 2.0.17 (GPL-3.0-or-later code)
with MakeHuman CC0 system assets and CC0 skins01/skins02, and project-owned optical eyes. No real people, no paid or gated
assets, no manual modelling/painting/annotation, no clean donor footage.

* Shots rendered: {len(man['shots'])} (train {man['counts']['train']}, validation {man['counts']['validation']}, test {man['counts']['test']}); plan digest `{plan['digest']}`.
* Renders: {cfg['render']['width']}x{cfg['render']['height']} stills/clips at {cfg['render']['samples']} spp (adaptive, no denoising); demo cases {cfg['data']['demo_width']}x{cfg['data']['demo_height']}.
* Paired target: per frame, one multipart EXR with `clean` (camera-visible designated ocular reflection closure replaced by a zero-radiance branch; (1-F) transmission and refraction kept), `reflection` (only F*glossy on camera-visible cornea/sclera), and emission label layers (ocular surface, skin, lashes/brows; iris and pupil through the refractive cornea; tear meniscus). Source = clean + reflection. Verified in `validation/pair/pair_validation_report.json`.
* Negatives (no-reflection identity cases): {man['negative_fraction_train']:.1%} of training shots. Saturation arises naturally from bright small sources and from forced-exposure augmentation (tracked per training stage in `sample_mix`).
* Blinks/lids: {blink} clips contain natural blink events (fast close, hold, slower reopen, some half-blinks); stills sample open (70%), partly closed (15%), nearly closed (8%) and closed (7%) lids.
* Split integrity ({'passed' if si['passed'] else 'FAILED'}): identities, iris/sclera texture seeds, skins, scene seeds and whole clips are disjoint; held-out iris structure family `spoke` and light family `ring_light` appear only in test. Counts: {json.dumps(si['counts'])}.
* Iris families by split: {json.dumps({f'{k[0]}/{k[1]}': v for k, v in sorted(fam.items())})}
* Light families by split: {json.dumps({f'{k[0]}/{k[1]}': v for k, v in sorted(lights.items())})}
* Camera simulation (GPU, paired, same noise realization): defocus/motion, exposure, white balance, shot/read noise, clipping, BT.709/sRGB/linear encoding, 8/10-bit quantisation, 4:2:0/4:2:2 chroma, sharpening; native iris sizes down to 8 px.
* Limitations: synthetic skin/eyes/lighting; MakeHuman lid geometry intrudes into the eyeball and is made transparent to non-camera rays inside the eye; cornea shadow rays use (1-F) transparency; no glasses/contacts/makeup-specific categories.
"""
    (run_dir / "DATASET_CARD.md").write_text(dataset_card, encoding="utf-8")

"""Build RUN_REPORT.json (contracts/run_report.schema.json) from evidence on disk.

Statuses are derived, never asserted: engineering_status is 'passed' only if every mandatory check
passed with evidence; synthetic_quality_status compares measured metrics with the FROZEN
acceptance thresholds (unavailable metrics are null and fail their gate); smoke runs can never
pass quality; real_quality_status is 'not_tested' unless local footage was actually processed.
"""

from __future__ import annotations

import hashlib
import json
import platform
import time
from pathlib import Path

from .acceptance import acceptance_sha256, load_acceptance

ROOT = Path(__file__).resolve().parents[3]
SPEC_REVISION = "EyeReflectionRemoval_SPEC.md revision 1, 10 September 2026"


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def rel(p: Path | str, run_dir: Path) -> str:
    p = Path(p).resolve()
    try:
        return str(p.relative_to(run_dir)).replace("\\", "/")
    except ValueError:
        try:
            return str(p.relative_to(ROOT)).replace("\\", "/")
        except ValueError:
            return str(p)


def check(name: str, passed: bool | None, evidence: list, reason: str) -> dict:
    status = "passed" if passed else ("not_run" if passed is None else "failed")
    ev = [e for e in evidence if e]
    if status == "passed" and not ev:
        status, reason = "failed", reason + " (no evidence file)"
    return {"name": name, "status": status, "evidence": ev, "reason": reason}


def metric(value, unit: str, basis: str, evidence: list, split: str = "test", bucket: str = "all", reason: str | None = None) -> dict:
    if value is None:
        return {"value": None, "unit": unit, "split": split, "bucket": bucket, "basis": basis,
                "reason_unavailable": reason or "not measured", "evidence": evidence}
    return {"value": float(value), "unit": unit, "split": split, "bucket": bucket, "basis": basis, "reason_unavailable": None,
            "evidence": evidence or ["(missing)"]}


def quality_gates(summary: dict, acc: dict, profile: str) -> tuple[str, list[str], dict]:
    q = acc["synthetic_quality"]
    t = summary.get("models", {}).get("temporal", {})
    gates = {
        "recoverable_masked_mae_reduction": (t.get("masked_mae_reduction"), ">=", q["recoverable_masked_mae_reduction_min"]),
        "recoverable_reflection_coverage": (t.get("coverage"), ">=", q["recoverable_reflection_coverage_min"]),
        "controlled_no_reflection_changed_pixels": (t.get("negative_case_changed_px"), "<=", q["controlled_no_reflection_changed_pixels_max"]),
        "controlled_off_eye_changed_pixels": (t.get("offeye_changed_px"), "<=", q["controlled_off_eye_changed_pixels_max"]),
        "pupil_area_relative_error_median": (t.get("pupil_rel_median"), "<=", q["pupil_area_relative_error_median_max"]),
        "pupil_area_relative_error_p95": (t.get("pupil_rel_p95"), "<=", q["pupil_area_relative_error_p95_max"]),
        "temporal_error_ratio_vs_single_frame": (summary.get("temporal_error_ratio_vs_single_frame"), "<=", q["temporal_error_ratio_vs_single_frame_max"]),
    }
    failed = []
    res = {}
    for k, (v, op, thr) in gates.items():
        ok = v is not None and (v >= thr if op == ">=" else v <= thr)
        res[k] = {"value": v, "threshold": thr, "op": op, "passed": ok}
        if not ok:
            failed.append(f"quality gate {k}: {v} {op} {thr} not met" if v is not None else f"quality gate {k}: unavailable (null) -> fails")
    if profile == "smoke":
        return "not_evaluated" if not summary else "failed", failed + ["smoke profile cannot satisfy the delivered quality gate"], res
    return ("passed" if not failed else "failed"), failed, res


def build_report(run_dir: Path, cfg: dict, ev: dict) -> dict:
    """ev: evidence dictionary assembled by build_demo (paths + parsed results)."""
    acc = load_acceptance()
    frozen = json.loads((run_dir / "manifests" / "acceptance_frozen.json").read_text(encoding="utf-8"))
    doctor = ev.get("doctor", {})
    dchk = doctor.get("checks", {})
    dpath = [rel(ev["doctor_path"], run_dir)] if ev.get("doctor_path") else []
    procs = ev.get("process_runs", [])
    proc_reports = [p["report"] for p in procs if p.get("report")]
    proc_paths = [rel(p["dir"] / "RUN_REPORT.json", run_dir) for p in procs if p.get("report")]
    suite = ev.get("preservation_suite", {})
    suite_path = [rel(ev["preservation_suite_path"], run_dir)] if ev.get("preservation_suite_path") else []
    evaluation = ev.get("evaluation", {})
    summ = evaluation.get("summary", {})
    eval_path = [rel(run_dir / "evaluation" / "evaluation.json", run_dir)] if evaluation else []
    special = ev.get("special_runs", {})
    checks = []
    # the frozen acceptance contract names this check for the build-demo target hardware (compute capability >= 10);
    # doctor's own kernel check accepts any architecture in the torch build
    kern = dchk.get("cuda_kernel", dchk.get("cuda_blackwell_kernel", {}))
    checks.append(check("cuda_blackwell_kernel", bool(kern.get("passed")) and bool(kern.get("blackwell")) if kern else None, dpath,
                        f"sm_{doctor.get('gpu', {}).get('capability', ['?', '?'])[0]}{doctor.get('gpu', {}).get('capability', ['?', '?'])[1]} kernels executed and verified"))
    rlog = ev.get("render_log", [])
    optix_all = bool(rlog) and all("RTX" in str(x.get("optix_enabled", "")) for x in rlog)
    checks.append(check("optix_render_device", bool(dchk.get("optix_render_device", {}).get("passed")) and optix_all,
                        dpath + ([rel(run_dir / "manifests" / "render_log.json", run_dir)] if rlog else []),
                        f"doctor OptiX render + {len(rlog)} dataset render jobs all on the OptiX device (CPU disabled)"))
    audits_ok = bool(proc_reports) and all(r["gpu_audit"]["cpu_numerical_image_fallbacks"] == 0 and r["gpu_audit"]["undeclared_transfers"] == 0
                                            for r in proc_reports)
    static_ok = all(r.get("static_scan", {}).get("passed") for r in proc_reports) if proc_reports else False
    train_audits = ev.get("train_audits", {})
    train_ok = bool(train_audits) and all(a and a["cpu_numerical_image_fallbacks"] == 0 and a["undeclared_transfers"] == 0 for a in train_audits.values())
    checks.append(check("strict_gpu_hot_path_audit", audits_ok and static_ok and train_ok, proc_paths[:6] + [rel(run_dir / "checkpoints" / s / "summary.json", run_dir) for s in train_audits],
                        f"dispatch-mode audit: 0 CPU numerical ops / 0 undeclared transfers in {len(proc_reports)} process runs and "
                        f"{len(train_audits)} training stages; static scan of hot-path modules found no NumPy/OpenCV/PIL image imports"))
    codec = dchk.get("nvdec_nvenc_supported_profile_roundtrip", {})
    own = ev.get("own_footage", {})
    checks.append(check("nvdec_nvenc_supported_profile_roundtrip", bool(codec.get("passed")) and bool(own.get("mp4_ok")),
                        dpath + own.get("evidence", []), "H.264 8-bit and HEVC Main10 NVENC->MP4->NVDEC round trips with preserved "
                        "timestamps; process runs on NVENC-encoded clips"))
    checks.append(check("cuda_color_reference_tests", dchk.get("cuda_color_reference_tests", {}).get("passed"), dpath + [rel(ROOT / "tests" / "unit" / "test_color_reference.py", run_dir)],
                        "CUDA BT.709/sRGB/V-Log transfers and Bradford primaries matrices vs colour-science (offline)"))
    checks.append(check("float_exr_roundtrip", dchk.get("float_exr_roundtrip", {}).get("passed"), dpath + [rel(ROOT / "tests" / "unit" / "test_exr.py", run_dir)],
                        "bit-exact FLOAT round trip incl. -0/inf/NaN and cropped data windows; official OpenEXR reads identically"))
    mods = [r["preservation"]["source_files_modified"] for r in proc_reports]
    checks.append(check("original_source_hash_unchanged", bool(mods) and all(m == 0 for m in mods), proc_paths[:6],
                        f"SHA-256 before/after identical for {len(mods)} process runs"))
    frames_ok = bool(proc_reports) and all(r["render"]["frames"] > 0 for r in proc_reports)
    checks.append(check("zero_changed_bits_outside_final_support", suite.get("outside_support_changed_bits") == 0 and frames_ok, suite_path + proc_paths[:4],
                        "4K fixture + every processed frame checked by IEEE-754 bit patterns outside the final support"))
    sz = special.get("strength_zero", {})
    checks.append(check("zero_changed_bits_strength_zero", suite.get("strength_zero_changed_bits") == 0 and sz.get("ok"), suite_path + sz.get("evidence", []),
                        "strength 0 bypasses arithmetic: bit-exact copies (fixture and a full process run)"))
    neg = special.get("empty_support", {})
    checks.append(check("zero_changed_bits_empty_support", suite.get("empty_support_changed_bits") == 0 and neg.get("ok"), suite_path + neg.get("evidence", []),
                        f"empty support returns the reference exactly: {neg.get('empty_support_frames_with_eyes_verified')} held-out frames with tracked "
                        f"eyes and empty support verified bit-exact in the pipeline + 4K fixture test (whether the model should have "
                        f"edited is scored by the quality gates; no-reflection case support px: {neg.get('negative_case_support_px_info')})"))
    ne = special.get("no_eye", {})
    checks.append(check("no_eye_passthrough", suite.get("no_eye_changed_bits") == 0 and ne.get("ok"), suite_path + ne.get("evidence", []),
                        "frames without detected eyes are passed through unchanged"))
    al = special.get("alpha", {})
    checks.append(check("source_alpha_unchanged", suite.get("alpha_changed_bits") == 0 and al.get("ok"), suite_path + al.get("evidence", []),
                        "alpha carried bit-identically (fixture + EXR input with alpha through process)"))
    g = suite.get("gate_after_inverse_warp", {})
    checks.append(check("final_gate_after_inverse_warp", g.get("outside_changed_bits") == 0 and g.get("support_within_hard_gate"), suite_path,
                        "crop-space delta non-zero everywhere; after inverse warp only the full-resolution hard gate changes"))
    rec = summ.get("detection_recall")
    checks.append(check("independent_full_frame_detection", rec is not None and rec >= 0.8, eval_path,
                        f"S3FD+FAN on full frames without render coordinates; recall {rec} vs generator eye positions (threshold 0.8)"))
    fa = [r["file_access"] for r in proc_reports]
    fa_ok = bool(fa) and all(f["forbidden_roots"] and not f["violations"] for f in fa)
    checks.append(check("no_clean_target_access_during_prediction", fa_ok, proc_paths[:4] + [rel(p["dir"] / "file_access.log", run_dir) for p in procs[:4]],
                        "prediction ran in separate processes with an audit hook denying eval_private/ (targets); all opens logged, 0 violations"))
    can = special.get("canonical", {})
    checks.append(check("fixed_per_shot_canonical_state", can.get("ok"), can.get("evidence", []), can.get("reason", "hybrid run not executed")))
    bc = special.get("blink_cut", {})
    checks.append(check("blink_cut_identity_resets", bc.get("ok"), bc.get("evidence", []), bc.get("reason", "not evaluated")))
    lm = special.get("long_stream", {})
    checks.append(check("bounded_memory_long_stream", bool(suite.get("bounded_memory", {}).get("bounded")) and lm.get("ok"), suite_path + lm.get("evidence", []),
                        lm.get("reason", "") + " ; fixture: flat peak allocator memory over 120 4K frames"))
    ts = special.get("timestamps", {})
    checks.append(check("timestamp_and_frame_count_preservation", bool(codec.get("passed")) and ts.get("ok"), dpath + ts.get("evidence", []),
                        "VFR timestamps preserved through NVENC/NVDEC; process previews carry the source PTS and frame counts"))
    si = ev.get("split_integrity", {})
    rs = special.get("resume", {})
    checks.append(check("reproducible_resume_and_disjoint_splits", bool(si.get("passed")) and rs.get("ok"),
                        [rel(run_dir / "manifests" / "split_integrity.json", run_dir)] + rs.get("evidence", []),
                        "identities/texture seeds/skins/scene seeds/clips disjoint across splits; interrupted training resumed from saved state"))
    gal = ev.get("gallery", {})
    checks.append(check("actual_trained_checkpoint_gallery", bool(gal.get("ok")), gal.get("evidence", []),
                        "gallery frames are the trained bundle's exported corrections applied to held-out inputs (checkpoint hash in each run)"))
    checks.append(check("tested_own_footage_cli", bool(own.get("ok")), own.get("evidence", []), own.get("reason", "")))
    eng_ok = all(c["status"] == "passed" for c in checks)
    names = [c["name"] for c in checks]
    missing = [n for n in acc["engineering"]["mandatory_pass_checks"] if n not in names]
    if missing:
        eng_ok = False
    sq_status, q_fail, gates = quality_gates(summ, acc, cfg["profile"]) if summ else ("not_evaluated", ["no evaluation"], {})
    t = summ.get("models", {}).get("temporal", {})
    s1 = summ.get("models", {}).get("single", {})
    metrics = {
        "recoverable_masked_mae_reduction": metric(t.get("masked_mae_reduction"), "fraction", "1 - sum|pred-target| / sum|input-target| over target-differs & not-clipped pixels, asinh-normalised working space, 12 held-out full-frame demo cases, conservative mode, five-frame model", eval_path),
        "recoverable_masked_mae_reduction_single_frame": metric(s1.get("masked_mae_reduction"), "fraction", "same, trained single-frame model", eval_path),
        "recoverable_reflection_coverage": metric(t.get("coverage"), "fraction", "recoverable target-differs pixels inside the final hard support", eval_path),
        "controlled_no_reflection_changed_pixels": metric(t.get("negative_case_changed_px"), "pixels", "support pixels on the controlled no-reflection case", eval_path, bucket="no_reflection_negative"),
        "controlled_off_eye_changed_pixels": metric(t.get("offeye_changed_px"), "pixels", "support pixels with ground-truth ocular coverage == 0, all cases", eval_path),
        "pupil_area_relative_error_median": metric(t.get("pupil_rel_median"), "fraction", "independent dark-disk evaluator on predictions vs render pupil labels, eyes with iris >= 24 px", eval_path, bucket="iris>=24px"),
        "pupil_area_relative_error_p95": metric(t.get("pupil_rel_p95"), "fraction", "as above, 95th percentile", eval_path, bucket="iris>=24px"),
        "anatomy_evaluator_validation_median": metric(summ.get("anatomy_evaluator_validation", {}).get("median"), "fraction", "evaluator on targets vs render labels (validation before use)", eval_path, bucket="iris>=24px"),
        "temporal_error_ratio_vs_single_frame": metric(summ.get("temporal_error_ratio_vs_single_frame"), "ratio", "mean |e_t - e_(t-1)| of eye-aligned prediction errors (five-frame / single-frame)", eval_path),
        "detection_recall": metric(summ.get("detection_recall"), "fraction", "independent full-frame detection vs generator eye centres", eval_path),
        "abstention_fraction": metric(t.get("abstention_fraction"), "fraction", "abstained / selected reflection pixels (conservative)", eval_path),
    }
    for k, v in ev.get("extra_metrics", {}).items():
        metrics[k] = v
    tr = ev.get("training", {})
    pres = {"reference_definition": "source frame decoded by the recorded decoder and converted once by the recorded GPU transform to FP32 working space (linear ACEScg by default); alpha carried separately",
            "source_files_modified": sum(mods) if mods else None,
            "outside_support_changed_bits": (suite.get("outside_support_changed_bits") if suite else None) if frames_ok else None,
            "strength_zero_changed_bits": suite.get("strength_zero_changed_bits") if sz.get("ok") else None,
            "empty_support_changed_bits": suite.get("empty_support_changed_bits") if neg.get("ok") else None,
            "alpha_changed_bits": suite.get("alpha_changed_bits") if al.get("ok") else None,
            "evidence": suite_path + proc_paths[:6]}
    arts = []
    for p, kind, prov in ev.get("artifacts", []):
        p = Path(p)
        if p.exists() and p.is_file():
            arts.append({"path": rel(p, run_dir), "kind": kind, "sha256": sha(p), "actually_created": True, "provenance": prov})
    real_status = ev.get("real_quality_status", "not_tested")
    warnings = list(ev.get("warnings", []))
    failures = [f"engineering check {c['name']}: {c['status']} - {c['reason']}" for c in checks if c["status"] != "passed"] + q_fail
    report = {
        "schema_version": "1.0", "report_kind": "implementation_run", "run_id": f"{cfg['profile']}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}",
        "profile": cfg["profile"], "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "spec_revision": SPEC_REVISION,
        "acceptance_sha256": acceptance_sha256(), "engineering_status": "passed" if eng_ok else "failed",
        "synthetic_quality_status": sq_status, "real_quality_status": real_status,
        "hardware": {"os": platform.platform(), "gpu_name": doctor.get("gpu", {}).get("name"),
                     "gpu_total_gib": round(doctor.get("gpu", {}).get("total_gib", 0.0), 3) if doctor else None,
                     "driver_version": doctor.get("gpu", {}).get("driver_version"), "cuda_available": bool(doctor)},
        "versions": {k: (str(v) if v is not None else None) for k, v in doctor.get("versions", {}).items()},
        "stages": ev.get("stages", []), "checks": checks, "metrics": metrics, "preservation": pres,
        "training": {"actually_executed": bool(tr.get("steps")), "optimizer_steps": tr.get("steps"), "checkpoint_path": tr.get("checkpoint_path"),
                     "checkpoint_sha256": tr.get("checkpoint_sha256"), "parameter_count": tr.get("parameter_count"),
                     "peak_gpu_gib": tr.get("peak_gpu_gib"), "data_manifest_path": tr.get("data_manifest_path"),
                     "split_integrity_status": "passed" if si.get("passed") else ("failed" if si else "not_run")},
        "artifacts": arts, "warnings": warnings, "failures": failures, "own_footage_cli_tested": bool(own.get("ok")),
    }
    report["versions"]["acceptance_frozen_sha256"] = frozen["sha256"]
    report["versions"]["quality_gates"] = json.dumps(gates)
    return report

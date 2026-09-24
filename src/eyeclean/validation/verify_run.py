"""`eyeclean verify-run`: schema + evidence + hash + frozen-threshold verification of RUN_REPORT.json.

A correctly shaped JSON object is not proof: every required engineering check must be present
by name with its evidence files on disk; artifact and checkpoint hashes are recomputed; the
acceptance file must match both the report and the value frozen before training; the quality
status must agree with the metrics and the frozen thresholds; smoke runs cannot pass quality;
null metrics need reasons and cannot pass.
"""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema

from .acceptance import SCHEMA, acceptance_sha256, load_acceptance
from .report import ROOT, sha

QUALITY_METRICS = {
    "recoverable_masked_mae_reduction": ("recoverable_masked_mae_reduction_min", ">="),
    "recoverable_reflection_coverage": ("recoverable_reflection_coverage_min", ">="),
    "controlled_no_reflection_changed_pixels": ("controlled_no_reflection_changed_pixels_max", "<="),
    "controlled_off_eye_changed_pixels": ("controlled_off_eye_changed_pixels_max", "<="),
    "pupil_area_relative_error_median": ("pupil_area_relative_error_median_max", "<="),
    "pupil_area_relative_error_p95": ("pupil_area_relative_error_p95_max", "<="),
    "temporal_error_ratio_vs_single_frame": ("temporal_error_ratio_vs_single_frame_max", "<="),
}


def _resolve(p: str, run_dir: Path) -> Path:
    q = Path(p)
    if q.is_absolute():
        return q
    for base in (run_dir, ROOT):
        if (base / q).exists():
            return base / q
    return run_dir / q


def verify_run(run_dir: Path) -> dict:
    run_dir = Path(run_dir).resolve()
    problems, notes = [], []
    rp = run_dir / "RUN_REPORT.json"
    if not rp.exists():
        return {"valid": False, "problems": [f"{rp} not found"]}
    rep = json.loads(rp.read_text(encoding="utf-8"))
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    v = jsonschema.Draft202012Validator(schema)
    for err in sorted(v.iter_errors(rep), key=lambda e: list(e.path)):
        problems.append(f"schema: {'/'.join(map(str, err.path))}: {err.message[:200]}")
    acc = load_acceptance()
    cur = acceptance_sha256()
    if rep.get("acceptance_sha256") != cur:
        problems.append(f"acceptance hash in report {rep.get('acceptance_sha256')} != current {cur}")
    fz = run_dir / "manifests" / "acceptance_frozen.json"
    if not fz.exists():
        problems.append("acceptance was never frozen before training (manifests/acceptance_frozen.json missing)")
    elif json.loads(fz.read_text(encoding="utf-8"))["sha256"] != cur:
        problems.append("contracts/acceptance.yaml changed after it was frozen")
    names = {c["name"]: c for c in rep.get("checks", [])}
    for n in acc["engineering"]["mandatory_pass_checks"]:
        c = names.get(n)
        if c is None:
            problems.append(f"mandatory check {n} missing")
            continue
        if c["status"] == "passed":
            for e in c["evidence"]:
                if not _resolve(e, run_dir).exists():
                    problems.append(f"check {n}: evidence {e} not found on disk")
    eng_should = all(names.get(n, {}).get("status") == "passed" for n in acc["engineering"]["mandatory_pass_checks"])
    if rep.get("engineering_status") == "passed" and not eng_should:
        problems.append("engineering_status=passed although a mandatory check did not pass")
    for a in rep.get("artifacts", []):
        p = _resolve(a["path"], run_dir)
        if not p.exists():
            problems.append(f"artifact {a['path']} missing")
        elif sha(p) != a["sha256"]:
            problems.append(f"artifact {a['path']} hash mismatch")
    tr = rep.get("training", {})
    if tr.get("checkpoint_path"):
        cp = _resolve(tr["checkpoint_path"], run_dir)
        if not cp.exists():
            problems.append(f"checkpoint {cp} missing")
        elif tr.get("checkpoint_sha256") and sha(cp) != tr["checkpoint_sha256"]:
            problems.append("checkpoint hash mismatch")
    q = acc["synthetic_quality"]
    m = rep.get("metrics", {})
    all_pass = True
    for k, (thr_key, op) in QUALITY_METRICS.items():
        mv = m.get(k)
        if mv is None:
            all_pass = False
            problems.append(f"quality metric {k} absent")
            continue
        if mv["value"] is None:
            all_pass = False
            if not mv.get("reason_unavailable"):
                problems.append(f"metric {k} is null without a reason")
            continue
        thr = q[thr_key]
        ok = mv["value"] >= thr if op == ">=" else mv["value"] <= thr
        all_pass &= ok
        for e in mv.get("evidence", []):
            if not _resolve(e, run_dir).exists():
                problems.append(f"metric {k}: evidence {e} missing")
    if rep.get("synthetic_quality_status") == "passed":
        if not all_pass:
            problems.append("synthetic_quality_status=passed but a frozen quality threshold is not met (or metric null)")
        if rep.get("profile") == "smoke":
            problems.append("smoke profile reports a quality pass")
    elif all_pass and rep.get("profile") != "smoke" and rep.get("synthetic_quality_status") == "failed":
        notes.append("all quality metrics meet thresholds but status is 'failed' (conservative, allowed)")
    if rep.get("real_quality_status") not in acc["real_quality"]["allowed_status"]:
        problems.append(f"real_quality_status {rep.get('real_quality_status')} not allowed")
    # a real-footage status other than not_tested needs evidence that local footage was actually processed
    processed = [p for p in (run_dir / "real_footage").glob("*/RUN_REPORT.json")] if (run_dir / "real_footage").exists() else []
    if not processed and rep.get("real_quality_status") != "not_tested":
        problems.append("no processed local real footage in <run>/real_footage but real_quality_status is not 'not_tested'")
    return {"valid": not problems, "problems": problems, "notes": notes, "engineering_status": rep.get("engineering_status"),
            "synthetic_quality_status": rep.get("synthetic_quality_status"), "real_quality_status": rep.get("real_quality_status"),
            "checks": {n: c["status"] for n, c in names.items()}}

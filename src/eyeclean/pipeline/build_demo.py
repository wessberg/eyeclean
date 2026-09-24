"""`eyeclean build-demo`: resumable end-to-end build for a profile.

Each stage records its result in <run>/manifests/build_state.json; completed stages are reused on
resume.  Prediction on held-out demo inputs runs in separate `eyeclean process` subprocesses that
cannot read <run>/eval_private (targets); evaluation reads targets only afterwards.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import torch

from ..config.loader import load_config
from ..gpu.runtime import require_cuda

ROOT = Path(__file__).resolve().parents[3]


def _state(run_dir: Path) -> dict:
    p = run_dir / "manifests" / "build_state.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"stages": {}}


def _save_state(run_dir: Path, st: dict) -> None:
    p = run_dir / "manifests" / "build_state.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".part")
    tmp.write_text(json.dumps(st, indent=1, default=str), encoding="utf-8")
    os.replace(tmp, p)


def _cli(args: list[str], log: Path, timeout: float = 7200) -> dict:
    cmd = [sys.executable, "-m", "eyeclean.cli.main", *args]
    t0 = time.time()
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w", encoding="utf-8", errors="replace") as f:
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT), timeout=timeout)
    return {"returncode": p.returncode, "seconds": round(time.time() - t0, 1), "log": str(log),
            "command": "uv run eyeclean " + " ".join(f'"{a}"' if " " in a else a for a in args)}


def build_demo(config_path: str, resume: str = "auto", workers: int = 2, stop_after: str | None = None, log=print,
               real_footage_files: list[str] | None = None, real_footage_color: str | None = None,
               real_footage_range: str | None = None) -> dict:
    cfg_obj = load_config(config_path)
    cfg = cfg_obj.raw
    run_dir = Path(cfg["run_dir"]).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    st = _state(run_dir) if resume != "never" else {"stages": {}}
    dev = require_cuda(cfg["gpu"]["policy"], cfg["gpu"]["device_index"])
    stages_out = []
    ev: dict = {"warnings": [], "stages": []}

    def stage(name: str, fn, force: bool = False):
        rec = st["stages"].get(name)
        if rec and rec.get("status") == "passed" and not force and resume != "never":
            log(f"[build] stage {name}: reuse ({rec.get('finished_at')})")
            return rec.get("result")
        log(f"[build] stage {name}: start")
        t0 = time.time()
        try:
            res = fn()
            rec = {"status": "passed", "result": res, "seconds": round(time.time() - t0, 1),
                   "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        except Exception as e:  # keep going: later stages report failures honestly
            rec = {"status": "failed", "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-4000:],
                   "seconds": round(time.time() - t0, 1)}
            log(f"[build] stage {name}: FAILED {e}")
        st["stages"][name] = rec
        _save_state(run_dir, st)
        if stop_after == name:
            raise SystemExit(0)
        return rec.get("result")

    cfg_arg = str(Path(config_path))
    # 1 doctor
    from ..validation.doctor import run_doctor

    stage("doctor", lambda: run_doctor("strict", run_dir / "doctor.json", full=True) and {"path": str(run_dir / "doctor.json")})
    # 2 generate
    from ..data.generate import generate

    stage("generate", lambda: {k: v for k, v in generate(cfg, resume="auto", workers=workers, log=log).items() if k != "failures"})
    from ..validation.inspection import dependency_manifest, sheets

    stage("inspection_sheets", lambda: sheets(run_dir))
    stage("dependency_manifest", lambda: dependency_manifest(run_dir, json.loads((run_dir / "doctor.json").read_text(encoding="utf-8"))), force=True)
    # 3 training
    from ..training.trainer import train_stage

    stage("train_guard", lambda: train_stage(cfg, "guard", resume="auto", log=log))
    if cfg["model"].get("reflection_presence"):  # v2: dedicated presence classifier on the frozen guard
        from ..training.presence import train_presence

        stage("train_presence", lambda: train_presence(cfg, resume="auto", log=log))
    for s in ("image", "temporal"):
        stage(f"train_{s}", lambda s=s: train_stage(cfg, s, resume="auto", log=log))
    # 4 bundle
    from ..training.finalize import assemble

    stage("bundle", lambda: assemble(cfg))
    best = run_dir / "checkpoints" / "best.safetensors"
    # 5 demo inputs
    from .demo_data import build_all, demo_cases

    stage("demo_inputs", lambda: {k: len(v["frames"]) for k, v in build_all(run_dir, cfg["seed"], dev).items()})
    cases = demo_cases(run_dir)
    forbid = str(run_dir / "eval_private")
    # 6 prediction subprocesses
    def predict():
        res = {}
        for case in cases:
            cdir = run_dir / "examples" / case
            pat = str(cdir / "input_exr" / "frame_%06d.exr")
            for mode_name, extra in (("temporal", []), ("single", ["--restore-mode", "single", "--no-profile"])):
                out = cdir / f"process_{mode_name}"
                if (out / "RUN_REPORT.json").exists() and resume != "never":
                    res[f"{case}/{mode_name}"] = {"returncode": 0, "reused": True}
                    continue
                if out.exists():
                    shutil.rmtree(out)
                r = _cli(["process", pat, "--checkpoint", str(best), "--input-color", "linear-rec709", "--fps", "24", "--mode", "conservative",
                          "--output", str(out), "--gpu-policy", "strict", "--export", "patches,preview,report", "--forbid-read", forbid, *extra],
                         run_dir / "logs" / f"process_{case}_{mode_name}.log")
                res[f"{case}/{mode_name}"] = r
            if "saturated" in case or "shot_cut" in case or "small_moving" in case:
                out = cdir / "process_hybrid"
                if not ((out / "RUN_REPORT.json").exists() and resume != "never"):
                    if out.exists():
                        shutil.rmtree(out)
                    res[f"{case}/hybrid"] = _cli(["process", pat, "--checkpoint", str(best), "--input-color", "linear-rec709", "--fps", "24",
                                                  "--mode", "hybrid", "--allow-synthesis", "--output", str(out), "--gpu-policy", "strict",
                                                  "--export", "patches,preview,report", "--forbid-read", forbid, "--no-profile"],
                                                 run_dir / "logs" / f"process_{case}_hybrid.log")
        bad = {k: v for k, v in res.items() if v.get("returncode") != 0}
        if bad:
            raise RuntimeError(f"process failures: {list(bad)}")
        return res

    stage("predict_demos", predict)
    # 7 special engineering runs
    from .special import (canonical_check, blink_cut_check, long_stream_run, no_eye_run, alpha_run, own_footage_tests, resume_test,
                          strength_zero_run)

    stage("special_strength_zero", lambda: strength_zero_run(run_dir, cases, best, forbid))
    stage("special_no_eye", lambda: no_eye_run(run_dir, best, dev))
    stage("special_alpha", lambda: alpha_run(run_dir, cases, best, dev))
    stage("special_long_stream", lambda: long_stream_run(run_dir, cases, best))
    stage("special_resume", lambda: resume_test(cfg, run_dir))
    stage("own_footage", lambda: own_footage_tests(run_dir, cases, best, dev))
    # 8 preservation suite
    from ..validation.preservation_suite import run_suite

    def suite():
        r = run_suite(dev)
        p = run_dir / "validation" / "preservation_suite.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(r, indent=1), encoding="utf-8")
        return {"path": str(p), "passed": r["passed"]}

    stage("preservation_suite", suite)
    # 9 evaluation
    from ..validation.evaluate import evaluate_run

    stage("evaluate", lambda: evaluate_run(run_dir)["summary"])
    stage("special_canonical", lambda: canonical_check(run_dir, cases))
    stage("special_blink_cut", lambda: blink_cut_check(run_dir, cases))
    from .special import oracle_diagnostic, real_footage

    stage("oracle_diagnostic", lambda: oracle_diagnostic(cfg, run_dir, best, dev))
    prev_rf = (st["stages"].get("real_footage") or {}).get("result") or {}
    rf_changed = bool(real_footage_files) and sorted(prev_rf.get("files") or []) != sorted(str(Path(f)) for f in real_footage_files)
    stage("real_footage", lambda: real_footage(run_dir, best, cfg, real_footage_files, real_footage_color, real_footage_range),
          force=rf_changed)
    # 10 gallery + report
    from .finalize_run import finalize

    res = stage("report", lambda: finalize(cfg, run_dir, st, best), force=True)
    return {"completed": True, "run_dir": str(run_dir), "report": res,
            "failed_stages": [k for k, v in st["stages"].items() if v.get("status") != "passed"]}

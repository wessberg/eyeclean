"""CLI integration: help, exit codes, strict errors and a small end-to-end process + apply-patches."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from conftest import smoke_checkpoint

ROOT = Path(__file__).resolve().parents[2]


def run(*args):
    return subprocess.run([sys.executable, "-m", "eyeclean.cli.main", *args], capture_output=True, text=True, cwd=str(ROOT), timeout=1800)


def test_help_lists_all_commands():
    r = run("--help")
    assert r.returncode == 0
    for c in ("doctor", "build-demo", "generate", "train", "evaluate", "process", "apply-patches", "verify-run"):
        assert c in r.stdout


def test_process_help_documents_required_flags():
    r = run("process", "--help")
    for f in ("--input-color", "--input-range", "--working-color", "--fps", "--timestamps", "--strength", "--max-frames", "--max-region",
              "--allow-synthesis", "--mode", "--export"):
        assert f in r.stdout


def _sequence(tmp: Path, dev, n=4):
    from eyeclean.io.exr import write_exr

    d = tmp / "seq"
    d.mkdir()
    g = torch.Generator(device=dev).manual_seed(0)
    for k in range(n):
        img = torch.rand((3, 180, 320), device=dev, generator=g) * 0.5
        write_exr(d / f"frame_{k:06d}.exr", {"R": img[0], "G": img[1], "B": img[2]})
    return d


def test_strict_errors(tmp, dev):
    ck = smoke_checkpoint()
    if ck is None:
        pytest.skip("no trained checkpoint")
    d = _sequence(tmp, dev)
    r = run("process", str(d / "frame_%06d.exr"), "--checkpoint", str(ck), "--input-color", "linear-rec709", "--output", str(tmp / "o1"))
    assert r.returncode == 4 and "--fps" in r.stderr  # sequences need explicit timing
    r = run("process", str(d / "frame_%06d.exr"), "--checkpoint", str(ck), "--fps", "24", "--output", str(tmp / "o2"))
    assert r.returncode == 4 and "--input-color" in r.stderr  # colour must be declared
    r = run("process", str(d / "frame_%06d.exr"), "--checkpoint", str(ck), "--input-color", "linear-rec709", "--fps", "24",
            "--mode", "hybrid", "--output", str(tmp / "o3"))
    assert r.returncode == 4 and "--allow-synthesis" in r.stderr
    r = run("process", str(d / "frame_%06d.exr"), "--checkpoint", str(ck), "--input-color", "linear-rec709", "--fps", "24",
            "--output", str(d))
    assert r.returncode == 4  # output collides with the input location


def test_process_and_apply_patches(tmp, dev):
    ck = smoke_checkpoint()
    if ck is None:
        pytest.skip("no trained checkpoint")
    d = _sequence(tmp, dev)
    out = tmp / "out"
    r = run("process", str(d / "frame_%06d.exr"), "--checkpoint", str(ck), "--input-color", "linear-rec709", "--fps", "24",
            "--output", str(out), "--export", "patches,report", "--no-profile")
    assert r.returncode == 0, r.stderr[-2000:]
    rep = json.loads((out / "RUN_REPORT.json").read_text())
    assert rep["preservation"]["source_files_modified"] == 0
    assert rep["gpu_audit"]["cpu_numerical_image_fallbacks"] == 0
    r = run("apply-patches", str(d / "frame_%06d.exr"), "--patches", str(out / "patches"), "--output", str(tmp / "m"))
    assert r.returncode == 0, r.stderr[-2000:]
    ap = json.loads((tmp / "m" / "apply_report.json").read_text())
    assert ap["verified"] and ap["outside_support_changed_bits"] == 0

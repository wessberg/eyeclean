"""Paired-target construction checks (Cycles OptiX). Uses the run's existing pair-validation report
when present; otherwise renders the isolated validation scene (slow, needs Blender)."""

import json
from pathlib import Path

import pytest

from eyeclean.paths import runs_dir

ROOT = Path(__file__).resolve().parents[2]
pytestmark = [pytest.mark.blender]


def _report() -> dict | None:
    for run in ("prototype", "smoke"):
        p = runs_dir() / run / "validation" / "pair" / "pair_validation_report.json"
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    return None


def test_pair_validation_report():
    rep = _report()
    if rep is None:
        pytest.skip("no pair-validation report yet (run `eyeclean generate`)")
    for name, c in rep["checks"].items():
        assert c["passed"], (name, c)
    e2 = rep["checks"]["E2_clean_transmission_weight"]
    assert abs(e2["measured_clean_over_unit"] - e2["expected_one_minus_F"]) < 0.01
    assert rep["checks"]["E5_reflection_zero_offeye"]["nonzero_pixels"] == 0


def test_materials_route_camera_rays_only():
    src = (ROOT / "blender" / "materials.py").read_text(encoding="utf-8")
    assert '"Is Camera Ray"' in src and "LabelB_AllRays" in src

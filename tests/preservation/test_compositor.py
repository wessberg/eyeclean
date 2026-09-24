import torch

from eyeclean.inference.compositor import EyePatch, apply_signed_correction, changed_bits_outside, changed_bits_total, composite, signed_correction
from eyeclean.validation.preservation_suite import run_suite


def test_4k_fixture_suite(dev):
    r = run_suite(dev)
    assert r["passed"], r
    assert r["fixture"]["width"] == 3840 and r["fixture"]["height"] == 2160


def test_nan_candidates_never_written(dev):
    ref = torch.rand((3, 32, 32), device=dev)
    hard = torch.ones((8, 8), dtype=torch.bool, device=dev)
    delta = torch.full((3, 8, 8), float("nan"), device=dev)
    res = composite(ref, [EyePatch((4, 4, 12, 12), delta, hard, torch.ones((8, 8), device=dev))], 1.0)
    assert changed_bits_total(ref, res.output) == 0
    assert res.assigned_pixels == 0


def test_strength_zero_and_empty_bypass(dev):
    ref = torch.randn((3, 20, 20), device=dev)
    ref[0, 0, 0] = -0.0
    hard = torch.ones((4, 4), dtype=torch.bool, device=dev)
    p = EyePatch((2, 2, 6, 6), torch.ones((3, 4, 4), device=dev), hard, torch.ones((4, 4), device=dev))
    r0 = composite(ref, [p], 0.0)
    assert r0.bypass == "strength_zero" and changed_bits_total(ref, r0.output) == 0
    r1 = composite(ref, [EyePatch((2, 2, 6, 6), p.delta, torch.zeros_like(hard), p.feather)], 1.0)
    assert r1.bypass == "empty_support" and changed_bits_total(ref, r1.output) == 0


def test_signed_correction_apply_roundtrip(dev):
    ref = torch.rand((3, 40, 40), device=dev) * 10
    hard = torch.zeros((10, 10), dtype=torch.bool, device=dev)
    hard[2:8, 3:9] = True
    delta = torch.randn((3, 10, 10), device=dev)
    res = composite(ref, [EyePatch((5, 5, 15, 15), delta, hard, torch.ones((10, 10), device=dev))], 1.0)
    corr = signed_correction(ref, res.output, res.support, (5, 5, 15, 15))
    assert float(corr[:, ~hard].abs().max()) == 0.0  # exactly zero outside support
    out2 = apply_signed_correction(ref, corr, hard, (5, 5, 15, 15))
    assert changed_bits_outside(ref, out2, res.support) == 0
    assert float((out2 - res.output).abs().max()) < 1e-5

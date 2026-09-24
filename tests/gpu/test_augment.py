import random

import pytest
import torch

from eyeclean.training.augment import CamParams, choose_exposure

pytestmark = pytest.mark.gpu


def test_exposure_ignores_out_of_frame_padding():
    """Regression: a native crop mostly outside the render (zero padding) must not drive the
    median to 0 and blow the crop out (seen on close-ups near the frame edge)."""
    dev = torch.device("cuda")
    clean = torch.zeros(3, 64, 64, device=dev)
    clean[:, :, :26] = 0.1  # ~40% rendered pixels, luminance 0.1; the rest is padding
    valid = torch.zeros(64, 64, dtype=torch.bool, device=dev)
    valid[:, :26] = True
    ocular = torch.zeros(64, 64, dtype=torch.bool, device=dev)
    p = CamParams()
    k = choose_exposure(clean, torch.zeros_like(clean), ocular, random.Random(0), p, valid=valid)
    assert 0.07 / 0.1 - 1e-6 <= k <= 0.3 / 0.1 + 1e-6
    k_all = choose_exposure(clean, torch.zeros_like(clean), ocular, random.Random(0), p)
    assert k_all > 1000  # what the unmasked median did


def test_monochrome_is_greyscale_and_shared_by_the_pair(dev):
    """Black-and-white delivery: both frames get the identical channel mix, so the pair still differs
    only by the reflection and the target is not given a colour cue the source lacks."""
    import random
    from dataclasses import replace
    from eyeclean.training.augment import CamParams, sample_params, simulate_pair

    sim = {"exposure_midgray_min": 0.08, "exposure_midgray_max": 0.5, "highlight_rolloff_prob": 0.0,
           "codec_prob": 0.0, "forced_saturation_prob": 0.0, "sclera_exposure_prob": 0.0, "monochrome_prob": 1.0}
    p = sample_params(random.Random(7), sim=sim)
    assert p.mono is not None and abs(sum(p.mono) - 1.0) < 1e-6
    S = 64
    g = torch.Generator(device=dev).manual_seed(3)
    clean = torch.rand(3, S, S, device=dev) * 0.5 + 0.1
    refl = torch.zeros(3, S, S, device=dev)
    refl[:, 20:30, 20:30] = 0.4
    lab = torch.zeros(6, S, S, device=dev)
    lab[0, 8:56, 8:56] = 1.0
    out = simulate_pair(clean, refl, lab[0], p, g)
    for k in ("src", "tgt"):
        v = out[k]
        assert float((v[0] - v[1]).abs().max()) < 2e-3, f"{k} is not greyscale"
        assert float((v[1] - v[2]).abs().max()) < 2e-3, f"{k} is not greyscale"
    # colour version of the same draw still differs between channels
    pc = replace(p, mono=None)
    oc = simulate_pair(clean, refl, lab[0], pc, torch.Generator(device=dev).manual_seed(3))
    assert float((oc["src"][0] - oc["src"][1]).abs().max()) > 1e-2

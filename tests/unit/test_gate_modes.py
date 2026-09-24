import pytest
import torch

from eyeclean.gpu.imgops import CropTransform
from eyeclean.inference.gate import DEFAULT_GATE, check_retired, crop_support, full_res_gate, highlight_excess

pytestmark = pytest.mark.gpu


def _inputs(dev, conf_logit: float, synth_logit: float):
    S = 32
    probs = torch.zeros(6, S, S, device=dev)
    probs[0, 8:24, 8:24] = 1.0  # ocular surface
    probs[4, 12:20, 12:20] = 1.0  # reflection
    out = {"support": torch.full((1, S, S), -9.0, device=dev), "conf": torch.full((1, S, S), conf_logit, device=dev),
           "synth": torch.full((1, S, S), synth_logit, device=dev), "residual": torch.full((3, S, S), 0.2, device=dev)}
    return probs, out


def test_low_confidence_reflection_abstains_in_conservative_and_is_synthesis_candidate_in_hybrid(dev):
    probs, out = _inputs(dev, conf_logit=-6.0, synth_logit=-6.0)  # confident it is NOT recoverable, synth head silent
    c = crop_support(probs, out, "conservative", DEFAULT_GATE)
    assert not bool(c["editable"].any()) and int(c["abstained"].sum()) == 64 and not bool(c["synth"].any())
    h = crop_support(probs, out, "hybrid", DEFAULT_GATE)
    assert int(h["synth"].sum()) == 64 and bool((h["editable"] == h["synth"]).all())
    assert not bool(h["abstained"].any())


def test_recoverable_reflection_is_not_synthesized(dev):
    probs, out = _inputs(dev, conf_logit=6.0, synth_logit=-6.0)
    for mode in ("conservative", "hybrid"):
        c = crop_support(probs, out, mode, DEFAULT_GATE)
        assert int(c["editable"].sum()) == 64 and not bool(c["synth"].any())


def test_no_reflection_means_no_edit_in_either_mode(dev):
    probs, out = _inputs(dev, conf_logit=-6.0, synth_logit=6.0)
    out["residual"] = torch.zeros_like(out["residual"])  # the network proposes no correction
    for mode in ("conservative", "hybrid"):
        c = crop_support(probs, out, mode, DEFAULT_GATE)
        assert not bool(c["editable"].any())


def _frg_inputs(dev, S: int = 256):
    surf = torch.zeros(S, S, dtype=torch.bool, device=dev)
    surf[60:200, 40:220] = True
    return {"surface": surf, "editable": surf.clone(), "synth": torch.zeros_like(surf),
            "conf": torch.ones(S, S, device=dev), "selected": surf.clone(),
            "abstained": torch.zeros_like(surf), "protected": torch.zeros_like(surf)}


@pytest.mark.parametrize("space", ["native", "crop"])
@pytest.mark.parametrize("fmode", ["support", "surface"])
def test_feather_is_exactly_zero_outside_hard_support(dev, space, fmode):
    """Spec 4.2: feather weights exist only inside H, in either margin space."""
    comp = _frg_inputs(dev)
    tf = CropTransform(100.0, 100.0, 5.27, 0.0, 256)
    g = dict(DEFAULT_GATE, margin_space=space, feather_mode=fmode)
    fr = full_res_gate(comp, tf, (60, 60, 140, 140), 17.7, g, None)
    A, H = fr["feather"], fr["hard"]
    assert not bool(((A > 0) & ~H).any()), "feather leaks outside the hard support"
    assert float(A.min()) >= 0.0 and float(A.max()) <= 1.0


def test_crop_margin_keeps_more_of_a_small_eye_than_whole_native_pixels(dev):
    """At an 18 px iris a 2 px native erosion is 11% of the iris; the same fraction resolved in crop
    space is sub-native-pixel, so it must not remove more of the surface."""
    comp = _frg_inputs(dev)
    tf = CropTransform(100.0, 100.0, 5.27, 0.0, 256)
    box = (60, 60, 140, 140)
    nat = full_res_gate(comp, tf, box, 17.7, dict(DEFAULT_GATE, margin_space="native", erode_min_px=2), None)
    crp = full_res_gate(comp, tf, box, 17.7, dict(DEFAULT_GATE, margin_space="crop", erode_min_crop_px=1,
                                                 erode_frac=0.0, feather_frac=0.0, feather_min_crop_px=1), None)
    assert int(crp["hard"].sum()) > int(nat["hard"].sum())


def test_large_eye_margins_agree_between_spaces(dev):
    """erode_frac * iris is scale invariant, so the two spaces must not diverge on a well-resolved eye."""
    comp = _frg_inputs(dev)
    tf = CropTransform(300.0, 300.0, 0.56, 0.0, 256)  # 120 px iris -> 67 crop px
    box = tf.frame_box(600, 600)
    nat = full_res_gate(comp, tf, box, 120.0, dict(DEFAULT_GATE, margin_space="native"), None)
    crp = full_res_gate(comp, tf, box, 120.0, dict(DEFAULT_GATE, margin_space="crop"), None)
    n, c = int(nat["hard"].sum()), int(crp["hard"].sum())
    assert abs(n - c) <= 0.08 * max(n, c), f"native {n} vs crop {c} disagree on a large eye"


def _presence_inputs(dev, residual: float):
    S = 32
    probs = torch.zeros(6, S, S, device=dev)
    probs[0, 8:24, 8:24] = 1.0                       # ocular surface
    probs[4, 10:20, 10:20] = 1.0                     # reflection mask
    out = {"support": torch.full((1, S, S), 6.0, device=dev), "conf": torch.full((1, S, S), 6.0, device=dev),
           "synth": torch.full((1, S, S), -6.0, device=dev),
           "residual": torch.full((3, S, S), residual, device=dev)}
    return probs, out


def test_presence_needs_both_classifier_and_a_real_correction(dev):
    """Measured at zero false alarms, the restorer's mean correction separates reflection-free eyes
    better than the classifier; neither signal may green-light an edit on its own."""
    g = dict(DEFAULT_GATE, min_mean_correction=0.02)
    probs, out = _presence_inputs(dev, 0.0)          # classifier certain, but nothing to correct
    assert not crop_support(probs, out, "conservative", g, refl_present=0.99)["reflection_present"]
    probs, out = _presence_inputs(dev, 0.30)         # both agree
    assert crop_support(probs, out, "conservative", g, refl_present=0.99)["reflection_present"]
    assert not crop_support(probs, out, "conservative", g, refl_present=0.10)["reflection_present"]


def test_presence_thresholds_of_zero_disable_each_test(dev):
    """The validation sweep must be able to turn either piece of evidence off."""
    probs, out = _presence_inputs(dev, 0.30)
    g = dict(DEFAULT_GATE, min_mean_correction=0.0, refl_present_thresh=0.0)
    assert crop_support(probs, out, "conservative", g, refl_present=0.01)["reflection_present"]
    probs, out = _presence_inputs(dev, 0.0)          # no correction proposed: still must not edit
    assert not crop_support(probs, out, "conservative", dict(g, min_mean_correction=0.02),
                            refl_present=0.99)["reflection_present"]


def test_track_hysteresis_relaxes_presence_only_with_consistent_evidence(dev):
    """Round 4: with track evidence established by the caller, a frame whose mean correction is below
    the strict threshold is still edited if it keeps selected area, a quarter of the correction
    threshold and half the classifier threshold; without the flag the decision is bit-identical."""
    probs, out = _inputs(dev, conf_logit=6.0, synth_logit=-6.0)
    out["residual"] = torch.full_like(out["residual"], 0.03)  # mean correction 0.03 < 0.08
    g = dict(DEFAULT_GATE, min_mean_correction=0.08, refl_present_thresh=0.5)
    strict = crop_support(probs, out, "conservative", g, refl_present=0.9)
    assert not strict["reflection_present"] and not strict["strict_present"] and not bool(strict["editable"].any())
    rel = crop_support(probs, out, "conservative", g, refl_present=0.9, relaxed=True)
    assert rel["reflection_present"] and not rel["strict_present"] and int(rel["editable"].sum()) == 64
    weak = dict(out, residual=torch.full_like(out["residual"], 0.01))  # below a quarter of the threshold
    assert not crop_support(probs, weak, "conservative", g, refl_present=0.9, relaxed=True)["reflection_present"]
    assert not crop_support(probs, out, "conservative", g, refl_present=0.2, relaxed=True)["reflection_present"]
    p0 = probs.clone()
    p0[4] = 0.0  # nothing selected: never edited, whatever the track says
    assert not crop_support(p0, out, "conservative", g, refl_present=0.9, relaxed=True)["reflection_present"]
    full = crop_support(probs, out, "conservative", dict(g, min_mean_correction=0.02), refl_present=0.9)
    assert full["reflection_present"] and full["strict_present"]


def test_highlight_excess_marks_only_the_highlight(dev):
    """Round 4: the membrane core comes from the source's own highlight; textured but unlit skin or
    sclera must not count as bright, and a small bright square must (its own pixels do not lift the
    background it is measured against)."""
    S = 256
    g = torch.Generator(device=dev).manual_seed(0)
    base = 0.02 + 0.01 * torch.rand(3, S, S, device=dev, generator=g)  # dark-ish textured linear field
    ex0 = highlight_excess(base, "linear-acescg")
    assert float((ex0 > 0.10).float().mean()) < 0.01
    img = base.clone()
    img[:, 120:140, 130:150] = 0.6  # a bright square (linear), ~20x20 crop px
    ex = highlight_excess(img, "linear-acescg")
    sq = torch.zeros(S, S, dtype=torch.bool, device=dev)
    sq[120:140, 130:150] = True
    assert bool((ex[sq] > 0.10).all())
    outside = (ex > 0.10) & ~sq
    assert int(outside.sum()) < 0.02 * S * S


def test_retired_gate_settings_are_refused():
    """Bundles that ask for a fill retired in 0.3.0 must fail loudly, never render differently."""
    check_retired(dict(DEFAULT_GATE))
    check_retired(dict(DEFAULT_GATE, fill_mode="network_value", membrane_pupil_aware=0, pupil_fit_mode="visible_boundary"))
    for bad in ({"fill_mode": "membrane_core"}, {"fill_mode": "polar_copy"}, {"restore_input": "hole"},
                {"value_pupil_circle": True}, {"fill_temporal_window": 2}, {"value_core_source": "excess"}):
        try:
            check_retired(dict(DEFAULT_GATE, **bad))
        except ValueError:
            continue
        raise AssertionError(f"{bad} was accepted")

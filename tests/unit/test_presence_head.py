import pytest
import torch

from eyeclean.inference.gate import DEFAULT_GATE, crop_support
from eyeclean.models.eyeguard import EyeGuard
from eyeclean.training.data import targets_from_batch
from eyeclean.training.losses import guard_loss

pytestmark = pytest.mark.gpu


def test_presence_head_is_optional_and_checkpoint_compatible(dev):
    g0 = EyeGuard().to(dev)
    g1 = EyeGuard(presence_head=True).to(dev)
    x = torch.rand(2, 4, 64, 64, device=dev)
    o0, o1 = g0(x), g1(x)
    assert "refl_present" not in o0 and o1["refl_present"].shape == (2,)
    missing, unexpected = g1.load_state_dict(g0.state_dict(), strict=False)
    assert set(missing) == {"refl_ev.weight", "refl_ev.bias"} and not unexpected


def _batch(dev, negative: bool, amp: float):
    S = 32
    lab = torch.zeros(1, 6, S, S, device=dev)
    lab[:, 0, 8:24, 8:24] = 1.0  # ocular
    src = torch.full((1, 3, S, S), 0.2, device=dev)
    tgt = src.clone()
    if not negative:
        src[:, :, 12:16, 12:16] += amp
    return {"src": src, "tgt": tgt, "labels": lab, "clipped": torch.zeros(1, 1, S, S, device=dev), "auth": torch.ones(1, 1, S, S, device=dev),
            "valid": torch.ones(1, 1, S, S, device=dev), "openness": torch.ones(1, device=dev), "negative": torch.tensor([negative], device=dev),
            "force_absent": torch.tensor([False], device=dev)}


def test_v2_targets_visible_faint_and_negative(dev):
    e = torch.full((1,), 0.4, device=dev)
    vis = targets_from_batch(_batch(dev, False, 0.5), e, vis=0.06)
    assert vis["refl_present"].item() == 1.0 and vis["refl_present_w"].item() == 1.0
    faint = targets_from_batch(_batch(dev, False, 0.012), e, vis=0.06)  # normalised change ~0.027
    assert faint["refl_present_w"].item() == 0.0  # faint-only reflection: don't care
    assert float(faint["mask_weight"][:, 4].min()) == 0.0 and float(faint["masks"][:, 4].max()) == 0.0
    neg = targets_from_batch(_batch(dev, True, 0.0), e, vis=0.06)
    assert neg["refl_present"].item() == 0.0 and neg["refl_present_w"].item() == 1.0
    v1 = targets_from_batch(_batch(dev, False, 0.5), e)
    assert "refl_present" not in v1 and "mask_weight" not in v1
    g = EyeGuard(presence_head=True).to(dev)
    from eyeclean.models.inputs import guard_input
    b = _batch(dev, True, 0.0)
    loss, parts = guard_loss(g(guard_input(b["src"], e)), neg, b["valid"])
    assert torch.isfinite(loss) and "refl_present" in parts


def test_gate_uses_presence_probability_when_given(dev):
    S = 32
    probs = torch.zeros(6, S, S, device=dev)
    probs[0, 8:24, 8:24] = 1.0
    probs[4, 12:20, 12:20] = 1.0
    out = {"support": torch.full((1, S, S), -9.0, device=dev), "conf": torch.full((1, S, S), 6.0, device=dev),
           "synth": torch.full((1, S, S), -9.0, device=dev), "residual": torch.full((3, S, S), 0.2, device=dev)}
    assert not bool(crop_support(probs, out, "conservative", DEFAULT_GATE, refl_present=0.1)["editable"].any())
    assert int(crop_support(probs, out, "conservative", DEFAULT_GATE, refl_present=0.9)["editable"].sum()) == 64
    assert int(crop_support(probs, out, "conservative", DEFAULT_GATE)["editable"].sum()) == 64  # v1 heuristic unchanged

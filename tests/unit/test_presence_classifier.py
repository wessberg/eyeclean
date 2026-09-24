import pytest
import torch

from eyeclean.models.presence import ReflectionPresence, presence_input
from eyeclean.training.presence import _metrics, presence_labels

pytestmark = pytest.mark.gpu


def test_no_visible_eye_surface_means_no_reflection(dev):
    m = ReflectionPresence().to(dev)
    gi = torch.rand(2, 4, 64, 64, device=dev)
    probs = torch.zeros(2, 6, 64, 64, device=dev)
    probs[0, 0, 16:48, 16:48] = 1.0  # eye 0: visible ocular surface; eye 1: closed (no surface)
    logit = m(presence_input(gi, probs))
    assert logit.shape == (2,) and float(logit[1]) < float(logit[0]) - 5.0


def _b(dev, negative: bool, amp: float, eye: bool = True):
    S = 32
    lab = torch.zeros(1, 6, S, S, device=dev)
    if eye:
        lab[:, 0, 8:24, 8:24] = 1.0
    src = torch.full((1, 3, S, S), 0.2, device=dev)
    tgt = src.clone()
    if not negative:
        src[:, :, 10:16, 10:16] += amp
    return {"src": src, "tgt": tgt, "labels": lab, "valid": torch.ones(1, 1, S, S, device=dev),
            "negative": torch.tensor([negative], device=dev), "force_absent": torch.tensor([False], device=dev)}


def test_presence_labels(dev):
    e = torch.full((1,), 0.4, device=dev)
    y, w = presence_labels(_b(dev, False, 0.6), e, 0.06)
    assert y.item() == 1 and w.item() == 1  # clear reflection
    y, w = presence_labels(_b(dev, True, 0.0), e, 0.06)
    assert y.item() == 0 and w.item() == 1  # reflection-free eye
    y, w = presence_labels(_b(dev, False, 0.03), e, 0.06)
    assert w.item() == 0  # weak reflection: don't care
    y, w = presence_labels(_b(dev, False, 0.6, eye=False), e, 0.06)
    assert y.item() == 0 and w.item() == 1  # no visible eye surface (e.g. closed): no reflection to edit


def test_metrics_threshold_at_one_percent():
    neg = [i / 1000 for i in range(1000)]  # uniform scores on negatives
    pos = [0.995, 0.999, 0.5]
    m = _metrics(pos, neg)
    assert m["threshold_at_fpr_1pct"] >= 0.989 and abs(m["tpr_at_fpr_1pct"] - 2 / 3) < 1e-9

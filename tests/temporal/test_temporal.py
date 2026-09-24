import torch

from eyeclean.inference import canonical as CAN
from eyeclean.models.eyerestore import RESTORE_IN, EyeRestore
from eyeclean.tracking.tracker import CutDetector, EyeTracker


def test_zero_gate_video_equals_single_frame(dev):
    m = EyeRestore(checkpoint=False).to(dev).eval()
    xs = torch.randn(1, 5, RESTORE_IN, 128, 128, device=dev)
    v = torch.ones(1, 5, dtype=torch.bool, device=dev)
    assert all(float(l.gate.abs().max()) == 0.0 for l in m.temporal.values())
    with torch.no_grad():
        a = m(xs[:, 2])
        b = m.forward_video(xs, v)
    # identical maths; cuDNN may choose different TF32 algorithms for batch 1 vs batch 5
    assert float((a["support"] - b["support"]).abs().max()) < 2e-3


def test_invalid_neighbours_have_no_influence(dev):
    m = EyeRestore(checkpoint=False).to(dev).eval()
    with torch.no_grad():
        for lvl in m.temporal.values():
            lvl.gate.fill_(1.0)  # activate fusion
    xs = torch.randn(1, 5, RESTORE_IN, 64, 64, device=dev)
    v = torch.tensor([[False, True, True, True, False]], device=dev)
    xs2 = xs.clone()
    xs2[:, 0] += 5.0
    xs2[:, 4] -= 5.0  # change only invalid frames (e.g. across a cut / closed eye)
    with torch.no_grad():
        a = m.forward_video(xs, v)["support"]
        b = m.forward_video(xs2, v)["support"]
    assert float((a - b).abs().max()) < 1e-4


def test_tracker_resets_on_cut():
    t = EyeTracker()
    t.new_shot()
    ids0 = t.update(0, [{"center": (100, 100), "width": 20}, {"center": (160, 100), "width": 20}])
    ids1 = t.update(1, [{"center": (101, 100), "width": 20}, {"center": (161, 101), "width": 20}])
    assert ids0 == ids1
    t.new_shot()
    ids2 = t.update(2, [{"center": (101, 100), "width": 20}])
    assert not set(ids2) & set(ids0)  # never transfer identity across a cut


def test_cut_detector(dev):
    cd = CutDetector()
    a = torch.rand(3, 90, 160, device=dev) * 0.2
    b = a + 0.01
    c = torch.rand(3, 90, 160, device=dev) * 0.2 + 0.7
    assert cd(a)[0] is True  # first frame starts a shot
    assert cd(b)[0] is False
    assert cd(c)[0] is True


def test_canonical_frozen_and_reused(dev):
    st = CAN.CanonicalState.new("shot0/track0", dev)
    S = 64
    crop = torch.rand(3, S, S, device=dev) * 0.2
    w = torch.ones(S, S, device=dev)
    CAN.observe(st, crop, w, w * 0, 32.0, 32.0, 20.0, 6.0)
    meta = CAN.freeze(st)
    assert meta["frozen"] and st.digest
    a, _ = CAN.render(st, S, 30.0, 31.0, 18.0, 5.0, 1.0)
    b, _ = CAN.render(st, S, 30.0, 31.0, 18.0, 5.0, 1.0)
    assert torch.equal(a, b)
    try:
        CAN.observe(st, crop, w, w * 0, 32.0, 32.0, 20.0, 6.0)
        raised = False
    except RuntimeError:
        raised = True
    assert raised  # no per-frame refitting once frozen

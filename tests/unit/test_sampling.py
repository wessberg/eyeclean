from pathlib import Path

from eyeclean.config.loader import load_config
from eyeclean.data.sampling import HELDOUT_LIGHT, blink_curve, plan_dataset, plan_digest, split_integrity, still_openness
from eyeclean.procedural.iris import HELDOUT_STRUCTURE

ROOT = Path(__file__).resolve().parents[2]


def test_plan_deterministic_and_disjoint():
    cfg = load_config(ROOT / "configs" / "smoke.yaml").raw
    a, b = plan_dataset(cfg), plan_dataset(cfg)
    assert plan_digest(a) == plan_digest(b)
    si = split_integrity(a)
    assert si["passed"], si["problems"]


def test_heldout_families_only_in_test():
    cfg = load_config(ROOT / "configs" / "prototype.yaml").raw
    p = plan_dataset(cfg)
    for iid, ident in p.identities.items():
        if ident["split"] != "test":
            assert ident["iris_family"] != HELDOUT_STRUCTURE
    for sh in p.shots:
        if sh["split"] != "test":
            assert all(L["family"] != HELDOUT_LIGHT for L in sh["lights"])
    si = split_integrity(p)
    assert si["heldout_iris_in_test"] and si["heldout_light_in_test"]


def test_negative_fraction_and_blinks():
    cfg = load_config(ROOT / "configs" / "prototype.yaml").raw
    p = plan_dataset(cfg)
    train = [s for s in p.shots if s["split"] == "train"]
    assert sum(s["negative"] for s in train) / len(train) >= cfg["data"]["negative_fraction_min"]
    clips = [s for s in p.shots if s["kind"] == "clip"]
    assert sum(s["has_blink"] for s in clips) >= 0.3 * len(clips)


def test_blink_curve_closes_and_reopens():
    import random

    c = blink_curve(16, 4, 1.0, random.Random(0))
    assert c[0] == 1.0 and min(c) == 0.0 and c[-1] > 0.95
    assert c.index(0.0) - 4 <= 3  # fast close


def test_still_openness_distribution():
    import random

    r = random.Random(1)
    v = [still_openness(r) for _ in range(4000)]
    closed = sum(x == 0.0 for x in v) / len(v)
    partial = sum(0.0 < x < 0.85 for x in v) / len(v)
    assert 0.04 < closed < 0.1 and 0.15 < partial < 0.35

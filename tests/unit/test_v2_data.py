"""v2 data options: the prototype plan must stay bit-identical; v2 must deliver the requested coverage."""

import collections
import math
import random
from pathlib import Path

import pytest
import torch

from eyeclean.config.loader import load_config
from eyeclean.data.sampling import AGE_BINS, plan_dataset, plan_digest, skin_meta, split_integrity, years_to_macro_age
from eyeclean.training.augment import CamParams, block_dct_quantize, rolloff, sample_params, simulate_pair

ROOT = Path(__file__).resolve().parents[2]


def test_prototype_plan_unchanged_by_v2_options():
    cfg = load_config(ROOT / "configs" / "prototype.yaml").raw
    assert "camera_sim" not in cfg["data"] and "demographics" not in cfg["data"]
    a = plan_dataset(cfg)
    assert plan_digest(a) == plan_digest(plan_dataset(cfg))
    assert not any(s.get("clean_iris_render") for s in a.shots)
    assert not any("demographics" in i for i in a.identities.values())


def test_v2_demographics_balanced_and_matched():
    cfg = load_config(ROOT / "configs" / "v2.yaml").raw
    plan = plan_dataset(cfg)
    assert split_integrity(plan)["passed"]
    tr = [i for i in plan.identities.values() if i["split"] == "train"]
    bins = collections.Counter((i["age_bin"], i["demographics"]["sex"]) for i in tr)
    assert set(b for b, _ in bins) == {b[0] for b in AGE_BINS}
    assert max(bins.values()) - min(bins.values()) <= 1  # every age bin x sex equally represented
    for i in tr:
        m = skin_meta(i["skin"])
        assert not m["costume"]
        assert m["sex"] in (i["demographics"]["sex"], "any")
        if i["age_bin"] == "child":
            assert m["age"] in ("young", "any")
        if m["natural_makeup"]:
            assert i["demographics"]["sex"] == "f"
    # the general still population keeps its ~50% key-softbox rate; portrait stills deliberately
    # draw the key from every light family, so they are counted separately below
    st = [s for s in plan.shots if s["split"] == "train" and s["kind"] == "still" and not s.get("portrait")]
    key = sum(any(L.get("key_softbox") for L in s["lights"]) for s in st) / len(st)
    assert 0.4 < key < 0.6
    assert all(s.get("clean_iris_render") for s in plan.shots)


def test_portrait_stills_are_deployment_shaped():
    """Eyes into the lens, one clear catchlight on the iris, every light family, both eyes used."""
    from collections import Counter

    cfg = load_config("configs/v2.yaml").raw
    plan = plan_dataset(cfg)
    # both deployment-shaped populations: the ordinary one and the fully-open-lid one, which differs
    # only in its openness draw and must satisfy every property below as well
    po = [s for s in plan.shots if s.get("portrait") and s["split"] == "train"]
    assert len(po) == cfg["data"]["portrait_pairs"]["train"] + cfg["data"]["portrait_open_pairs"]["train"]
    assert all(s["kind"] == "still" and s["lights"][0].get("portrait_key") for s in po)
    # near-axis camera and gaze: this is the reporter/portrait case, not an off-angle one
    assert all(abs(s["camera"]["azimuth_deg"]) <= 10 and abs(s["camera"]["elevation_deg"]) <= 8 for s in po)
    assert all(abs(s["frames"][0]["gaze"]["L"][0]) <= 8 for s in po)
    # the key sweeps around the iris rather than sitting above it, and is not always a softbox
    fams = Counter(s["lights"][0]["family"] for s in po)
    assert len(fams) >= 8, fams
    assert fams.most_common(1)[0][1] / len(po) < 0.35, "one light family dominates the portrait key"
    az = [s["lights"][0]["azimuth_deg"] for s in po]
    assert min(az) < -30 and max(az) > 30
    # framings the user asked for, with left and right used interchangeably
    frm = Counter(s["framing"] for s in po)
    assert set(frm) == {"two_eyes", "one_eye", "ultra_closeup"} and all(v > 0.15 * len(po) for v in frm.values())
    tgt = Counter(s["camera"]["target"] for s in po)
    assert abs(tgt["eye_L"] - tgt["eye_R"]) < 0.1 * (tgt["eye_L"] + tgt["eye_R"])
    ultra = [s["camera"]["iris_px"] for s in po if s["framing"] == "ultra_closeup"]
    assert min(ultra) >= 150 and max(ultra) > 250, "ultra close-ups are not actually close"
    # reflection-free portrait eyes still exist: they buy the no-false-edit contract
    assert 0.2 < sum(s["negative"] for s in po) / len(po) < 0.3


def test_age_mapping_matches_makehuman():
    assert years_to_macro_age(1) == 0.0 and abs(years_to_macro_age(11) - 0.1875) < 1e-9
    assert abs(years_to_macro_age(25) - 0.5) < 1e-9 and years_to_macro_age(90) == 1.0


@pytest.mark.gpu
def test_rolloff_is_monotone_continuous_and_reaches_white(dev):
    x = torch.linspace(0, 10, 20001, device=dev)
    y = rolloff(x, 0.5, 4.0)
    assert bool((y[1:] >= y[:-1] - 1e-7).all())
    assert float(y[x <= 0.5].sub(x[x <= 0.5]).abs().max()) == 0.0
    assert abs(float(y[x >= 4.0].min()) - 1.0) < 1e-6 and float(y.max()) <= 1.0 + 1e-6
    assert float((y[1:] - y[:-1]).abs().max()) < 1e-3  # no jump at the knee


@pytest.mark.gpu
def test_codec_error_is_shared_by_source_and_target(dev):
    """With the codec on, source and target differ only by the (pre-codec) reflection term."""
    g = torch.Generator(device=dev)
    g.manual_seed(1)
    clean = torch.rand((3, 64, 64), device=dev, generator=g) * 0.3
    refl = torch.zeros_like(clean)
    refl[:, 30:34, 30:34] = 0.4
    ocular = torch.zeros((64, 64), device=dev)
    ocular[20:44, 20:44] = 1
    p = CamParams(exposure=1.0, encoding="rec709", bits=8, chroma="420", codec_q=1.0, codec_off=(3, 5))
    out = simulate_pair(clean, refl, ocular, p, None)
    p0 = CamParams(exposure=1.0, encoding="rec709", bits=8, chroma="420")
    out0 = simulate_pair(clean, refl, ocular, p0, None)
    d_codec = (out["tgt"] - out["src"]).abs().amax(0)
    d_plain = (out0["tgt"] - out0["src"]).abs().amax(0)
    far = torch.ones_like(d_codec, dtype=torch.bool)
    far[26:38, 26:38] = False  # away from the reflection (and its blur/chroma footprint)
    assert float(d_codec[far].max()) < 1e-5 and float(d_plain[far].max()) < 1e-5
    assert float((out["src"] - out0["src"]).abs().max()) > 1e-3  # the codec did change the image


@pytest.mark.gpu
def test_block_dct_identity_at_tiny_quantisation(dev):
    x = torch.rand((20, 27), device=dev) * 255
    y = block_dct_quantize(x, ((1,) * 8,) * 8, 1e-4, 8, (2, 5))
    assert y.shape == x.shape and float((x - y).abs().max()) < 1e-2


def test_v1_camera_draws_unchanged_without_sim():
    a = sample_params(random.Random(7))
    b = sample_params(random.Random(7), sim=None)
    assert a == b and a.knee == 0.0 and a.codec_q == 0.0
    r = random.Random(7)
    c = sample_params(r, sim={"highlight_rolloff_prob": 1.0, "codec_prob": 1.0})
    assert c.knee > 0 and (c.codec_q > 0 or c.encoding == "linear")
    assert math.isclose(c.defocus_sigma, a.defocus_sigma)  # v2 draws come after the v1 draws


def test_generate_accepts_superset_plan_and_refuses_reseeded_one(tmp_path):
    """Round 2: growing ``portrait_pairs`` must resume onto the rendered shots; changing the identity
    pool re-seeds every shot and must still be refused."""
    import json

    from eyeclean.data.generate import GenerationError, plan_extends, prepare_plan

    cfg = load_config(ROOT / "configs" / "smoke.yaml").raw
    cfg["data"]["portrait_pairs"] = {"train": 4, "validation": 2, "test": 2}
    run = tmp_path / "run"
    base = prepare_plan(cfg, run, "auto")
    plan_p = run / "manifests" / "dataset_plan.json"
    old = json.loads(plan_p.read_text(encoding="utf-8"))
    assert old["digest"] == plan_digest(base) and not old.get("extends")
    # same config again: unchanged file
    prepare_plan(cfg, run, "auto")
    assert json.loads(plan_p.read_text(encoding="utf-8")) == old
    # grow the portrait population: accepted, old shots untouched, backup written
    grown = dict(cfg, data=dict(cfg["data"], portrait_pairs={"train": 8, "validation": 2, "test": 2}))
    logs = []
    ext = prepare_plan(grown, run, "auto", log=logs.append)
    new = json.loads(plan_p.read_text(encoding="utf-8"))
    assert new["digest"] == plan_digest(ext) and new["extends"] == old["digest"]
    assert len(new["shots"]) == len(old["shots"]) + 4
    by_id = {s["shot_id"]: s for s in new["shots"]}
    assert all(by_id[s["shot_id"]] == s for s in old["shots"])
    assert list(run.glob("manifests/dataset_plan.superseded_*.json")) and logs and "extended" in logs[0]
    # --resume require still refuses any change
    with pytest.raises(GenerationError):
        prepare_plan(cfg, run, "require")
    # re-seeding the pool (identity_counts) is a different dataset
    reseed = dict(grown, data=dict(grown["data"], identity_counts={"train": 9, "validation": 4, "test": 4}))
    assert not plan_extends(new, json.loads(json.dumps(plan_dataset(reseed).to_json())))[0]
    with pytest.raises(GenerationError, match="config changed"):
        prepare_plan(reseed, run, "auto")
    # and so is shrinking it
    with pytest.raises(GenerationError, match="config changed"):
        prepare_plan(cfg, run, "auto")


def test_portrait_native_iris_bias_is_optional_and_bounded():
    """Round 2 sampler option: absent = identical random stream; present = native iris inside the
    requested range, never above the rendered size, default law for eyes rendered too small."""
    from eyeclean.training.data import sample_geo

    a = [sample_geo(random.Random(7), D, "L", 8.0).native_iris for D in (20.0, 60.0, 250.0)]
    b = [sample_geo(random.Random(7), D, "L", 8.0, native_range=None).native_iris for D in (20.0, 60.0, 250.0)]
    assert a == b
    r = random.Random(3)
    for _ in range(300):
        g = sample_geo(r, 250.0, "L", 8.0, native_range=(12.0, 36.0))
        assert 12.0 <= g.native_iris <= 36.0 and g.s1 == g.native_iris / 250.0
        g = sample_geo(r, 20.0, "L", 8.0, native_range=(12.0, 36.0))   # clipped to the rendered size
        assert 12.0 <= g.native_iris <= 20.0
        g = sample_geo(r, 10.0, "L", 8.0, native_range=(12.0, 36.0))   # too small: default law
        assert g.native_iris <= 10.0
    cfg = load_config(ROOT / "configs" / "v2.yaml").raw
    opt = cfg["data"]["camera_sim"]["portrait_native_iris"]
    assert opt == {"prob": 0.5, "min_px": 12, "max_px": 36}
    proto = load_config(ROOT / "configs" / "prototype.yaml").raw
    assert "camera_sim" not in proto["data"]


def test_portrait_open_pairs_are_fully_open_and_additive():
    """Round 2: the fully-open population must not re-seed anything already planned, must draw
    openness 0.92-1.0, and must otherwise be an ordinary deployment-shaped portrait shot."""
    import json

    from eyeclean.data.generate import plan_extends

    cfg = load_config(ROOT / "configs" / "v2.yaml").raw
    assert cfg["data"]["portrait_open_pairs"] == {"train": 4096, "validation": 128, "test": 128}
    without = dict(cfg, data={k: v for k, v in cfg["data"].items() if k != "portrait_open_pairs"})
    a, b = plan_dataset(without), plan_dataset(cfg)
    ok, why = plan_extends(json.loads(json.dumps(a.to_json())), json.loads(json.dumps(b.to_json())))
    assert ok, why
    assert len(b.shots) - len(a.shots) == 4096 + 128 + 128
    wide = [s for s in b.shots if s.get("wide_open")]
    assert len(wide) == 4352 and len({s["shot_id"] for s in b.shots}) == len(b.shots)
    for s in wide:
        assert "_portraitopen_" in s["shot_id"] and s["portrait"] and s["kind"] == "still"
        assert s["lights"][0].get("portrait_key") and s["framing"] in ("two_eyes", "one_eye", "ultra_closeup")
        for f in s["frames"]:
            assert 0.92 <= f["openness"]["L"] <= 1.0 and 0.92 <= f["openness"]["R"] <= 1.0
    assert split_integrity(b)["passed"]
    # negatives still bought at the usual rate, so the no-false-edit contract is unaffected
    neg = sum(s["negative"] for s in wide) / len(wide)
    assert 0.18 < neg < 0.32


def test_reflection_shift_is_optional_and_moves_only_the_reflection():
    """Sampler option camera_sim.reflection_shift: shift_zero translates by whole pixels with zero fill
    and no wrap-around; the option is absent from v2.yaml, so the default stream is unchanged."""
    from eyeclean.training.data import shift_zero

    x = torch.zeros(3, 6, 6)
    x[:, 2, 2] = 1.0
    y = shift_zero(x, 2, -1)
    assert float(y[:, 1, 4].sum()) == 3.0 and float(y.sum()) == 3.0
    z = shift_zero(x, 5, 0)  # pushed out of the frame: nothing wraps back in
    assert float(z.sum()) == 0.0
    assert torch.equal(shift_zero(x, 0, 0), x)
    v2 = load_config(ROOT / "configs" / "v2.yaml").raw
    assert "reflection_shift" not in v2["data"]["camera_sim"]

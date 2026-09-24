from pathlib import Path

import pytest

from eyeclean.config.loader import load_config
from eyeclean.config.schema import ConfigError

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("name", ["smoke", "prototype", "quality"])
def test_shipped_profiles_validate(name):
    c = load_config(ROOT / "configs" / f"{name}.yaml")
    assert c["profile"] == name
    assert c["gpu"]["cpu_image_fallback"] is False


def test_extends_is_relative_and_deep_merged(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "child.yaml").write_text(f"extends: {(ROOT / 'configs' / 'prototype.yaml').as_posix()}\nrender:\n  samples: 7\n", encoding="utf-8")
    c = load_config(sub / "child.yaml")
    assert c["render"]["samples"] == 7 and c["render"]["width"] == 512  # sibling keys survive the merge
    smoke = load_config(ROOT / "configs" / "smoke.yaml")
    assert smoke["render"]["width"] == 256 and smoke["render"]["engine"] == "CYCLES"


def test_unknown_key_rejected_with_suggestion(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text(f"extends: {(ROOT / 'configs' / 'prototype.yaml').as_posix()}\nrender:\n  widht: 10\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="did you mean 'width'"):
        load_config(p)


def test_contract_constants_enforced(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text(f"extends: {(ROOT / 'configs' / 'prototype.yaml').as_posix()}\ngpu:\n  cpu_image_fallback: true\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="cpu_image_fallback"):
        load_config(p)


def test_cyclic_extends_rejected(tmp_path):
    (tmp_path / "a.yaml").write_text("extends: b.yaml\n", encoding="utf-8")
    (tmp_path / "b.yaml").write_text("extends: a.yaml\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="cyclic"):
        load_config(tmp_path / "a.yaml")


def test_step_cap_consistency(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text(f"extends: {(ROOT / 'configs' / 'smoke.yaml').as_posix()}\nresources:\n  max_total_optimizer_steps: 10\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="exceed"):
        load_config(p)

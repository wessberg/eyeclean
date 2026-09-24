"""Model registry: the default resolves by name, and `fetch_model` verifies what it downloads (CPU only)."""
import hashlib

import pytest

from eyeclean.io.errors import InputError
from eyeclean.models.registry import fetch_model, load_registry


def test_registry_default_is_a_distributed_single_frame_model():
    reg = load_registry()
    entry = reg["models"][reg["default"]]
    assert entry["restore_mode"] == "single" and entry.get("distributed", True)
    assert len(entry["sha256"]) == 64 and entry["file"].endswith(".safetensors")


def _reg(tmp_path, payload: bytes, sha: str | None = None, url: bool = True):
    src = tmp_path / "hosted.bin"
    src.write_bytes(payload)
    return {"schema": "eyeclean.models/1", "default": "m", "models": {"m": {
        "file": "m.safetensors", "sha256": sha or hashlib.sha256(payload).hexdigest(),
        "url": src.resolve().as_uri() if url else None, "restore_mode": "single"}}}


def test_fetch_model_downloads_and_verifies(tmp_path):
    reg = _reg(tmp_path, b"weights")
    res = fetch_model(None, tmp_path / "models", reg)
    assert res["downloaded"] and (tmp_path / "models" / "m.safetensors").read_bytes() == b"weights"
    assert not fetch_model("m", tmp_path / "models", reg)["downloaded"]  # an installed, verified file is kept


def test_fetch_model_rejects_a_wrong_hash_and_keeps_nothing(tmp_path):
    reg = _reg(tmp_path, b"weights", sha="0" * 64)
    with pytest.raises(InputError, match="SHA-256"):
        fetch_model("m", tmp_path / "models", reg)
    assert not any((tmp_path / "models").iterdir())


def test_fetch_model_without_url_says_where_to_put_the_file(tmp_path):
    with pytest.raises(InputError, match="no download URL"):
        fetch_model("m", tmp_path / "models", _reg(tmp_path, b"weights", url=False))

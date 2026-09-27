import os
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture(scope="session")
def dev():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required (strict GPU policy)")
    return torch.device("cuda")


@pytest.fixture()
def tmp(tmp_path):
    return tmp_path


def smoke_checkpoint() -> Path | None:
    from eyeclean.paths import runs_dir
    runs = runs_dir()
    for p in (runs / "prototype" / "checkpoints" / "best.safetensors", runs / "smoke" / "checkpoints" / "best.safetensors"):
        if p.exists():
            return p
    from eyeclean.models.registry import load_registry, locate   # Round 52: the installed default model

    return locate(load_registry()["default"])

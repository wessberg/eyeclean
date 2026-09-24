"""Isolated one-time conversion of legacy face-alignment pickles to safetensors.

Security decision: the upstream S3FD/2DFAN4 files are PyTorch pickles.  They are loaded exactly
once, here, with ``torch.load(weights_only=True)`` (restricted unpickler: tensors and primitive
containers only; arbitrary code objects are refused), after verifying their SHA-256 against the
asset lock.  Inference only ever loads the converted safetensors files, whose hashes are recorded.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from ..assets.fetch import LOCK_PATH, THIRD_PARTY

CONVERTED = THIRD_PARTY / "face_alignment" / "converted"
SOURCES = {"s3fd": ("s3fd_weights", "face_alignment/s3fd-619a316812.pth"),
           "fan2d": ("fan2d_weights", "face_alignment/2DFAN4-11f355bf06.pth.tar")}


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def convert(name: str) -> Path:
    key, rel = SOURCES[name]
    src = THIRD_PARTY / rel
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))["assets"][key]
    digest = _sha(src)
    if digest != lock["sha256"]:
        raise RuntimeError(f"{src}: hash {digest} != lock {lock['sha256']}; refusing to unpickle")
    sd = torch.load(src, map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    sd = {k.replace("module.", ""): v.contiguous() for k, v in sd.items() if torch.is_tensor(v)}
    CONVERTED.mkdir(parents=True, exist_ok=True)
    out = CONVERTED / f"{name}.safetensors"
    save_file(sd, str(out), metadata={"source": rel, "source_sha256": digest, "loader": "torch.load(weights_only=True)"})
    prov = {"name": name, "source": rel, "source_sha256": digest, "converted": out.name, "converted_sha256": _sha(out),
            "tensors": len(sd), "license": lock["license"], "provenance": lock["provenance"],
            "security": "legacy pickle loaded once with weights_only=True after hash verification; inference loads safetensors only"}
    (CONVERTED / f"{name}.provenance.json").write_text(json.dumps(prov, indent=2), encoding="utf-8")
    return out


def load(name: str, device) -> dict:
    p = CONVERTED / f"{name}.safetensors"
    prov_p = CONVERTED / f"{name}.provenance.json"
    if not p.exists():
        convert(name)
    prov = json.loads(prov_p.read_text(encoding="utf-8"))
    if _sha(p) != prov["converted_sha256"]:
        raise RuntimeError(f"{p} hash mismatch against its provenance record")
    return load_file(str(p), device=str(device))


if __name__ == "__main__":
    for _n in SOURCES:
        print(convert(_n))

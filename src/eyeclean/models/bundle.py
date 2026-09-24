"""Checkpoint bundle: one safetensors file with every trained component.

Tensor prefixes: ``guard.`` (EyeGuard), ``restore_single.`` (the single-frame restorer: EyeRestoreTX in
the released model, the CNN in bundles `build-demo` writes), ``presence.`` (the optional reflection-presence
classifier) and ``restore.`` (the optional five-frame CNN EyeRestore with temporal fusion, run only by
restore_mode "temporal").  The safetensors metadata holds a JSON manifest: model config, per-component
provenance (stage, step, source file SHA-256), calibration and gate parameters, dataset/plan
digests and the frozen acceptance hash.  Loading verifies component presence and the file hash
recorded next to it; pickles are never used.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from .eyeguard import EyeGuard
from .eyerestore import build_restore
from .presence import ReflectionPresence

COMPONENTS = ("guard", "restore_single")  # required; "restore" (the five-frame CNN) and "presence" are optional


def sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_bundle(out: Path, parts: dict[str, Path], manifest: dict) -> dict:
    tensors = {}
    prov = {}
    for comp, path in parts.items():
        sd = load_file(str(path))
        for k, v in sd.items():
            tensors[f"{comp}.{k}"] = v
        prov[comp] = {"source": str(path), "source_sha256": sha256(path), "tensors": len(sd)}
    man = dict(manifest, components=prov, schema="eyeclean.bundle/1")
    tmp = out.with_suffix(".part")
    save_file(tensors, str(tmp), metadata={"eyeclean_manifest": json.dumps(man)})
    tmp.replace(out)
    digest = sha256(out)
    rec = {"path": str(out), "sha256": digest, "manifest": man}
    out.with_suffix(".manifest.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    return rec


def read_manifest(path: Path) -> dict:
    with safe_open(str(path), framework="pt") as f:
        md = f.metadata() or {}
    if "eyeclean_manifest" not in md:
        raise ValueError(f"{path}: not an EyeClean bundle (missing manifest metadata)")
    return json.loads(md["eyeclean_manifest"])


class Bundle:
    def __init__(self, path: str | Path, device: torch.device, verify_hash: bool = True):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"checkpoint {path} not found")
        self.path = path
        self.sha256 = sha256(path)
        side = path.with_suffix(".manifest.json")
        if verify_hash and side.exists():
            rec = json.loads(side.read_text(encoding="utf-8"))
            if rec["sha256"] != self.sha256:
                raise ValueError(f"{path}: SHA-256 {self.sha256} does not match its manifest {rec['sha256']}")
        self.manifest = read_manifest(path)
        missing = [c for c in COMPONENTS if c not in self.manifest.get("components", {})]
        if missing:
            raise ValueError(f"{path}: bundle lacks components {missing}")
        retired = [k for k in ("single_frame_ensemble", "single_frame_refiner") if self.manifest.get(k) not in (None, "none")]
        retired += ["inpainter"] if "inpainter" in self.manifest.get("components", {}) else []
        if retired:
            raise ValueError(f"{path}: uses {retired}, retired in EyeClean 0.3.0; use EyeClean <= 0.2 for this bundle")
        sd = load_file(str(path), device=str(device))
        self.guard = EyeGuard(presence_head="guard.refl_ev.weight" in sd).to(device).eval()
        self.guard.load_state_dict({k[6:]: v for k, v in sd.items() if k.startswith("guard.")})
        mcfg = self.manifest["model"]
        # the five-frame CNN restorer: only --restore-mode temporal runs it, so a bundle for single-frame use
        # may leave it out (the released model does)
        self.restore = None
        if "restore" in self.manifest["components"]:
            self.restore = build_restore(mcfg, checkpoint=False).to(device).eval()
            self.restore.load_state_dict({k[8:]: v for k, v in sd.items() if k.startswith("restore.")})
        # The single-frame slot may hold a different architecture from the five-frame one: Round 21
        # retrains only this path as a transformer, and the temporal restorer stays the CNN it was
        # trained as.  `restore_single_arch` says which to construct; absent means the CNN, so every
        # bundle written before Round 21 loads unchanged.
        arch = mcfg.get("restore_single_arch", "cnn")
        if arch == "tx":
            from .eyerestore_tx import build_restore_tx
            self.restore_single = build_restore_tx(mcfg.get("restore_single_tx"), checkpoint=False).to(device).eval()
        elif arch == "cnn":
            self.restore_single = build_restore(mcfg, checkpoint=False).to(device).eval()
        else:
            raise ValueError(f"{path}: unknown restore_single_arch {arch!r}")
        self.restore_single.load_state_dict({k[15:]: v for k, v in sd.items() if k.startswith("restore_single.")})
        # optional v2 component: per-eye visible-reflection presence classifier on the frozen guard
        self.presence = None
        if "presence" in self.manifest.get("components", {}):
            self.presence = ReflectionPresence().to(device).eval()
            self.presence.load_state_dict({k[9:]: v for k, v in sd.items() if k.startswith("presence.")})
        for m in (self.guard, self.restore, self.restore_single, self.presence):
            if m is None:
                continue
            for p in m.parameters():
                p.requires_grad_(False)
        self.crop_size = int(mcfg["crop_size"])
        self.calibration = self.manifest.get("calibration", {})
        self.gate = self.manifest.get("gate", {})

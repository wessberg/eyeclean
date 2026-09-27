"""Model resolution: a model NAME (from registry.json) or a bundle PATH -> the file and how to run it.

Search order for a name: $EYECLEAN_MODEL_DIR (os.pathsep-separated), the user model directory
(%LOCALAPPDATA%/eyeclean/models on Windows, $XDG_DATA_HOME/eyeclean/models or ~/.local/share/eyeclean/models
elsewhere), then a checkout's models/ and <runs>/v2/checkpoints (eyeclean.paths.runs_dir).  With no model given, $EYECLEAN_MODEL, then
the registry default.  `fetch_model` downloads a registry model from its `url` and verifies its SHA-256.  A bundle path that matches a registry entry by SHA-256 inherits its settings
(e.g. restore_mode), so the promoted model needs no hidden flags.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

from ..io.errors import InputError
from ..paths import runs_dir

REGISTRY = Path(__file__).with_name("registry.json")
_CHECKOUT = Path(__file__).resolve().parents[3]


@dataclass
class ResolvedModel:
    path: Path
    name: str | None
    entry: dict | None
    sha256: str

    @property
    def restore_mode(self) -> str | None:
        return (self.entry or {}).get("restore_mode")

    def as_dict(self) -> dict:
        return {"name": self.name, "path": str(self.path), "sha256": self.sha256, "registry_entry": self.entry is not None}


def load_registry() -> dict:
    return json.loads(REGISTRY.read_text(encoding="utf-8"))


def user_model_dir() -> Path:
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "eyeclean" / "models"
    base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / "eyeclean" / "models"


def search_dirs() -> list[Path]:
    dirs = [Path(p) for p in os.environ.get("EYECLEAN_MODEL_DIR", "").split(os.pathsep) if p]
    dirs.append(user_model_dir())
    dirs += [d for d in (_CHECKOUT / "models", runs_dir() / "v2" / "checkpoints") if d.is_dir()]
    return dirs


def file_sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def locate(name: str, reg: dict | None = None) -> Path | None:
    reg = reg or load_registry()
    entry = reg["models"].get(name)
    fname = entry["file"] if entry else name
    for d in search_dirs():
        p = d / fname
        if p.is_file():
            return p
    return None


def resolve_model(spec: str | None) -> ResolvedModel:
    """spec: None (default), a registry name, or a path to a .safetensors bundle."""
    reg = load_registry()
    spec = spec or os.environ.get("EYECLEAN_MODEL") or reg["default"]
    p = Path(spec)
    if p.suffix == ".safetensors" or p.exists():
        if not p.is_file():
            raise InputError(f"model file {p} not found")
        sha = file_sha256(p)
        for name, entry in reg["models"].items():
            if entry.get("sha256") == sha:
                return ResolvedModel(p, name, entry, sha)
        return ResolvedModel(p, None, None, sha)
    entry = reg["models"].get(spec)
    if entry is None:
        raise InputError(f"unknown model {spec!r}: known {sorted(reg['models'])}, or pass a .safetensors path")
    path = locate(spec, reg)
    if path is None:
        raise InputError(f"model {spec!r} ({entry['file']}) not found in {[str(d) for d in search_dirs()]}; "
                         f"place it in one of them or set EYECLEAN_MODEL_DIR")
    sha = file_sha256(path)
    if entry.get("sha256") and sha != entry["sha256"]:
        raise InputError(f"model file {path} has SHA-256 {sha}, registry expects {entry['sha256']}")
    return ResolvedModel(path, spec, entry, sha)


def list_models() -> dict:
    reg = load_registry()
    out = {"schema": reg["schema"], "default": reg["default"], "search_dirs": [str(d) for d in search_dirs()], "models": {}}
    for name, entry in reg["models"].items():
        p = locate(name, reg)
        out["models"][name] = dict(entry, installed=str(p) if p else None)
    return out


def fetch_model(name: str | None = None, dest_dir: Path | None = None, reg: dict | None = None) -> dict:
    """Download registry model `name` (default: the registry default) from its `url` into `dest_dir` (default:
    the user model directory) and verify its SHA-256.  An installed file that already verifies is kept."""
    import shutil
    import urllib.request

    reg = reg or load_registry()
    name = name or reg["default"]
    entry = reg["models"].get(name)
    if entry is None:
        raise InputError(f"unknown model {name!r}: known {sorted(reg['models'])}")
    dest = Path(dest_dir) if dest_dir else user_model_dir()
    path = dest / entry["file"]
    installed = path if dest_dir else (locate(name, reg) or path)  # without --dir, any search folder counts
    if installed.is_file() and file_sha256(installed) == entry["sha256"]:
        return {"name": name, "path": str(installed), "sha256": entry["sha256"], "downloaded": False}
    if not entry.get("url"):
        raise InputError(f"model {name!r} has no download URL in this release; download {entry['file']} from the "
                         f"project's release page and place it in {dest}")
    dest.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    req = urllib.request.Request(entry["url"], headers={"User-Agent": "eyeclean-fetch-model"})
    with urllib.request.urlopen(req, timeout=120) as r, open(part, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 22)
    sha = file_sha256(part)
    if sha != entry["sha256"]:
        part.unlink()
        raise InputError(f"downloaded {entry['file']} has SHA-256 {sha}, registry expects {entry['sha256']}")
    os.replace(part, path)
    return {"name": name, "path": str(path), "sha256": sha, "downloaded": True}

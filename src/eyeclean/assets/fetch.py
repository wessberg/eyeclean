"""Download public, ungated third-party assets into the project cache with hash pinning.

Every asset records its source URL, license statement and SHA-256.  Assets whose publisher
provides a hash (Blender extensions platform, torch-hub style filename prefixes) are verified
against it; the others are pinned trust-on-first-use in ``third_party/assets.lock.json`` and
verified on every later run.  Nothing here uploads data or accepts gated terms.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
THIRD_PARTY = ROOT / "third_party"
LOCK_PATH = THIRD_PARTY / "assets.lock.json"


@dataclass(frozen=True)
class Asset:
    key: str
    url: str
    dest: str
    license: str
    provenance: str
    sha256: str | None = None  # full published hash
    sha256_prefix: str | None = None  # torch-hub style filename hash prefix
    required_for: tuple[str, ...] = ()
    mirrors: tuple[str, ...] = ()


ASSETS: tuple[Asset, ...] = (
    Asset(
        key="mpfb_extension",
        url="https://extensions.blender.org/download/sha256:4f0a879d64a39bf646fbf5f53601ac678855da329d650617dca5737548239a87/add-on-mpfb-v2.0.17.zip",
        dest="mpfb/add-on-mpfb-v2.0.17.zip",
        license="GPL-3.0-or-later (MPFB code; extensions.blender.org listing)",
        provenance="Official Blender Extensions platform listing id=mpfb version 2.0.17, blender_version_min 4.2.0",
        sha256="4f0a879d64a39bf646fbf5f53601ac678855da329d650617dca5737548239a87",
        required_for=("head_adapter",),
    ),
    Asset(
        key="makehuman_system_assets",
        url="https://files.makehumancommunity.org/asset_packs/makehuman_system_assets/makehuman_system_assets_cc0.zip",
        mirrors=("https://files2.makehumancommunity.org/asset_packs/makehuman_system_assets/makehuman_system_assets_cc0.zip",),
        dest="makehuman/makehuman_system_assets_cc0.zip",
        license="CC0-1.0 (per static.makehumancommunity.org/assets/assetpacks/makehuman_system_assets.html)",
        provenance="MakeHuman Community asset pack 'MakeHuman system assets'",
        required_for=("head_adapter",),
    ),
    Asset(
        key="skins01",
        url="https://files.makehumancommunity.org/asset_packs/skins01/skins01_cc0.zip",
        mirrors=("https://files2.makehumancommunity.org/asset_packs/skins01/skins01_cc0.zip",),
        dest="makehuman/skins01_cc0.zip",
        license="CC0-1.0 (per static.makehumancommunity.org/assets/assetpacks/skins01.html)",
        provenance="MakeHuman Community asset pack 'skins01' (natural female skins)",
        required_for=("head_adapter",),
    ),
    Asset(
        key="skins02",
        url="https://files.makehumancommunity.org/asset_packs/skins02/skins02_cc0.zip",
        mirrors=("https://files2.makehumancommunity.org/asset_packs/skins02/skins02_cc0.zip",),
        dest="makehuman/skins02_cc0.zip",
        license="CC0-1.0 (per static.makehumancommunity.org/assets/assetpacks/skins02.html)",
        provenance="MakeHuman Community asset pack 'skins02' (natural male skins)",
        required_for=("head_adapter",),
    ),
    Asset(
        key="s3fd_weights",
        url="https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth",
        dest="face_alignment/s3fd-619a316812.pth",
        license=(
            "No explicit weight license published; distributed by the BSD-3-Clause face-alignment project "
            "(github.com/1adrianb/face-alignment). Used locally for research evaluation only, never redistributed."
        ),
        provenance="face_alignment/detection/sfd/sfd_detector.py models_urls['s3fd'] (legacy pickle state dict)",
        sha256_prefix="619a316812",
        required_for=("detector",),
    ),
    Asset(
        key="fan2d_weights",
        url="https://www.adrianbulat.com/downloads/python-fan/2DFAN4-11f355bf06.pth.tar",
        dest="face_alignment/2DFAN4-11f355bf06.pth.tar",
        license=(
            "No explicit weight license published; distributed by the BSD-3-Clause face-alignment project. "
            "Used locally for research evaluation only, never redistributed."
        ),
        provenance="face_alignment/api.py models_urls['2DFAN-4'] (legacy pickle state dict)",
        sha256_prefix="11f355bf06",
        required_for=("detector",),
    ),
    Asset(
        key="face_alignment_source",
        url="https://codeload.github.com/1adrianb/face-alignment/zip/refs/heads/master",
        dest="face_alignment/face-alignment-master.zip",
        license="BSD-3-Clause (face-alignment code; LICENSE in archive)",
        provenance="Reference source for the ported S3FD/FAN network definitions (archive is not imported at runtime)",
        required_for=("detector",),
    ),
)


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_lock() -> dict:
    if LOCK_PATH.exists():
        return json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    return {"schema_version": "1.0", "assets": {}}


def save_lock(lock: dict) -> None:
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = LOCK_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(lock, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, LOCK_PATH)


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "eyeclean-asset-fetch/0.1"})
    with urllib.request.urlopen(req, timeout=120) as r, open(part, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 22)
    os.replace(part, dest)


def verify(asset: Asset, path: Path, lock: dict) -> str:
    digest = sha256_file(path)
    if asset.sha256 and digest != asset.sha256:
        raise RuntimeError(f"{asset.key}: SHA-256 mismatch {digest} != published {asset.sha256}")
    if asset.sha256_prefix and not digest.startswith(asset.sha256_prefix):
        raise RuntimeError(f"{asset.key}: SHA-256 {digest} does not start with filename prefix {asset.sha256_prefix}")
    pinned = lock["assets"].get(asset.key, {}).get("sha256")
    if pinned and pinned != digest and asset.key != "face_alignment_source":
        raise RuntimeError(f"{asset.key}: SHA-256 {digest} differs from lock {pinned}")
    return digest


def fetch(keys: list[str] | None = None, verbose: bool = True) -> dict:
    lock = load_lock()
    selected = [a for a in ASSETS if keys is None or a.key in keys]
    for asset in selected:
        dest = THIRD_PARTY / asset.dest
        if not dest.exists():
            last_err = None
            for url in (asset.url, *asset.mirrors):
                try:
                    t0 = time.time()
                    if verbose:
                        print(f"[fetch] {asset.key}: {url}", flush=True)
                    _download(url, dest)
                    if verbose:
                        print(f"[fetch] {asset.key}: {dest.stat().st_size/2**20:.1f} MiB in {time.time()-t0:.1f}s", flush=True)
                    last_err = None
                    break
                except Exception as e:  # try the mirror
                    last_err = e
                    print(f"[fetch] {asset.key}: failed {url}: {e}", flush=True)
            if last_err is not None:
                raise RuntimeError(f"could not download {asset.key}: {last_err}")
        digest = verify(asset, dest, lock)
        entry = {
            "url": asset.url,
            "path": str(dest.relative_to(ROOT)).replace("\\", "/"),
            "sha256": digest,
            "bytes": dest.stat().st_size,
            "license": asset.license,
            "provenance": asset.provenance,
            "hash_basis": "published" if asset.sha256 else ("filename_prefix+tofu" if asset.sha256_prefix else "trust_on_first_use"),
            "required_for": list(asset.required_for),
        }
        if asset.key == "face_alignment_source" and dest.exists():
            with zipfile.ZipFile(dest) as z:
                entry["archive_root"] = z.namelist()[0].split("/")[0]
        prev = lock["assets"].get(asset.key)
        if prev and "fetched_at" in prev and prev.get("sha256") == digest:
            entry["fetched_at"] = prev["fetched_at"]
        else:
            entry["fetched_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        lock["assets"][asset.key] = entry
        save_lock(lock)
        if verbose:
            print(f"[fetch] {asset.key}: ok sha256={digest}", flush=True)
    return lock


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    fetch(argv or None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

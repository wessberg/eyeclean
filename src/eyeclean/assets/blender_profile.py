"""Install MPFB and the CC0 MakeHuman asset packs into the project-local Blender profile.

The user's own Blender configuration is never touched: everything goes under
third_party/blender_user (BLENDER_USER_RESOURCES), used only by EyeClean's headless renders.
"""

from __future__ import annotations

import json
import shutil
import sys
import zipfile
from pathlib import Path

from .fetch import THIRD_PARTY, fetch

PROFILE = THIRD_PARTY / "blender_user"
EXT_DIR = PROFILE / "extensions" / "user_default" / "mpfb"
DATA_DIR = PROFILE / "extensions" / ".user" / "user_default" / "mpfb" / "data"
PACKS = ("makehuman_system_assets", "skins01", "skins02")


def install(force: bool = False) -> dict:
    lock = fetch(["mpfb_extension", *PACKS])["assets"]
    marker = PROFILE / "eyeclean_profile.json"
    want = {k: lock[k]["sha256"] for k in ("mpfb_extension", *PACKS)}
    if marker.exists() and not force:
        have = json.loads(marker.read_text(encoding="utf-8"))
        if have.get("assets") == want and (EXT_DIR / "blender_manifest.toml").exists() and (DATA_DIR / "skins").exists():
            return {"status": "already_installed", "profile": str(PROFILE)}
    if EXT_DIR.exists():
        shutil.rmtree(EXT_DIR)
    EXT_DIR.mkdir(parents=True)
    with zipfile.ZipFile(THIRD_PARTY / lock["mpfb_extension"]["path"].split("third_party/", 1)[1]) as z:
        z.extractall(EXT_DIR)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for k in PACKS:
        with zipfile.ZipFile(THIRD_PARTY / lock[k]["path"].split("third_party/", 1)[1]) as z:
            z.extractall(DATA_DIR)
    marker.write_text(json.dumps({"assets": want, "licenses": {k: lock[k]["license"] for k in want}}, indent=2), encoding="utf-8")
    return {"status": "installed", "profile": str(PROFILE), "mpfb": str(EXT_DIR), "data": str(DATA_DIR)}


if __name__ == "__main__":
    print(json.dumps(install("--force" in sys.argv), indent=2))

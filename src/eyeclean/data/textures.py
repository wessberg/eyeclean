"""Write per-identity procedural iris/sclera textures (CUDA synthesis -> FLOAT EXR)."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from ..io.exr import PIXEL_HALF, write_exr
from ..procedural.iris import generate_iris, generate_sclera


def _valid_exr(path: Path) -> bool:
    """Existence is not enough after a crash: verify every scanline chunk leader (metadata)."""
    import struct

    from ..io.exr import probe_exr

    if not path.exists():
        return False
    try:
        info = probe_exr(path)
        h = info["parts"][0]
        b = path.read_bytes()
        offs = struct.unpack_from(f"<{h.height}Q", b, info["header_end"])
        return all(struct.unpack_from("<i", b, o)[0] == h.data_window[1] + k for k, o in enumerate(offs))
    except Exception:
        return False


def write_identity_textures(identity: dict, out_dir: Path, device: torch.device) -> dict:
    """identity['iris'][side], identity['sclera'][side] are parameter dicts. Returns paths per side."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for side in ("L", "R"):
        pa = out_dir / f"{side}_iris_albedo.exr"
        ph = out_dir / f"{side}_iris_height.exr"
        sa = out_dir / f"{side}_sclera_albedo.exr"
        sr = out_dir / f"{side}_sclera_rough.exr"
        if not all(_valid_exr(p) for p in (pa, ph, sa, sr)):
            iris = generate_iris(identity["iris"][side], device=device)
            scl = generate_sclera(identity["sclera"][side], device=device)
            # Blender samples v=0 at the bottom row; our row 0 is rho=0 (pupil margin) -> flip rows
            a = torch.flip(iris["albedo"], dims=[1])
            h = torch.flip(iris["height"][0], dims=[0])
            s = torch.flip(scl["albedo"], dims=[1])
            r = torch.flip(scl["roughness"][0], dims=[0])
            # HALF precision is ample for albedo/relief textures and keeps ~200 identities small
            write_exr(pa, {"R": a[0], "G": a[1], "B": a[2]}, pixel_type=PIXEL_HALF)
            write_exr(ph, {"R": h, "G": h, "B": h}, pixel_type=PIXEL_HALF)
            write_exr(sa, {"R": s[0], "G": s[1], "B": s[2]}, pixel_type=PIXEL_HALF)
            write_exr(sr, {"R": r, "G": r, "B": r}, pixel_type=PIXEL_HALF)
        paths[side] = {"iris_albedo": str(pa), "iris_height": str(ph), "sclera_albedo": str(sa), "sclera_roughness": str(sr)}
    (out_dir / "params.json").write_text(json.dumps({"iris": identity["iris"], "sclera": identity["sclera"]}, indent=1), encoding="utf-8")
    return paths

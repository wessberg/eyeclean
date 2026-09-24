"""Human-inspection contact sheets of the rendered pairs (saved regardless of automated checks)."""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from torchvision.io import encode_jpeg

from ..color import transforms as CT
from ..gpu.audit import boundary


def _disp(x: torch.Tensor, k: float) -> torch.Tensor:
    return (CT.srgb_inverse_eotf(torch.clamp(x * k, 0, 1)) * 255).round().to(torch.uint8)


def sheets(run_dir: Path, per_split: int = 8, seed: int = 0) -> dict:
    from safetensors.torch import load_file

    dev = torch.device("cuda")
    man = json.loads((run_dir / "manifests" / "dataset_manifest.json").read_text(encoding="utf-8"))
    out = run_dir / "validation" / "inspection"
    out.mkdir(parents=True, exist_ok=True)
    r = random.Random(seed)
    files = []
    for split in ("train", "validation", "test"):
        shots = [s for s in man["shots"] if s["split"] == split and s["kind"] != "demo"]
        rows = []
        for s in r.sample(shots, min(per_split, len(shots))):
            t = load_file(str(run_dir / "data" / "cache" / s["cache"]))
            k = r.randrange(t["clean"].shape[0])
            c, rf = t["clean"][k].to(dev).float(), t["reflection"][k].to(dev).float()
            lab = t["labels"][k].to(dev).float() / 255
            lum = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
            e = 0.18 / float(torch.quantile(lum.flatten()[::8], 0.5).clamp(min=1e-5))
            src = c if s["negative"] else c + rf
            lab_a = (lab[0:3].clamp(0, 1) * 255).to(torch.uint8)
            lab_b = (lab[3:6].clamp(0, 1) * 255).to(torch.uint8)
            row = torch.cat([_disp(src, e), _disp(c, e), _disp(rf, 8 * e), lab_a, lab_b], 2)
            rows.append(torch.nn.functional.interpolate(row[None].float(), scale_factor=0.5, mode="area")[0].to(torch.uint8))
        if not rows:
            continue
        W = max(x.shape[2] for x in rows)
        rows = [torch.nn.functional.pad(x, (0, W - x.shape[2])) for x in rows]
        sheet = torch.cat(rows, 1)
        p = out / f"{split}_pairs.jpg"
        data = encode_jpeg(sheet.contiguous(), quality=88)
        with boundary("preview_png"):
            p.write_bytes(bytes(data.cpu().numpy()))
        files.append(str(p))
    (out / "README.txt").write_text("Columns: source (clean + designated reflection; clean only for negatives) | clean target | "
                                    "reflection x8 | labels A (R=ocular surface, G=skin, B=lashes/brows) | labels B (R=iris, "
                                    "G=pupil through the refractive cornea, B=tear meniscus).\n", encoding="utf-8")
    return {"files": files}


def dependency_manifest(run_dir: Path, doctor: dict) -> dict:
    import hashlib

    root = Path(__file__).resolve().parents[3]
    lock = root / "uv.lock"
    assets = json.loads((root / "third_party" / "assets.lock.json").read_text(encoding="utf-8"))
    conv = {}
    for n in ("s3fd", "fan2d"):
        p = root / "third_party" / "face_alignment" / "converted" / f"{n}.provenance.json"
        if p.exists():
            conv[n] = json.loads(p.read_text(encoding="utf-8"))
    dep = {"uv_lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest() if lock.exists() else None,
           "versions": doctor.get("versions", {}), "driver": doctor.get("gpu", {}).get("driver_version"),
           "gpu": doctor.get("gpu", {}).get("name"), "assets": assets["assets"], "converted_weights": conv}
    p = run_dir / "manifests" / "dependencies.json"
    p.write_text(json.dumps(dep, indent=1), encoding="utf-8")
    lic = {k: v["license"] for k, v in assets["assets"].items()}
    lic.update({"project_python": "MIT (LICENSE)", "project_blender_scripts": "GPL-3.0-or-later (LICENSE)"})
    (run_dir / "manifests" / "licenses.json").write_text(json.dumps(lic, indent=1), encoding="utf-8")
    return {"dependencies": str(p), "licenses": str(run_dir / "manifests" / "licenses.json")}

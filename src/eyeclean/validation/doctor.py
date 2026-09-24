"""`eyeclean doctor`: inspect the workstation and run executable smoke tests (evidence, not claims)."""

from __future__ import annotations

import json
import os
import platform
import shutil
import sys
import tempfile
import time
from pathlib import Path

import torch

from ..gpu.runtime import cuda_kernel_smoke, device_info, require_cuda

ROOT = Path(__file__).resolve().parents[3]


def versions() -> dict:
    import importlib.metadata as md

    v = {"python": sys.version.split()[0], "platform": platform.platform()}
    for pkg in ("torch", "torchvision", "pynvvideocodec", "nvidia-cuda-runtime-cu12", "safetensors", "numpy", "pyyaml", "jsonschema",
                "openexr", "colour-science", "av", "eyeclean"):
        try:
            v[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            v[pkg] = None
    v["torch_cuda"] = torch.version.cuda
    v["cudnn"] = str(torch.backends.cudnn.version())
    try:
        from ..render.blender import find_blender

        b = find_blender()
        v["blender"] = f"{b.version} (hash {b.build_hash})"
    except Exception as e:  # reported, not fatal for version listing
        v["blender"] = f"not found: {e}"[:200]
    try:
        from ..gpu.runtime import ensure_cuda12_runtime_on_dll_path

        ensure_cuda12_runtime_on_dll_path()
        import PyNvVideoCodec

        v["pynvvideocodec_runtime"] = getattr(PyNvVideoCodec, "__version__", None)
    except Exception as e:
        v["pynvvideocodec_runtime"] = f"import failed: {e}"[:200]
    lock = ROOT / "third_party" / "assets.lock.json"
    if lock.exists():
        a = json.loads(lock.read_text(encoding="utf-8"))["assets"]
        v["mpfb_extension"] = "2.0.17 sha256:" + a["mpfb_extension"]["sha256"][:16] if "mpfb_extension" in a else None
    return v


def check_optix_render(out_dir: Path) -> dict:
    from ..render.blender import find_blender, run_script

    b = find_blender()
    script = out_dir / "doctor_optix.py"
    script.write_text(
        "import bpy, sys, json, time\n"
        "sys.path.insert(0, r'" + str(ROOT / "blender") + "')\n"
        "from common import setup_optix, log\n"
        "sc = bpy.context.scene\n"
        "dev = setup_optix(sc)\n"
        "log('OptiX devices', dev)\n"
        "sc.cycles.samples = 16; sc.render.resolution_x = 96; sc.render.resolution_y = 64\n"
        "sc.render.image_settings.file_format = 'OPEN_EXR'; sc.render.image_settings.exr_codec = 'NONE'\n"
        "sc.render.filepath = sys.argv[-1]\n"
        "t = time.time(); bpy.ops.render.render(write_still=True)\n"
        "print('EC_RENDER_OK', json.dumps({'seconds': time.time()-t, 'device': sc.cycles.device, 'enabled': dev['enabled']}))\n",
        encoding="utf-8")
    import subprocess

    from ..render.blender import env

    log = out_dir / "doctor_optix.log"
    proc = subprocess.run([b.path, "-b", "--factory-startup", "--python-exit-code", "1", "--python", str(script), "--",
                           str(out_dir / "doctor_optix.exr")], capture_output=True, text=True, env=env(), timeout=600)
    log.write_text(proc.stdout + proc.stderr, encoding="utf-8")
    ok_line = [l for l in proc.stdout.splitlines() if l.startswith("EC_RENDER_OK")]
    info = json.loads(ok_line[0].split(" ", 1)[1]) if ok_line else None
    return {"blender": b.as_dict(), "passed": bool(proc.returncode == 0 and info and info["device"] == "GPU" and info["enabled"]),
            "render": info, "log": str(log)}


def check_codec_roundtrip(out_dir: Path, dev) -> dict:
    from ..io.video import VideoReader, VideoWriter, planes_to_rgb

    res = {}
    ok = True
    for codec, bits in (("h264", 8), ("hevc", 10)):
        p = out_dir / f"doctor_{codec}_{bits}.mp4"
        W, H = 640, 360
        w = VideoWriter(str(p), W, H, 24.0, codec, bits)
        frames = []
        pts = [0, 3750, 7500, 12000, 15750, 19500, 23250, 27000]
        for i, t in enumerate(pts):
            ys, xs = torch.meshgrid(torch.linspace(0, 1, H, device=dev), torch.linspace(0, 1, W, device=dev), indexing="ij")
            f = torch.stack([xs, ys, torch.full_like(xs, (i % 4) / 4)]) * 0.8 + 0.1
            frames.append(f)
            w.write(f, t)
        w.close()
        r = VideoReader(str(p), dev)
        got, err = [], []
        for k, d in enumerate(r):
            got.append(d.pts)
            err.append(float((planes_to_rgb(d, False, "bt709") - frames[k]).abs().mean()))
        c_ok = got == pts and max(err) < 0.02 and r.info.bit_depth == bits
        ok &= c_ok
        res[f"{codec}_{bits}bit"] = {"frames_in": len(pts), "frames_out": len(got), "timestamps_preserved": got == pts,
                                     "mean_abs_rgb_error": round(sum(err) / len(err), 5), "decoded_bit_depth": r.info.bit_depth,
                                     "surface": r.info.surface_format, "passed": c_ok}
    res["passed"] = ok
    return res


def check_exr(out_dir: Path, dev) -> dict:
    from ..io.exr import read_exr, write_exr

    H, W = 41, 67
    g = torch.Generator(device=dev)
    g.manual_seed(3)
    ch = {n: torch.randn((H, W), device=dev, generator=g) * 100 for n in ("R", "G", "B", "mask.Y")}
    ch["R"][0, 0] = -0.0
    ch["G"][1, 1] = float("inf")
    ch["B"][2, 2] = float("nan")
    p = out_dir / "doctor_exr.exr"
    write_exr(p, ch, display_size=(3840, 2160), origin=(1500, 900))
    img = read_exr(p, dev)
    exact = all(bool((img.channels[n].view(torch.int32) == ch[n].view(torch.int32)).all()) for n in ch)
    official = None
    try:
        import numpy as np
        import OpenEXR

        with OpenEXR.File(str(p), separate_channels=True) as f:
            official = all(np.array_equal(f.channels()[n].pixels.view(np.int32), ch[n].cpu().numpy().view(np.int32)) for n in ch)
            dwin = f.header()["dataWindow"]
            official &= (int(dwin[0][0]), int(dwin[0][1])) == (1500, 900)
    except ImportError:
        official = None
    return {"bit_exact_own_reader": exact, "official_openexr_agrees": official, "passed": bool(exact and official is not False)}


def check_color(dev) -> dict:
    from ..color import transforms as CT

    try:
        import colour
        import numpy as np
    except ImportError:
        return {"passed": None, "reason": "colour-science not installed (dev group)"}
    # domains: OETF/encode on scene-linear >= 0 (negatives are extended linearly by design and not
    # defined by the references), decode/EOTF on [0, 1] code values; relative error where values grow
    x = torch.linspace(0.0, 12.0, 4001, device=dev, dtype=torch.float32)
    xs = x.cpu().numpy().astype(np.float64)
    errs = {}
    # BT.709's rounded constants make the two branches meet discontinuously at L=0.018 (V=0.081);
    # a sample exactly on the breakpoint can fall on different branches in FP32 vs float64
    keep = np.abs(xs - 0.018) > 1e-6
    errs["bt709_oetf"] = float(np.abs(CT.bt709_oetf(x).cpu().numpy() - colour.models.oetf_BT709(xs))[keep].max())
    vv = np.linspace(0, 1, 4001)
    keep_v = np.abs(vv - 0.081) > 1e-6
    errs["bt709_inverse_oetf"] = float(np.abs(CT.bt709_inverse_oetf(torch.tensor(vv, device=dev, dtype=torch.float32)).cpu().numpy()
                                              - colour.models.oetf_inverse_BT709(vv))[keep_v].max())
    v = torch.linspace(0.0, 1.0, 4001, device=dev)
    vd = colour.models.log_decoding_VLog(v.cpu().numpy().astype(np.float64))
    errs["vlog_decode_rel"] = float((np.abs(CT.vlog_decode(v).cpu().numpy() - vd) / np.maximum(np.abs(vd), 1e-3)).max())
    errs["vlog_encode"] = float(np.abs(CT.vlog_encode(x).cpu().numpy() - colour.models.log_encoding_VLog(xs)).max())
    errs["srgb_eotf"] = float(np.abs(CT.srgb_eotf(v).cpu().numpy() - colour.models.eotf_sRGB(v.cpu().numpy().astype(np.float64))).max())
    rgb = torch.rand((3, 64, 64), device=dev) * 4 - 0.5
    cs = colour.RGB_COLOURSPACES
    for name, m, src in (("709_to_ap1", CT.M_709_TO_AP1, "ITU-R BT.709"), ("vgamut_to_ap1", CT.M_VGAMUT_TO_AP1, "V-Gamut")):
        ref = colour.RGB_to_RGB(np.moveaxis(rgb.cpu().numpy().astype(np.float64), 0, -1), cs[src], cs["ACEScg"], "Bradford")
        got = np.moveaxis(CT.apply_matrix(rgb, m).cpu().numpy(), 0, -1)
        errs[name] = float(np.abs(got - ref).max())
    passed = all(v < 1e-5 for v in errs.values())
    return {"max_abs_error": errs, "passed": bool(passed), "reference": f"colour-science {colour.__version__} (offline test only)"}


def run_doctor(policy: str = "strict", out: Path | None = None, full: bool = True) -> dict:
    t0 = time.time()
    dev = require_cuda(policy)
    di = device_info(dev)
    tmp = Path(tempfile.mkdtemp(prefix="eyeclean_doctor_"))
    rep = {"host": {"os": platform.platform(), "cpu": platform.processor(), "python": sys.version.split()[0],
                    "free_disk_gib": round(shutil.disk_usage(ROOT).free / 2**30, 1)},
           "gpu": di.__dict__, "versions": versions(), "checks": {}}
    rep["checks"]["cuda_kernel"] = cuda_kernel_smoke(dev)
    rep["checks"]["float_exr_roundtrip"] = check_exr(tmp, dev)
    rep["checks"]["cuda_color_reference_tests"] = check_color(dev)
    if full:
        rep["checks"]["optix_render_device"] = check_optix_render(tmp)
        rep["checks"]["nvdec_nvenc_supported_profile_roundtrip"] = check_codec_roundtrip(tmp, dev)
    rep["passed"] = all(c.get("passed") for c in rep["checks"].values() if c.get("passed") is not None)
    rep["seconds"] = round(time.time() - t0, 1)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    return rep

"""`eyeclean apply-patches`: reference compositor for the exported signed corrections.

Reads the process run's source_manifest.json (recorded colour interpretation and source hashes),
decodes the unchanged source through the same declared GPU transform, keeps every pixel outside
the hard support bit-identical, adds the signed delta once inside it (the delta is already gated;
no second alpha multiplication) and writes uncompressed full-frame FLOAT EXR masters plus a
verification report.  Output of third-party grades/scaling after this step is outside the
unchanged-pixel guarantee.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import torch

from ..gpu.audit import GpuAudit, audited
from ..gpu.runtime import require_cuda
from ..io.errors import InputError
from ..io.exr import read_exr, write_exr
from .compositor import apply_signed_correction, changed_bits_outside
from .frames import FrameSource
from .process import hash_files, input_files


def run_apply(input_path: str, patches: Path, out: Path, gpu_policy: str = "strict", max_frames: int | None = None) -> dict:
    t0 = time.time()
    dev = require_cuda(gpu_policy)
    run_dir = patches.parent
    man_p = run_dir / "source_manifest.json"
    idx_p = patches / "index.json"
    if not man_p.exists() or not idx_p.exists():
        raise InputError(f"{patches}: expected index.json and ../source_manifest.json from `eyeclean process`")
    man = json.loads(man_p.read_text(encoding="utf-8"))
    index = {e["frame"]: e for e in json.loads(idx_p.read_text(encoding="utf-8"))["frames"]}
    files = input_files(input_path)
    hashes = hash_files(files)
    expected = man["sha256_before"]
    mism = [f for f in hashes if expected.get(f) != hashes[f]]
    if mism or len(hashes) != len(expected):
        raise InputError(f"source does not match the processed source recorded in {man_p} (hash mismatch: {mism[:3]})")
    out = out.resolve()
    if any(os.path.normcase(str(Path(f).parent.resolve())) == os.path.normcase(str(out)) for f in files):
        raise InputError("output directory collides with the input location")
    (out / "master").mkdir(parents=True, exist_ok=True)
    interp = man["interpretation"]
    timing = man.get("sequence_timing") or {}
    fps = timing.get("fps") if timing.get("source") == "cli_fps" else (None if timing.get("source") else None)
    ts = timing.get("path") if timing.get("source") == "timestamps_file" else None
    if not man.get("video") and fps is None and ts is None:
        fps = 1.0  # frame order only; timing is irrelevant for compositing
    src = FrameSource(input_path, interp, dev, fps, ts, max_frames)
    audit = GpuAudit(strict=True)
    frames, outside_bad, max_rt, n_patches = 0, 0, 0.0, 0
    with audited(audit):
        for fr in src:
            e = index.get(fr.index)
            outp = fr.reference.clone()
            support = torch.zeros(fr.reference.shape[1:], dtype=torch.bool, device=dev)
            for eye in (e or {}).get("eyes", []):
                if "patch" not in eye:
                    continue
                pimg = read_exr(run_dir / eye["patch"], dev)
                mimg = read_exr(run_dir / eye["mask"], dev)
                rimg = read_exr(run_dir / eye["replacement"], dev)
                dw = pimg.header.data_window
                box = (dw[0], dw[1], dw[2] + 1, dw[3] + 1)
                delta = torch.stack([pimg.channels[c].float() for c in "RGB"])
                hard = mimg.channels["hard"].float() > 0.5
                apply_signed_correction(fr.reference, delta, hard, box, outp)
                support[box[1]:box[3], box[0]:box[2]] |= hard
                repl = torch.stack([rimg.channels[c].float() for c in "RGB"])
                roi = outp[:, box[1]:box[3], box[0]:box[2]]
                if bool(hard.any()):
                    max_rt = max(max_rt, float((roi - repl).abs()[:, hard].max()))
                n_patches += 1
            outside_bad += changed_bits_outside(fr.reference, outp, support)
            ch = {"R": outp[0], "G": outp[1], "B": outp[2]}
            if fr.alpha is not None:
                ch["A"] = fr.alpha.float()
            write_exr(out / "master" / f"frame_{fr.index:06d}.exr", ch, attributes={"eyeclean/semantics": "apply-patches full-frame master",
                                                                                    "eyeclean/working_color": interp["working_color"]})
            frames += 1
    after = hash_files(files)
    rep = {"report_kind": "apply_patches", "frames": frames, "patches_applied": n_patches, "outside_support_changed_bits": outside_bad,
           "max_abs_diff_vs_process_replacement": max_rt, "source_unchanged": after == hashes,
           "gpu_audit": audit.summary(), "interpretation": interp, "seconds": round(time.time() - t0, 2),
           "verified": outside_bad == 0 and after == hashes and frames == len(man["timestamps_s"]) and max_rt < 1e-4,
           "note": "signed deltas re-applied in FP32 can differ from the process output by float rounding; the replacement "
                   "patches carry the exact process output values"}
    (out / "apply_report.json").write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    return rep

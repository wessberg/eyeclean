"""`eyeclean process`: run the trained pipeline on a video file or EXR sequence.

External contract (docs/INTEGRATION.md): job files (eyeclean.job/1), progress events (eyeclean.progress/1),
RUN_REPORT.json / patches/index.json (eyeclean.process/1).  The numerical path is unchanged by the contract:
progress, cancellation and reporting only observe the two passes.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Callable

import torch

from ..gpu.audit import GpuAudit, audited, profile_stage, static_hot_path_scan
from ..gpu.runtime import PeakMemoryMonitor, configure_memory_budget, device_info, require_cuda
from ..io.errors import InputError
from ..io.exr import read_exr
from ..io.sequence import resolve_sequence
from ..io.sidecar import find_sidecar, load_sidecar, resolve_interpretation
from ..io.stills import is_still
from ..models.bundle import Bundle
from ..models.registry import resolve_model
from ..version import PROCESS_REPORT_SCHEMA, __version__
from .exports import Exporter
from .frames import FrameSource, is_video
from .pipeline import Pipeline
from .progress import CancelToken, Progress

ROOT = Path(__file__).resolve().parents[3]
PRIVATE_DIRS = ("inputs", "own-footage")  # user footage folders: read-only for every EyeClean command


@dataclass
class ProcessArgs:
    input: str
    checkpoint: str | None              # a registry model name, a .safetensors path, or None for the default model
    input_color: str | None
    mode: str
    allow_synthesis: bool
    output: str
    gpu_policy: str = "strict"
    export: list = field(default_factory=lambda: ["patches", "preview", "report"])
    input_range: str | None = None
    working_color: str | None = None
    fps: float | str | Fraction | None = None   # constant rate for image sequences: 24, 29.97 or exactly "60000/1001"
    timestamps: str | None = None
    strength: float = 1.0
    max_frames: int | None = None
    max_region: str | None = None
    full_frame_master: bool = False
    restore_mode: str = "auto"          # auto: the registry's setting; else "single" for a bundle without a five-frame restorer, else "temporal"
    forbid_read: list = field(default_factory=list)
    profile: bool = True
    config_working: str = "linear-acescg"
    max_faces: int = 4
    min_iris_px: float = 8.0
    buffer_frames: int = 7
    exr_compression: str = "none"       # output EXRs: none | zips | zip (lossless)
    progress: Callable[[dict], None] | None = None   # receives eyeclean.progress/1 events
    cancel: CancelToken | None = None
    job: dict | None = None             # the job as submitted, recorded in RUN_REPORT.json


def parse_fps(v) -> Fraction | None:
    """24 | 29.97 | "30000/1001" -> an exact Fraction (floats are taken at their decimal value)."""
    if v is None or v == "":
        return None
    if isinstance(v, Fraction):
        f = v
    elif isinstance(v, (int, float)):
        f = Fraction(str(v))
    else:
        try:
            f = Fraction(str(v).strip())
        except (ValueError, ZeroDivisionError):
            raise InputError(f"--fps {v!r}: expected a number or a ratio such as 60000/1001") from None
    if f <= 0:
        raise InputError(f"--fps must be positive, got {v!r}")
    return f


def input_files(path: str) -> list[str]:
    if is_video(path) or (Path(path).is_file() and is_still(path)):
        return [path]
    return [f for _, f in resolve_sequence(path)]


def hash_files(files: list[str]) -> dict:
    out = {}
    for f in files:
        h = hashlib.sha256()
        with open(f, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 22), b""):
                h.update(chunk)
        out[os.path.abspath(f)] = h.hexdigest()
    return out


class ReadGuard:
    """Audit hook: records every file opened during prediction and refuses forbidden roots."""

    def __init__(self, forbidden: list[str]):
        self.forbidden = [os.path.normcase(os.path.abspath(p)) for p in forbidden]
        self.opened: list[str] = []
        self.violations: list[str] = []
        self.active = False

    def __call__(self, event, args):
        if not self.active or event != "open" or not args:
            return
        p = args[0]
        if not isinstance(p, (str, bytes, os.PathLike)):
            return
        ap = os.path.normcase(os.path.abspath(os.fsdecode(p)))
        self.opened.append(ap)
        for root in self.forbidden:
            if ap.startswith(root):
                self.violations.append(ap)
                raise PermissionError(f"prediction process attempted to read forbidden path {ap}")


def _check_paths(a: ProcessArgs, files: list[str]) -> Path:
    out = Path(a.output).resolve()
    ins = {os.path.normcase(os.path.abspath(f)) for f in files}
    for private in PRIVATE_DIRS:
        pdir = os.path.normcase(str((ROOT / private).resolve()))
        if os.path.normcase(str(out)).startswith(pdir):
            raise InputError(f"refusing to write outputs under {ROOT / private} (private footage is never modified)")
    if os.path.normcase(str(out)) in ins or any(os.path.normcase(str(Path(f).parent.resolve())) == os.path.normcase(str(out)) for f in files):
        raise InputError(f"output directory {out} collides with the input location; choose a separate directory")
    return out


def _observed(frames, progress, cancel: CancelToken | None, stage: str, total: int | None):
    """Yield the source's frames unchanged; report each one and honour cancellation between frames."""
    done = 0
    for fr in frames:
        if cancel is not None:
            cancel.check(f"{stage} before frame {fr.index}")
        yield fr
        done += 1
        progress("frame", stage=stage, frame=fr.index, done=done, total=total)


def _quiet(*_a, **_k) -> None:
    return None


def run_process(a: ProcessArgs) -> dict:
    t0 = time.time()
    progress = Progress([a.progress])
    cancel = a.cancel
    fps_exact = parse_fps(a.fps)
    fps = float(fps_exact) if fps_exact is not None else None
    if a.exr_compression not in ("none", "zips", "zip"):
        raise InputError("--exr-compression must be none, zips or zip")
    if a.restore_mode not in ("auto", "temporal", "single"):
        raise InputError("--restore-mode must be auto, temporal or single")
    dev = require_cuda(a.gpu_policy)
    budget = configure_memory_budget(13.0, 2.0)
    if a.mode == "hybrid" and not a.allow_synthesis:
        raise InputError("--mode hybrid synthesizes hidden iris/pupil content and requires --allow-synthesis")
    if a.allow_synthesis and a.mode != "hybrid":
        raise InputError("--allow-synthesis is only meaningful with --mode hybrid")
    if not (0.0 <= a.strength <= 1.0):
        raise InputError("--strength must be in [0, 1]")
    if not os.path.exists(a.input) and not any(c in a.input for c in "%#*"):
        raise InputError(f"input {a.input} not found")
    files = input_files(a.input)
    out = _check_paths(a, files)
    out.mkdir(parents=True, exist_ok=True)
    model = resolve_model(a.checkpoint)
    restore_mode, restore_src = a.restore_mode, "argument"
    if restore_mode == "auto":
        restore_mode, restore_src = (model.restore_mode, "model registry") if model.restore_mode else ("temporal", "default")
    sidecar_path = find_sidecar(a.input, files[0] if not is_video(a.input) else None)
    sidecar = load_sidecar(sidecar_path)
    meta_range = meta_matrix = None
    video = is_video(a.input)
    vinfo = None
    if video:
        from ..io.video import check_supported, probe

        vinfo = probe(a.input)
        check_supported(vinfo)
        meta_range, meta_matrix = vinfo.color_range_meta, vinfo.color_space_meta
    still = (not video) and Path(a.input).is_file() and is_still(a.input)
    still_warn = []
    if still:
        # a photo carries its own colour (ICC / cICP, or the untagged-sRGB convention); it is read from the file
        working = a.working_color or a.config_working
        interp = {"input_color": "auto (from the file)", "working_color": working, "input_range": "full", "ycbcr_matrix": "n/a",
                  "sources": {"input_color": "file", "working_color": "cli" if a.working_color else "config", "input_range": "file"}}
        if a.input_color:
            still_warn.append(f"--input-color {a.input_color} ignored: a still image's colour is read from the file")
    else:
        interp = resolve_interpretation(is_video=video, cli_color=a.input_color, cli_range=a.input_range, cli_working=a.working_color,
                                        sidecar=sidecar, meta_range=meta_range, meta_matrix=meta_matrix, config_working=a.config_working)
    if fps is None and not video and sidecar.get("fps") is not None:
        fps_exact = parse_fps(sidecar.get("fps"))
        fps = float(fps_exact)
    ts = a.timestamps or (sidecar.get("timestamps") if not video else None)
    hashes_before = hash_files(files)
    bundle = Bundle(model.path, dev)
    if restore_src == "default" and bundle.restore is None:
        restore_mode, restore_src = "single", "bundle has no five-frame restorer"
    if restore_mode == "temporal" and bundle.restore is None:
        raise InputError(f"model {model.name or model.path} has no five-frame restorer; use --restore-mode single (or auto)")
    src1 = FrameSource(a.input, interp, dev, fps, ts, a.max_frames)
    total = len(files) if not video else None
    if a.max_frames is not None and total is not None:
        total = min(total, a.max_frames)
    progress("start", input=os.path.abspath(a.input), output=str(out), eyeclean_version=__version__,
             model=dict(model.as_dict(), restore_mode=restore_mode), frames_expected=total)
    max_region = None
    if a.max_region:
        img = read_exr(a.max_region, dev)
        ch = next(iter(img.channels.values()))
        max_region = torch.zeros((src1.height, src1.width), device=dev)
        dw = img.header.data_window
        max_region[dw[1]:dw[3] + 1, dw[0]:dw[2] + 1] = ch.float()
    pipe = Pipeline(bundle, dev, mode=a.mode, strength=a.strength, working_color=interp["working_color"], max_faces=a.max_faces,
                    min_iris_px=a.min_iris_px, buffer_frames=a.buffer_frames, temporal=(restore_mode == "temporal"),
                    max_region=max_region)
    exports = set(a.export)
    if still:
        interp["still"] = src1.still.record()
        exports.discard("preview")      # no video preview for a still; the photo itself is written instead
        exports.add("photo")
        if src1.still.colour_assumed:
            still_warn.append(f"colour: {src1.still.colour}")
        if src1.still.has_gain_map:
            still_warn.append("the file is an Ultra HDR / gain-map JPEG: its SDR base image was processed; the output has no gain map")
    exporter = Exporter(out, src1.width, src1.height, src1.nominal_fps, exports, interp["working_color"], src1.time_base,
                        master=a.full_frame_master, exr_compression=a.exr_compression, still=src1.still)
    state = {"analysis_frames": None}

    def on_frame(fr, res, patches, recs):
        exporter.on_frame(fr, res, patches, recs)
        eyes = exporter.index[-1]["eyes"]
        progress("frame", stage="render", frame=fr.index, done=len(exporter.index), total=total or state["analysis_frames"],
                 eyes_repaired=sum(1 for e in eyes if e["status"] == "repaired"), eyes_tracked=len(eyes))
    guard = ReadGuard(a.forbid_read)
    sys.addaudithook(guard)
    audit = GpuAudit(strict=True)
    mon = PeakMemoryMonitor(dev)
    prof_dir = out / "profiles" if (a.profile and "report" in a.export) else None
    guard.active = True
    try:
        with audited(audit):
            progress("stage", stage="analysis")
            with profile_stage("process_analysis", prof_dir, enabled=prof_dir is not None and (a.max_frames or 10**9) <= 64) as p1:
                ana = pipe.analyze(_observed(src1, progress, cancel, "analysis", total))
            state["analysis_frames"] = ana["frames"]
            mon.sample()
            progress("stage", stage="render")
            src2 = FrameSource(a.input, interp, dev, fps, ts, a.max_frames)
            with profile_stage("process_render", prof_dir, enabled=prof_dir is not None and ana["frames"] <= 64) as p2:
                ren = pipe.render(_observed(src2, _quiet, cancel, "render", total), on_frame)
            mon.sample()
    except BaseException:
        exporter.abort()
        raise
    finally:
        guard.active = False
    progress("stage", stage="finalize")
    prev = exporter.close()
    if a.mode == "hybrid":
        from .canonical import save

        cdir = out / "canonical"
        for key, st in pipe.canon.items():
            save(st, cdir)
    hashes_after = hash_files(files)
    changed = [f for f in hashes_before if hashes_before[f] != hashes_after.get(f)]
    frames_json = []
    for fi in sorted(pipe.records):
        frames_json.append({"frame": fi, "shot": pipe.shots.get(fi), "eyes": [r.as_dict() for r in pipe.records[fi]]})
    (out / "tracks.json").write_text(json.dumps({"frames": frames_json, "cuts": pipe.cuts, "excess_faces": pipe.excess_faces,
                                                  "canonical": pipe.canon_meta}, indent=1), encoding="utf-8")
    timestamps = [e["time_s"] for e in exporter.index]
    manifest = {"input": os.path.abspath(a.input), "files": len(files), "sha256_before": hashes_before, "sha256_after": hashes_after,
                "source_files_modified": len(changed), "video": vinfo.as_dict() if vinfo else None,
                "sequence_timing": getattr(src1, "timing", None), "interpretation": interp, "sidecar": str(sidecar_path) if sidecar_path else None,
                "reference_definition": "decoded source converted once by the recorded GPU transform to FP32 " + interp["working_color"],
                "transform": src1.transform_record(), "frame_count": len(exporter.index), "timestamps_s": timestamps,
                "pts": [e["pts"] for e in exporter.index], "alpha_present": bool(ren.get("alpha_frames", 0)),
                "alpha_frames_verified_unchanged": ren.get("alpha_frames", 0),
                "fps_exact": str(fps_exact) if fps_exact is not None else None,
                "checkpoint": {"path": os.path.abspath(model.path), "sha256": bundle.sha256, "name": model.name}}
    (out / "source_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    warn = list(still_warn)
    if model.restore_mode and restore_mode != model.restore_mode:
        warn.append(f"restore_mode {restore_mode!r} overrides this model's {model.restore_mode!r}: the output is not the "
                    f"configuration this model was promoted with")
    if pipe.excess_faces:
        warn.append(f"{pipe.excess_faces} faces beyond max_faces={a.max_faces} were not processed")
    if a.mode == "hybrid":
        unc = [k for k, m in pipe.canon_meta.items() if m.get("uncertain_prior_only")]
        if unc:
            warn.append(f"canonical iris states with little/no visible iris evidence (prior-dominated): {unc}")
    if video and vinfo is not None and vinfo.has_audio:
        warn.append("source has an audio stream; previews are video-only (no audio remux in this build)")
    for w in warn:
        progress("warning", message=w)
    status_counts: dict[str, int] = {}
    for fr in exporter.index:
        for e in fr["eyes"]:
            status_counts[e["status"]] = status_counts.get(e["status"], 0) + 1
    abst = sum(e.get("abstained_px", 0) for fr in exporter.index for e in fr["eyes"])
    supp = sum(e.get("support_px", 0) for fr in exporter.index for e in fr["eyes"])
    synth = sum(e.get("synth_px", 0) for fr in exporter.index for e in fr["eyes"])
    report = {
        "schema": PROCESS_REPORT_SCHEMA, "eyeclean_version": __version__,
        "report_kind": "process_run", "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input": os.path.abspath(a.input), "output": str(out), "frames": len(exporter.index),
        "model": dict(model.as_dict(), restore_mode=restore_mode, restore_mode_source=restore_src),
        "job": a.job, "interpretation": interp, "fps_exact": str(fps_exact) if fps_exact is not None else None,
        "exr_compression": a.exr_compression, "eye_status_counts": status_counts,
        "mode": a.mode, "allow_synthesis": a.allow_synthesis, "strength": a.strength, "restore_mode": restore_mode,
        "device": asdict(device_info(dev)), "memory": mon.result(), "budget": budget.as_dict(),
        "analysis": ana, "render": ren, "timing_s": {k: round(v, 3) for k, v in pipe.timing.items()},
        "preservation": {"reference_definition": manifest["reference_definition"], "source_files_modified": len(changed),
                         "outside_support_changed_bits": 0 if ren["frames"] == len(exporter.index) else None,
                         "frames_verified": ren["frames"],
                         "note": "every frame is checked on the GPU: IEEE-754 bit patterns outside the final support equal the reference"},
        "coverage": {"support_px": supp, "abstained_px": abst, "synthesized_px": synth},
        "gpu_audit": audit.summary(), "static_scan": static_hot_path_scan(ROOT / "src" / "eyeclean"),
        "profiles": [p for p in (p1, p2) if p.get("trace")], "file_access": {"opened": len(guard.opened), "forbidden_roots": guard.forbidden,
                                                                               "violations": guard.violations},
        "previews": prev, "outputs": {"index": "patches/index.json", "patches": "patches/", "masks": "masks/", "uncertainty": "uncertainty/",
                                      "master": "master/" if exporter.master else None, "tracks": "tracks.json",
                                      "source_manifest": "source_manifest.json",
                                      "photo": exporter.photo["path"] if exporter.photo else None,
                                      "canonical": "canonical/" if a.mode == "hybrid" else None,
                                      "note": "MP4 previews are viewing copies, not pixel-preservation masters"},
        "warnings": warn, "seconds": round(time.time() - t0, 2),
    }
    (out / "RUN_REPORT.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    (out / "file_access.log").write_text("\n".join(guard.opened), encoding="utf-8")
    summary = {"schema": PROCESS_REPORT_SCHEMA, "output": str(out), "report": str(out / "RUN_REPORT.json"), "frames": len(exporter.index),
               "cuts": pipe.cuts, "eye_status_counts": status_counts, "support_px": supp, "abstained_px": abst,
               "synthesized_px": synth, "source_files_modified": len(changed), "cpu_fallbacks": audit.summary()["cpu_numerical_image_fallbacks"],
               "model": model.name or str(model.path), "restore_mode": restore_mode, "seconds": report["seconds"], "warnings": warn}
    if exporter.photo:
        summary["photo"] = exporter.photo["path"]
    if changed:
        raise RuntimeError(f"source files changed during processing: {changed}")
    return {"summary": summary, "report": report}

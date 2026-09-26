"""EyeClean command-line interface.

Exit codes (stable, docs/INTEGRATION.md): 0 ok, 2 configuration/usage error, 3 GPU policy error,
4 unsupported input/format, 5 runtime failure, 6 verification failed, 7 cancelled (cancel file, Ctrl+C).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

EXIT_OK, EXIT_CONFIG, EXIT_GPU, EXIT_INPUT, EXIT_RUNTIME, EXIT_VERIFY, EXIT_CANCELLED = 0, 2, 3, 4, 5, 6, 7




def _print(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


def cmd_doctor(a) -> int:
    from ..validation.doctor import run_doctor

    rep = run_doctor(a.gpu_policy, out=Path(a.output) if a.output else None, full=not a.quick)
    _print(rep)
    return EXIT_OK if rep["passed"] else EXIT_GPU


def cmd_generate(a) -> int:
    from ..config.loader import load_config
    from ..data.generate import generate
    from ..gpu.runtime import require_cuda
    from ..render.childproc import cancel_all
    from .pause import KeyboardPause

    cfg = load_config(a.config)
    require_cuda(cfg["gpu"]["policy"], cfg["gpu"]["device_index"])
    # Ctrl+C must reach the Blender workers: they run in a thread pool, where it never arrives
    with KeyboardPause(tag="generate", on_interrupt=cancel_all) as pause:
        res = generate(cfg.raw, resume=a.resume, workers=a.workers, pause=pause)
    _print({k: v for k, v in res.items() if k != "failures"} | {"failures": res["failures"][:20], "n_failures": len(res["failures"])})
    return EXIT_OK if not res["failures"] else EXIT_RUNTIME


def cmd_train(a) -> int:
    from ..config.loader import load_config
    from ..training.trainer import train_stage

    cfg = load_config(a.config)
    res = train_stage(cfg.raw, a.stage, resume=a.resume)
    _print(res)
    return EXIT_OK


def cmd_evaluate(a) -> int:
    from ..validation.evaluate import evaluate_run

    res = evaluate_run(Path(a.run), split=a.split)
    _print(res.get("summary", res))
    return EXIT_OK


def cmd_process(a) -> int:
    from ..inference.job import DEFAULTS, job_record, load_job, merge
    from ..inference.process import ProcessArgs, run_process
    from ..inference.progress import CancelToken, JsonlSink

    o = merge(load_job(a.job) if a.job else None, {k: getattr(a, k, None) for k in list(DEFAULTS) + ["input", "output"]})
    sinks = []
    if a.progress == "jsonl":
        sinks.append(JsonlSink(None))
    if o["progress_file"]:
        sinks.append(JsonlSink(o["progress_file"]))
    a._progress_sinks = sinks

    def emit(ev):
        for s in sinks:
            s(ev)

    pa = ProcessArgs(
        input=o["input"], checkpoint=o["model"], input_color=o["input_color"], mode=o["mode"], allow_synthesis=o["allow_synthesis"],
        output=o["output"], gpu_policy=a.gpu_policy, export=o["export"], input_range=o["input_range"], working_color=o["working_color"],
        fps=o["fps"], timestamps=o["timestamps"], strength=o["strength"], max_frames=o["max_frames"], max_region=o["max_region"],
        full_frame_master=o["full_frame_master"], restore_mode=o["restore_mode"], forbid_read=a.forbid_read, profile=not a.no_profile,
        exr_compression=o["exr_compression"], progress=emit if sinks else None,
        cancel=CancelToken(o["cancel_file"]) if o["cancel_file"] else None,
        job=job_record(o, str(Path(a.job).resolve()) if a.job else None))
    res = run_process(pa)
    if sinks:
        emit({"schema": "eyeclean.progress/1", "event": "done", "t": res["report"]["seconds"], "summary": res["summary"]})
    if a.progress != "jsonl":
        _print(res["summary"])
    return EXIT_OK


def cmd_fix(a) -> int:
    from ..inference.fix import fix_file

    results = [fix_file(f, model=a.model, output=a.output, many=len(a.inputs) > 1, keep_work=a.keep_work) for f in a.inputs]
    _print(results[0] if len(results) == 1 else results)
    return EXIT_OK


def cmd_models(a) -> int:
    from ..models.registry import fetch_model, list_models

    if a.fetch is not None:
        _print(fetch_model(a.fetch or None, Path(a.dir) if a.dir else None))
        return EXIT_OK
    _print(list_models())
    return EXIT_OK


def cmd_apply(a) -> int:
    from ..inference.apply_patches import run_apply

    res = run_apply(a.input, Path(a.patches), Path(a.output), a.gpu_policy, max_frames=a.max_frames)
    _print(res)
    return EXIT_OK if res.get("verified") else EXIT_VERIFY


def cmd_verify(a) -> int:
    from ..validation.verify_run import verify_run

    res = verify_run(Path(a.run))
    _print(res)
    return EXIT_OK if res["valid"] else EXIT_VERIFY


def cmd_build_demo(a) -> int:
    from ..pipeline.build_demo import build_demo

    res = build_demo(a.config, resume=a.resume, workers=a.workers, stop_after=a.stop_after, real_footage_files=a.real_footage or None,
                     real_footage_color=a.real_footage_color, real_footage_range=a.real_footage_range)
    _print(res)
    return EXIT_OK if res.get("completed") else EXIT_RUNTIME


def cmd_review(a) -> int:
    from ..validation.review_video import review

    procs = {}
    for item in a.process:
        name, _, d = item.rpartition("=")
        procs[name or Path(d).name] = Path(d)
    rows = {}
    for item in a.reference_row:
        label, _, idx = item.rpartition("=")
        rows[label] = int(idx)
    res = review(a.input, procs, Path(a.output), a.reference, rows, a.reference_crop, a.reference_block, a.reference_label_px,
                 a.reference_source_row, a.reference_fps, a.slow, a.max_frames)
    _print(res)
    return EXIT_OK


def cmd_fetch(a) -> int:
    from ..assets.fetch import fetch

    lock = fetch(a.keys or None)
    _print({k: {"sha256": v["sha256"], "license": v["license"]} for k, v in lock["assets"].items()})
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    from ..version import __version__

    p = argparse.ArgumentParser(prog="eyeclean", description="GPU eye-reflection (catchlight) removal for video (strict GPU policy).")
    p.add_argument("--version", action="version", version=f"eyeclean {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("doctor", help="inspect GPU/driver/Blender/OptiX/codecs/EXR and run smoke tests")
    d.add_argument("--gpu-policy", default="strict", choices=["strict"])
    d.add_argument("--output", help="write the doctor report JSON here")
    d.add_argument("--quick", action="store_true", help="skip the Blender render and codec round trip")
    d.set_defaults(fn=cmd_doctor)

    b = sub.add_parser("build-demo", help="run the full autonomous pipeline for a profile (generate, train, evaluate, gallery, report)")
    b.add_argument("--config", required=True)
    b.add_argument("--resume", default="auto", choices=["auto", "never", "require"])
    b.add_argument("--workers", type=int, default=2, help="parallel headless Blender workers")
    b.add_argument("--stop-after", default=None, help="stop after a named stage (debugging)")
    b.add_argument("--real-footage", action="append", default=[], help="real video to sample for human review (repeatable; read-only; "
                                                                        "default: files in inputs/)")
    b.add_argument("--real-footage-color", default=None, help="explicit input colour for --real-footage files (when metadata/sidecar is absent)")
    b.add_argument("--real-footage-range", default=None, choices=["full", "limited"])
    b.set_defaults(fn=cmd_build_demo)

    g = sub.add_parser("generate", help="plan, render (Cycles OptiX) and cache the paired CG dataset")
    g.add_argument("--config", required=True)
    g.add_argument("--resume", default="auto", choices=["auto", "never", "require"])
    g.add_argument("--workers", type=int, default=2)
    g.set_defaults(fn=cmd_generate)

    t = sub.add_parser("train", help="train one stage: guard (EyeGuard), image (single-frame EyeRestore) or temporal (5-frame)")
    t.add_argument("--config", required=True)
    t.add_argument("--stage", required=True, choices=["guard", "image", "temporal"])
    t.add_argument("--resume", default="auto", choices=["auto", "never", "require"])
    t.set_defaults(fn=cmd_train)

    e = sub.add_parser("evaluate", help="held-out evaluation of a run (independent full-frame detection, metrics by bucket)")
    e.add_argument("--run", required=True)
    e.add_argument("--split", default="test", choices=["test", "validation"])
    e.set_defaults(fn=cmd_evaluate)

    pr = sub.add_parser("process", help="process a video file or EXR sequence (contract: docs/INTEGRATION.md)")
    pr.add_argument("input", nargs="?", default=None, help="video file, EXR directory / %%06d / #### pattern / glob (or from --job)")
    pr.add_argument("--job", default=None, help="eyeclean.job/1 JSON file (contracts/process_job.schema.json); flags override it")
    pr.add_argument("--model", "--checkpoint", dest="model", default=None,
                    help="registry model name (`eyeclean models`) or .safetensors path; default: $EYECLEAN_MODEL or the registry default")
    pr.add_argument("--input-color", default=None, help="linear-acescg | linear-rec709 | rec709 | rec709-display | srgb | panasonic-vlog-vgamut")
    pr.add_argument("--mode", default=None, choices=["conservative", "hybrid"])
    pr.add_argument("--allow-synthesis", action="store_true", default=None,
                    help="required for hybrid mode: permits labeled synthesized iris/pupil content")
    pr.add_argument("--output", default=None)
    pr.add_argument("--gpu-policy", default="strict", choices=["strict"])
    pr.add_argument("--export", default=None, help="comma list: patches,preview,report,master,photo (default patches,preview,report; a photo input writes the photo)")
    pr.add_argument("--input-range", default=None, choices=["auto", "full", "limited"])
    pr.add_argument("--working-color", default=None, choices=["linear-acescg", "linear-rec709"])
    pr.add_argument("--fps", default=None, help="image sequences only: constant frame rate, e.g. 24, 25 or exactly 60000/1001")
    pr.add_argument("--timestamps", default=None, help="JSON/text file with explicit per-frame timestamps for image sequences")
    pr.add_argument("--strength", type=float, default=None)
    pr.add_argument("--max-frames", type=int, default=None)
    pr.add_argument("--max-region", default=None, help="optional maximum-allowed edit mask: ONE single-channel full-frame EXR for the clip")
    pr.add_argument("--full-frame-master", action="store_true", default=None, help="also write full-frame FLOAT EXR masters")
    pr.add_argument("--exr-compression", default=None, choices=["none", "zips", "zip"], help="output EXR compression (lossless; default none)")
    pr.add_argument("--progress", default="none", choices=["none", "jsonl"],
                    help="jsonl: stdout carries only eyeclean.progress/1 JSON lines, ending with a done or error event")
    pr.add_argument("--progress-file", default=None, help="also append the progress JSON lines to this file")
    pr.add_argument("--cancel-file", default=None, help="stop cleanly (exit 7) as soon as this file exists")
    pr.add_argument("--restore-mode", default=None, choices=["auto", "temporal", "single"],
                    help="auto (default): the model registry's setting for this bundle")
    pr.add_argument("--forbid-read", action="append", default=[], help=argparse.SUPPRESS)
    pr.add_argument("--no-profile", action="store_true", help=argparse.SUPPRESS)
    pr.set_defaults(fn=cmd_process)

    ap = sub.add_parser("apply-patches", help="reference compositor: apply signed EXR corrections to the source")
    ap.add_argument("input")
    ap.add_argument("--patches", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--gpu-policy", default="strict", choices=["strict"])
    ap.add_argument("--max-frames", type=int, default=None)
    ap.set_defaults(fn=cmd_apply)

    v = sub.add_parser("verify-run", help="verify RUN_REPORT.json against the schema, frozen thresholds and evidence on disk")
    v.add_argument("--run", required=True)
    v.set_defaults(fn=cmd_verify)

    rv = sub.add_parser("review-video", help="eye-focused review videos of processed footage (split screen; optional matched-crop "
                                             "comparison with an external eye-crop video)")
    rv.add_argument("input", help="the original video that was processed (read-only)")
    rv.add_argument("--process", action="append", required=True, help="NAME=DIR of an `eyeclean process` output (repeatable)")
    rv.add_argument("--output", required=True)
    rv.add_argument("--reference", default=None, help="external comparison video of the same frames: rows of nearest-neighbour eye panels")
    rv.add_argument("--reference-row", action="append", default=[], help="LABEL=ROW: external row to include (0-based)")
    rv.add_argument("--reference-source-row", type=int, default=0, help="row of the external video that shows the untouched source")
    rv.add_argument("--reference-crop", type=int, default=48, help="source pixels per external panel side")
    rv.add_argument("--reference-block", type=int, default=10, help="external nearest-neighbour magnification")
    rv.add_argument("--reference-label-px", type=int, default=18, help="height of the external label bar drawn over each row")
    rv.add_argument("--reference-fps", type=float, default=12.0)
    rv.add_argument("--slow", type=int, default=4, help="slow-motion factor of the extra split-screen copy")
    rv.add_argument("--max-frames", type=int, default=None)
    rv.set_defaults(fn=cmd_review)

    fx = sub.add_parser("fix", help="repair a photo (JPEG, PNG, WebP, TIFF, ...): colour, orientation and bit depth come from the "
                                     "file; writes <name>.eyeclean.<ext> next to it")
    fx.add_argument("inputs", nargs="+", help="one or more photos")
    fx.add_argument("--model", default=None, help="a registry model name or a .safetensors path (default: the registry default)")
    fx.add_argument("--output", default=None, help="output file (one input) or directory")
    fx.add_argument("--keep-work", action="store_true", help="keep the full process output (EXR patches, masks, report)")
    fx.set_defaults(fn=cmd_fix)

    m = sub.add_parser("models", help="list the known models, where they are installed and the model search path")
    m.add_argument("--fetch", nargs="?", const="", default=None, metavar="NAME",
                   help="download a model (default: the registry default) and verify its SHA-256")
    m.add_argument("--dir", default=None, help="with --fetch: where to put it (default: the user model directory)")
    m.set_defaults(fn=cmd_models)

    f = sub.add_parser("fetch-assets", help="download/verify public third-party assets (hash pinned)")
    f.add_argument("keys", nargs="*")
    f.set_defaults(fn=cmd_fetch)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    a = parser.parse_args(argv)
    from ..config.schema import ConfigError
    from ..gpu.runtime import GpuPolicyError

    from ..color.transforms import ColorConfigError
    from ..inference.progress import Cancelled
    from ..io.errors import InputError
    from ..io.exr import ExrError

    def fail(code: int, kind: str, msg: str) -> int:
        print(f"eyeclean: {kind}: {msg}", file=sys.stderr)
        ev = {"schema": "eyeclean.progress/1", "event": "error", "code": code, "kind": kind, "message": msg}
        for s in getattr(a, "_progress_sinks", []):
            try:
                s(ev)
            except Exception:  # noqa: BLE001
                pass
        return code

    try:
        return a.fn(a)
    except ConfigError as e:
        return fail(EXIT_CONFIG, "configuration error", str(e))
    except GpuPolicyError as e:
        return fail(EXIT_GPU, "GPU policy error", str(e))
    except (Cancelled, KeyboardInterrupt) as e:
        return fail(EXIT_CANCELLED, "cancelled", (str(e) or "interrupted") + "; the output directory is incomplete (no RUN_REPORT.json)")
    except (InputError, ExrError, ColorConfigError) as e:
        return fail(EXIT_INPUT, "unsupported input", str(e))
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        return fail(EXIT_RUNTIME, "failed", str(e))


if __name__ == "__main__":
    raise SystemExit(main())

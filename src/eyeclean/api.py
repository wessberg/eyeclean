"""EyeClean Python API (same-environment callers).  Contract: docs/INTEGRATION.md.

    from eyeclean.api import process

    res = process("interview.mp4", "out/", input_color="rec709", on_progress=print)
    print(res.summary["eye_status_counts"], res.report_path)

    # or with a job (dict or eyeclean.job/1 file); keyword arguments override it
    res = process(job="shot042.eyeclean.json", cancel=stop_event)

Callers in another Python / torch / CUDA environment (Blender, Nuke, a render farm) should run the CLI as a
subprocess instead (`eyeclean process --job ... --progress jsonl`), or use the dependency-free client in
clients/python.  Notes for embedding:
  * the call holds the GPU for the whole clip and runs a strict GPU audit: `on_progress` receives plain dicts
    and must not do CPU tensor work;
  * it installs a file-access audit hook (Python cannot remove one), so prefer one process per job in
    long-lived hosts;
  * exceptions: eyeclean.api.Cancelled, InputError (unsupported input), ConfigError, GpuPolicyError, RuntimeError.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .config.schema import ConfigError  # noqa: F401  (re-exported)
from .gpu.runtime import GpuPolicyError  # noqa: F401  (re-exported)
from .inference.job import DEFAULTS, job_record, load_job, merge, validate_job
from .inference.process import ProcessArgs, run_process
from .inference.progress import Cancelled, CancelToken  # noqa: F401  (re-exported)
from .io.errors import InputError  # noqa: F401  (re-exported)
from .models.registry import list_models, resolve_model  # noqa: F401  (re-exported)
from .version import __version__  # noqa: F401  (re-exported)


@dataclass
class ProcessResult:
    output: Path
    summary: dict
    report: dict

    @property
    def report_path(self) -> Path:
        return self.output / "RUN_REPORT.json"

    @property
    def index_path(self) -> Path:
        return self.output / "patches" / "index.json"


def _cancel_token(cancel) -> CancelToken | None:
    if cancel is None:
        return None
    if isinstance(cancel, CancelToken):
        return cancel
    if isinstance(cancel, threading.Event):
        return CancelToken(fn=cancel.is_set)
    if isinstance(cancel, (str, Path)):
        return CancelToken(path=str(cancel))
    if callable(cancel):
        return CancelToken(fn=cancel)
    raise TypeError("cancel must be a threading.Event, a callable returning bool, or a file path")


def process(input: str | None = None, output: str | None = None, *, job: dict | str | Path | None = None,
            on_progress: Callable[[dict], None] | None = None, cancel=None, **options) -> ProcessResult:
    """Process a video or EXR sequence.  options: any eyeclean.job/1 key (model, input_color, fps, export,
    exr_compression, max_frames, restore_mode, strength, working_color, ...)."""
    unknown = set(options) - set(DEFAULTS)
    if unknown:
        raise TypeError(f"unknown option(s) {sorted(unknown)}; valid: {sorted(DEFAULTS)}")
    job_file = None
    if isinstance(job, (str, Path)):
        job_file = str(Path(job).resolve())
        job = load_job(job)
    elif isinstance(job, dict):
        job = validate_job(job, None, "job dict")
    o = merge(job, dict(options, input=input, output=output))
    pa = ProcessArgs(input=o["input"], checkpoint=o["model"], input_color=o["input_color"], mode=o["mode"],
                     allow_synthesis=o["allow_synthesis"], output=o["output"], export=o["export"], input_range=o["input_range"],
                     working_color=o["working_color"], fps=o["fps"], timestamps=o["timestamps"], strength=o["strength"],
                     max_frames=o["max_frames"], max_region=o["max_region"], full_frame_master=o["full_frame_master"],
                     restore_mode=o["restore_mode"], exr_compression=o["exr_compression"], progress=on_progress,
                     cancel=_cancel_token(cancel), job=job_record(o, job_file))
    res = run_process(pa)
    return ProcessResult(Path(res["summary"]["output"]), res["summary"], res["report"])

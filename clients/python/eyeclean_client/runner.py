"""Run the EyeClean engine as a subprocess: job file in, eyeclean.progress/1 events out, typed errors by exit code.

The engine lives in its own environment (Python 3.12, PyTorch, CUDA).  Point to it with one of:
  EYECLEAN_CMD     the full command, e.g.  "C:/tools/eyeclean/.venv/Scripts/eyeclean.exe"  or  "/opt/eyeclean/.venv/bin/python -m eyeclean"
  EYECLEAN_PYTHON  the engine environment's python (runs `python -m eyeclean`)
  PATH             an `eyeclean` executable
or pass engine=[...] explicitly.
"""
from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

EXIT_NAMES = {0: "ok", 2: "configuration error", 3: "GPU policy error", 4: "unsupported input", 5: "runtime failure",
              6: "verification failed", 7: "cancelled"}


class EyeCleanError(RuntimeError):
    def __init__(self, code: int, message: str, events: list):
        super().__init__(f"eyeclean exited {code} ({EXIT_NAMES.get(code, 'unknown')}): {message}")
        self.code, self.message, self.events = code, message, events


class EyeCleanConfigError(EyeCleanError):
    pass


class EyeCleanGpuError(EyeCleanError):
    pass


class EyeCleanInputError(EyeCleanError):
    pass


class EyeCleanCancelled(EyeCleanError):
    pass


_BY_CODE = {2: EyeCleanConfigError, 3: EyeCleanGpuError, 4: EyeCleanInputError, 7: EyeCleanCancelled}


def find_engine() -> List[str]:
    cmd = os.environ.get("EYECLEAN_CMD")
    if cmd:
        return shlex.split(cmd, posix=(os.name != "nt"))
    py = os.environ.get("EYECLEAN_PYTHON")
    if py:
        return [py, "-m", "eyeclean"]
    exe = shutil.which("eyeclean")
    if exe:
        return [exe]
    raise FileNotFoundError("EyeClean engine not found: set EYECLEAN_CMD or EYECLEAN_PYTHON, or put `eyeclean` on PATH")


@dataclass
class RunResult:
    returncode: int
    output: Path
    summary: Optional[dict]
    events: list = field(default_factory=list)

    @property
    def report(self) -> dict:
        return json.loads((self.output / "RUN_REPORT.json").read_text(encoding="utf-8"))


def _cancel_probe(cancel) -> Optional[Callable[[], bool]]:
    if cancel is None:
        return None
    if isinstance(cancel, threading.Event):
        return cancel.is_set
    if isinstance(cancel, (str, os.PathLike)):
        return lambda: os.path.exists(cancel)
    if callable(cancel):
        return cancel
    raise TypeError("cancel must be a threading.Event, a callable returning bool, or a file path to watch")


def run(job, *, engine: Optional[List[str]] = None, on_event: Optional[Callable[[dict], None]] = None,
        cancel=None, stderr_path=None, check: bool = True, keep_events: bool = False) -> RunResult:
    """Run one job.  job: an eyeclean.job/1 dict (written to a temporary file) or a path to a job file.
    on_event receives every progress event.  cancel: a threading.Event, a callable returning True, or a file path whose
    appearance cancels (e.g. a host's "release the GPU" flag); the run then stops cleanly (EyeCleanCancelled)."""
    engine = engine or find_engine()
    tmpdir = tempfile.mkdtemp(prefix="eyeclean_job_")
    try:
        if isinstance(job, dict):
            job = dict(job)
            job.setdefault("schema", "eyeclean.job/1")
            for k in ("input", "output", "timestamps", "max_region"):
                if isinstance(job.get(k), str):
                    job[k] = os.path.abspath(job[k])
            job_path = Path(tmpdir) / "job.json"
            job_path.write_text(json.dumps(job, indent=1), encoding="utf-8")
            output = Path(job["output"])
        else:
            job_path = Path(job).resolve()
            spec = json.loads(job_path.read_text(encoding="utf-8"))
            output = (job_path.parent / spec["output"]).resolve()
        cancel_file = Path(tmpdir) / "CANCEL"
        argv = [*engine, "process", "--job", str(job_path), "--progress", "jsonl", "--cancel-file", str(cancel_file)]
        err = open(stderr_path, "a", encoding="utf-8") if stderr_path else subprocess.PIPE
        proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=err, text=True, encoding="utf-8", errors="replace", bufsize=1)
        stderr_chunks: List[str] = []
        if proc.stderr is not None:                   # drain stderr concurrently so a chatty engine never blocks on a full pipe
            threading.Thread(target=lambda: stderr_chunks.extend(proc.stderr), daemon=True).start()
        stop = threading.Event()
        probe = _cancel_probe(cancel)
        if probe is not None:
            def watch():
                while not stop.wait(0.5):
                    if probe():
                        cancel_file.write_text("cancel", encoding="utf-8")
                        return
            threading.Thread(target=watch, daemon=True).start()
        events, last, summary = [], None, None
        for line in proc.stdout:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            last = ev
            if keep_events:
                events.append(ev)
            if ev.get("event") == "done":
                summary = ev.get("summary")
            if on_event is not None:
                on_event(ev)
        code = proc.wait()
        stop.set()
        stderr_text = "".join(stderr_chunks)
        if stderr_path:
            err.close()
        if code != 0 and check:
            msg = (last or {}).get("message") or stderr_text.strip()[-2000:] or "no message"
            raise _BY_CODE.get(code, EyeCleanError)(code, msg, events)
        return RunResult(code, output, summary, events)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

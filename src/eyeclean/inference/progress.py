"""Progress events and cancellation for `eyeclean process` (contract: docs/INTEGRATION.md, eyeclean.progress/1).

Events are plain dicts; `JsonlSink` writes one JSON object per line (flushed) to stdout or a file.
Every event carries ``schema``, ``event`` and ``t`` (seconds since the run started).  Events:
  start     {input, output, model, frames_expected}
  stage     {stage: "analysis" | "render" | "finalize"}
  frame     {stage, frame, done, total, eyes_repaired}   (one per frame and stage)
  warning   {message}
  done      {summary}                                     (the last line of a successful run)
  error     {code, kind, message}                         (the last line of a failed or cancelled run)
Cancellation: a CancelToken is checked before every frame of both passes.  It fires when its file
exists or its callable returns True; the run then stops with Cancelled (CLI exit code 7).  A
cancelled output directory is incomplete: it has no RUN_REPORT.json.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Callable

from ..version import PROGRESS_SCHEMA


class Cancelled(Exception):
    """The caller asked the run to stop (cancel file / callable / Ctrl+C)."""


class CancelToken:
    def __init__(self, path: str | None = None, fn: Callable[[], bool] | None = None):
        self.path, self.fn = path, fn

    def requested(self) -> bool:
        return bool((self.path and os.path.exists(self.path)) or (self.fn and self.fn()))

    def check(self, where: str) -> None:
        if self.requested():
            raise Cancelled(f"cancelled during {where}" + (f" ({self.path} exists)" if self.path and os.path.exists(self.path) else ""))


class Progress:
    """Fan-out to zero or more sinks (callables taking one event dict)."""

    def __init__(self, sinks: list[Callable[[dict], None]] | None = None):
        self.sinks = [s for s in (sinks or []) if s is not None]
        self.t0 = time.time()

    def __call__(self, event: str, **fields) -> None:
        if not self.sinks:
            return
        ev = {"schema": PROGRESS_SCHEMA, "event": event, "t": round(time.time() - self.t0, 3), **fields}
        for s in self.sinks:
            s(ev)


class JsonlSink:
    def __init__(self, path: str | None = None):
        self.fh = open(path, "a", encoding="utf-8", buffering=1) if path else sys.stdout

    def __call__(self, ev: dict) -> None:
        self.fh.write(json.dumps(ev, default=str) + "\n")
        self.fh.flush()

    def close(self) -> None:
        if self.fh is not sys.stdout:
            self.fh.close()

"""Typed access to an EyeClean output directory (eyeclean.process/1).

    out = RunOutput("shot042_eyeclean")
    for fr in out.frames():
        for eye in fr.eyes:
            if eye.status == "repaired":
                corr = out.read(eye.patch)          # numpy channels R, G, B: output - source, working space
                x0, y0, x1, y1 = eye.data_window    # [x0, x1) x [y0, y1) in full-frame pixels

Applying a patch is one addition inside its window: result[y0:y1, x0:x1] = source[y0:y1, x0:x1] + correction (exactly zero
outside the hard support).  The mask file's `feather` channel is the per-pixel blend weight that was used.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional

from .exr import Exr, read_exr

SCHEMA = "eyeclean.process/1"


@dataclass
class EyeInfo:
    track: int
    eye: str
    status: str                      # "repaired" | "untouched"
    reasons: List[str]
    support_px: int
    data_window: Optional[List[int]] = None
    patch: Optional[str] = None
    replacement: Optional[str] = None
    mask: Optional[str] = None
    uncertainty: Optional[str] = None
    raw: dict = field(default_factory=dict)


@dataclass
class FrameInfo:
    frame: int
    time_s: float
    eyes: List[EyeInfo]


class RunOutput:
    def __init__(self, path):
        self.path = Path(path)
        if not (self.path / "RUN_REPORT.json").exists():
            raise FileNotFoundError(f"{self.path}: no RUN_REPORT.json (the run is incomplete, failed or was cancelled)")
        self.report = json.loads((self.path / "RUN_REPORT.json").read_text(encoding="utf-8"))
        self.index = json.loads((self.path / "patches" / "index.json").read_text(encoding="utf-8"))
        if self.report.get("schema") != SCHEMA:
            raise ValueError(f"{self.path}: report schema {self.report.get('schema')!r}, this client reads {SCHEMA}")

    @property
    def working_color(self) -> str:
        return self.index["working_color"]

    @property
    def size(self):
        return self.index["width"], self.index["height"]

    def frames(self) -> Iterator[FrameInfo]:
        for fr in self.index["frames"]:
            eyes = [EyeInfo(e["track"], e["eye"], e["status"], list(e.get("reasons", [])), int(e.get("support_px", 0)),
                            e.get("data_window"), e.get("patch"), e.get("replacement"), e.get("mask"), e.get("uncertainty"), e)
                    for e in fr["eyes"]]
            yield FrameInfo(fr["frame"], fr["time_s"], eyes)

    def read(self, rel: str) -> Exr:
        return read_exr(self.path / rel)

"""Uncompressed scanline EXR image-sequence input (strict route for non-hardware-decodable media).

Patterns: printf style (``frame_%06d.exr``), ``#`` padding (``frame_####.exr``), a glob, or a
directory.  Timing must be explicit: ``--fps`` for constant-rate sequences or ``--timestamps``
(JSON list / text lines of seconds) for variable timing.  Nothing is invented.
"""

from __future__ import annotations

import glob
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import torch

from .errors import InputError
from .exr import ExrError, probe_exr, read_exr


@dataclass
class SequenceFrame:
    index: int
    path: str
    frame_number: int
    time_s: float
    rgb: torch.Tensor  # (3,H,W) FP32 as stored (source primaries/transfer)
    alpha: torch.Tensor | None
    display_window: tuple
    data_window: tuple


def resolve_sequence(pattern: str) -> list[tuple[int, str]]:
    p = Path(pattern)
    files: list[str] = []
    if p.is_dir():
        files = sorted(glob.glob(str(p / "*.exr")))
    elif "%" in p.name:
        # match basenames only: frame_%06d.exr -> ^frame_(\d{6})\.exr$
        parts = re.split(r"(%0?\d*d)", p.name)
        rx = "".join((r"(\d{%s})" % re.match(r"%0?(\d*)d", s).group(1) if re.match(r"%0?(\d*)d", s).group(1) else r"(\d+)")
                     if re.fullmatch(r"%0?\d*d", s) else re.escape(s) for s in parts)
        files = sorted(str(f) for f in p.parent.iterdir() if f.is_file() and re.fullmatch(rx, f.name))
    elif "#" in pattern:
        files = sorted(glob.glob(re.sub(r"#+", lambda m: "[0-9]" * len(m.group(0)), pattern)))
    else:
        files = sorted(glob.glob(pattern))
    out = []
    for f in files:
        m = re.findall(r"(\d+)", Path(f).stem)
        out.append((int(m[-1]) if m else len(out), f))
    out.sort()
    if not out:
        raise InputError(f"no EXR frames match {pattern!r}")
    return out


def load_timestamps(path: str, n: int) -> list[float]:
    txt = Path(path).read_text(encoding="utf-8").strip()
    try:
        vals = json.loads(txt)
        if isinstance(vals, dict):
            vals = vals.get("timestamps_s") or vals.get("timestamps")
    except json.JSONDecodeError:
        vals = [float(x) for x in txt.split()]
    vals = [float(v) for v in vals]
    if len(vals) < n:
        raise InputError(f"--timestamps provides {len(vals)} values for {n} frames")
    if any(b <= a for a, b in zip(vals, vals[1:])):
        raise InputError("--timestamps must be strictly increasing")
    return vals[:n]


class SequenceReader:
    def __init__(self, pattern: str, device: torch.device, fps: float | None, timestamps: str | None, max_frames: int | None = None):
        self.frames = resolve_sequence(pattern)
        if max_frames is not None:
            self.frames = self.frames[:max_frames]
        n = len(self.frames)
        if timestamps:
            self.times = load_timestamps(timestamps, n)
            self.timing = {"source": "timestamps_file", "path": timestamps}
        elif fps:
            self.times = [k / fps for k in range(n)]
            self.timing = {"source": "cli_fps", "fps": fps}
        else:
            raise InputError("EXR sequences carry no trustworthy timing: pass --fps for constant-rate sequences or "
                             "--timestamps for explicit per-frame times (variable frame rate)")
        self.dev = device
        info = probe_exr(self.frames[0][1])
        if info["multipart"] or info["tiled"] or info["deep"]:
            raise InputError(f"{self.frames[0][1]}: multipart/tiled/deep EXR unsupported; provide single-part scanline files")
        h = info["parts"][0]
        self.display_window = h.display_window
        self.width = h.display_window[2] - h.display_window[0] + 1
        self.height = h.display_window[3] - h.display_window[1] + 1

    def __len__(self) -> int:
        return len(self.frames)

    def __iter__(self):
        for k, (num, path) in enumerate(self.frames):
            try:
                img = read_exr(path, self.dev)
            except ExrError as e:
                raise InputError(str(e)) from e
            names = set(img.channels)
            if not {"R", "G", "B"} <= names:
                raise InputError(f"{path}: expected R, G, B channels, found {sorted(names)}")
            dw, disp = img.header.data_window, img.header.display_window
            if disp != self.display_window:
                raise InputError(f"{path}: display window {disp} differs from the first frame {self.display_window}")
            W, H = self.width, self.height
            rgb = torch.zeros((3, H, W), dtype=torch.float32, device=self.dev)
            alpha = None
            x0, y0 = dw[0] - disp[0], dw[1] - disp[1]
            h_, w_ = img.header.height, img.header.width
            # data window may be cropped: place it into the full display window (zeros elsewhere)
            for i, c in enumerate("RGB"):
                rgb[i, y0:y0 + h_, x0:x0 + w_] = img.channels[c].float()
            if "A" in names:
                alpha = torch.zeros((H, W), dtype=img.channels["A"].dtype, device=self.dev)
                alpha[y0:y0 + h_, x0:x0 + w_] = img.channels["A"]
            yield SequenceFrame(k, path, num, self.times[k], rgb, alpha, disp, dw)

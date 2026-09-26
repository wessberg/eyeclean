"""Frame sources producing the declared decoded working-space reference B_t (FP32, CUDA).

B_t is the source frame decoded by the recorded decoder and converted ONCE by the recorded GPU
transform; exact preservation is measured against it.  Input alpha is carried separately and
never modified.  Sources can be re-opened, so the analysis pass and the output pass see
bit-identical references without holding a whole shot in memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from ..color import transforms as CT
from ..io.errors import InputError
from ..io.sequence import SequenceReader
from ..io.stills import is_still, read_still
from ..io.video import VideoReader, planes_to_rgb, probe


@dataclass
class FrameRef:
    index: int
    time_s: float
    pts: int | None
    reference: torch.Tensor  # (3,H,W) FP32 working space, read-only by convention
    alpha: torch.Tensor | None
    white: float  # linear (source primaries) white level used for clipped-pixel detection
    source_name: str
    codes: torch.Tensor | None = None  # stills: the decoded code values (3,H,W), so unedited pixels are written back exactly


def is_video(path: str) -> bool:
    p = Path(path)
    return p.is_file() and p.suffix.lower() not in (".exr",) and not is_still(path)


class FrameSource:
    def __init__(self, path: str, interp: dict, device: torch.device, fps: float | None = None, timestamps: str | None = None,
                 max_frames: int | None = None):
        self.path, self.interp, self.dev = path, interp, device
        self.fps_arg, self.ts_arg, self.max_frames = fps, timestamps, max_frames
        self.video = is_video(path)
        self.still = None
        if not self.video and Path(path).is_file() and is_still(path):
            self._still = read_still(path, device)       # (linear 709, alpha, codes, StillInfo); decoded once per source
            self.still = self._still[3]
            self.width, self.height = self.still.width, self.still.height
            self.time_base = (1, 90000)
            self.nominal_fps = float(fps) if fps else 1.0
            self.info = None
        elif self.video:
            if fps:
                raise InputError("--fps applies only to image sequences; video files keep their real timestamps")
            self.info = probe(path)
            self.width, self.height = self.info.width, self.info.height
            self.time_base = self.info.time_base
            self.nominal_fps = self.info.fps or 24.0
        else:
            seq = SequenceReader(path, device, fps, timestamps, max_frames)
            self.seq_files = [f for _, f in seq.frames]
            self.width, self.height = seq.width, seq.height
            self.timing = seq.timing
            self.time_base = (1, 90000)
            if fps:
                self.nominal_fps = fps
            elif len(seq.times) > 1:
                self.nominal_fps = (len(seq.times) - 1) / max(1e-6, seq.times[-1] - seq.times[0])
            else:
                self.nominal_fps = 24.0
            self.info = None

    def transform_record(self) -> dict:
        if self.still is not None:
            return {"input_color": self.interp["input_color"], "working_color": self.interp["working_color"],
                    "input_range": "full", "ycbcr_matrix": "n/a",
                    "transform": f"{self.still.format} decoded by {self.still.decode} -> EXIF orientation "
                                 f"{self.still.orientation} applied -> {self.still.colour} -> linear Rec.709 -> the working "
                                 "space, FP32 on CUDA"}
        d = {"input_color": self.interp["input_color"], "working_color": self.interp["working_color"],
             "input_range": self.interp["input_range"], "ycbcr_matrix": self.interp["ycbcr_matrix"],
             "transform": "decoded codes -> R'G'B' (explicit range/matrix, MPEG-2 chroma siting) -> inverse transfer -> "
                          "primaries matrix to the working space, FP32 on CUDA" if self.video else
                          "EXR samples (FLOAT/HALF->FP32) -> primaries matrix to the working space, FP32 on CUDA"}
        return d

    def __iter__(self):
        ic, wc = self.interp["input_color"], self.interp["working_color"]
        if self.still is not None:
            lin, alpha, codes, info = self._still
            ref = CT.linear709_to_working(lin, wc).contiguous()
            yield FrameRef(0, 0.0, 0, ref, alpha, 1.0, Path(self.path).name, codes)
            return
        if self.video:
            r = VideoReader(self.path, self.dev, self.max_frames)
            tb = r.info.time_base
            full = self.interp["input_range"] == "full"
            # white level in linear source units = inverse transfer of code value 1.0
            white = float(CT.encoded_to_linear(torch.ones(3, 1, 1, device=self.dev), ic)[0].max())
            for dp in r:
                rgb_enc = planes_to_rgb(dp, full, self.interp["ycbcr_matrix"])
                ref = CT.to_working(rgb_enc, ic, wc).contiguous()
                self.info = r.info
                yield FrameRef(dp.index, dp.pts * tb[0] / tb[1], dp.pts, ref, None, white, f"{Path(self.path).name}#{dp.index}")
        else:
            seq = SequenceReader(self.path, self.dev, self.fps_arg, self.ts_arg, self.max_frames)
            white = None
            for f in seq:
                ref = CT.to_working(f.rgb, ic, wc).contiguous()
                if white is None:
                    white = _plateau_white(f.rgb)
                yield FrameRef(f.index, f.time_s, int(round(f.time_s * 90000)), ref, f.alpha, white, Path(f.path).name)


def _plateau_white(rgb: torch.Tensor) -> float:
    """Linear inputs carry no white level: detect a clipping plateau (>= 64 pixels within 0.1% of
    the frame maximum in some channel); otherwise treat the input as unclipped (inf)."""
    m = rgb.amax()
    near = (rgb >= m * 0.999).any(0).sum()
    if float(m) > 0 and int(near) >= 64:
        return float(m)
    return float("inf")

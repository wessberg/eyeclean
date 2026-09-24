"""Shot-cut detection and per-shot eye tracking.

Cut detection compares small luminance thumbnails of consecutive detector-proxy frames on the
GPU (mean absolute difference + histogram distance).  Tracks never survive a cut: a new shot
resets all tracks (and with them temporal buffers and canonical iris states), so no latent is
ever transferred between shots or people.  Association inside a shot is greedy nearest-centre
matching gated by eye width; only small per-eye scalars are read back for control/logging.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


class CutDetector:
    def __init__(self, mad_thresh: float = 0.10, hist_thresh: float = 0.35):
        self.prev = None
        self.prev_hist = None
        self.mad_thresh, self.hist_thresh = mad_thresh, hist_thresh

    @torch.no_grad()
    def __call__(self, proxy: torch.Tensor) -> tuple[bool, dict]:
        lum = (0.2126 * proxy[0] + 0.7152 * proxy[1] + 0.0722 * proxy[2])[None, None]
        th = F.adaptive_avg_pool2d(lum, (18, 32))[0, 0]
        hist = torch.histc(lum.flatten(), bins=32, min=0.0, max=1.0)
        hist = hist / hist.sum().clamp(min=1)
        if self.prev is None:
            self.prev, self.prev_hist = th, hist
            return True, {"mad": None, "hist": None, "first": True}
        mad = float((th - self.prev).abs().mean())
        hd = float(0.5 * (hist - self.prev_hist).abs().sum())
        self.prev, self.prev_hist = th, hist
        cut = mad > self.mad_thresh and hd > self.hist_thresh * 0.5 or hd > self.hist_thresh
        return cut, {"mad": mad, "hist": hd}


@dataclass
class Track:
    track_id: int
    shot: int
    center: tuple[float, float]
    width: float
    last_frame: int
    frames: list = field(default_factory=list)


class EyeTracker:
    def __init__(self, max_gap: int = 2):
        self.next_id = 0
        self.tracks: list[Track] = []
        self.shot = -1
        self.max_gap = max_gap
        self.closed: list[Track] = []

    def new_shot(self) -> int:
        self.closed.extend(self.tracks)
        self.tracks = []
        self.shot += 1
        return self.shot

    def update(self, frame: int, obs: list[dict]) -> list[int]:
        """obs: dicts with 'center' (x, y) floats and 'width'. Returns a track id per observation."""
        alive = [t for t in self.tracks if frame - t.last_frame <= self.max_gap + 1]
        self.closed.extend(t for t in self.tracks if t not in alive)
        self.tracks = alive
        pairs = []
        for i, o in enumerate(obs):
            for j, t in enumerate(self.tracks):
                d = ((o["center"][0] - t.center[0]) ** 2 + (o["center"][1] - t.center[1]) ** 2) ** 0.5
                gate = 0.6 * max(o["width"], t.width)
                if d < gate:
                    pairs.append((d, i, j))
        pairs.sort()
        ids = [-1] * len(obs)
        used_t = set()
        for d, i, j in pairs:
            if ids[i] >= 0 or j in used_t:
                continue
            ids[i] = self.tracks[j].track_id
            used_t.add(j)
            t = self.tracks[j]
            t.center, t.width, t.last_frame = obs[i]["center"], 0.7 * t.width + 0.3 * obs[i]["width"], frame
            t.frames.append(frame)
        for i, o in enumerate(obs):
            if ids[i] < 0:
                t = Track(self.next_id, self.shot, o["center"], o["width"], frame, [frame])
                self.next_id += 1
                self.tracks.append(t)
                ids[i] = t.track_id
        return ids

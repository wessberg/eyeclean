"""Decode stills (and, later, video frames) whose codec has no GPU decoder: PNG, WebP, TIFF, BMP, CMYK JPEG, ...

PyAV (FFmpeg) does the entropy decoding on the CPU and hands back the stored code values.  They cross to the GPU once,
as raw bytes (declared boundary ``cpu_decode_fallback_upload``); unpacking 16-bit samples and every later pixel
operation happen on CUDA.  No pixel arithmetic runs on the host.  This package sits outside the audited hot path on
purpose (it imports numpy / av); the run report records that the fallback was used.
"""

from __future__ import annotations

import av
import numpy as np
import torch

from ..gpu.audit import boundary
from ..io.errors import InputError

_ALPHA = ("rgba", "bgra", "argb", "abgr", "ya8", "ya16", "yuva", "rgba64", "bgra64", "gbrap")


def decode_still(path: str, dev: torch.device) -> tuple[torch.Tensor, torch.Tensor | None, int]:
    """-> (codes (3,H,W) int32 CUDA, alpha (H,W) int32 CUDA or None, bit depth 8 | 16)."""
    try:
        with av.open(path) as c:
            stream = c.streams.video[0]
            frame = next(c.decode(stream))
    except (av.error.FFmpegError, StopIteration, IndexError) as e:
        raise InputError(f"{path}: could not decode the image ({e})") from None
    name = frame.format.name
    deep = any(k in name for k in ("48", "64", "16", "10", "12")) and not name.startswith("pal")
    alpha = any(name.startswith(k) for k in _ALPHA)
    fmt = ("rgba64le" if alpha else "rgb48le") if deep else ("rgba" if alpha else "rgb24")
    arr = frame.to_ndarray(format=fmt)                     # (H, W, C) uint8 / uint16, as stored
    H, W, C = arr.shape
    raw = np.ascontiguousarray(arr).view(np.uint8).reshape(-1)
    with boundary("cpu_decode_fallback_upload"):
        g = torch.from_numpy(raw).to(dev)
    if deep:
        g = g.view(H, W, C, 2).to(torch.int32)
        px = g[..., 0] + 256 * g[..., 1]
        depth = 16
    else:
        px = g.view(H, W, C).to(torch.int32)
        depth = 8
    px = px.permute(2, 0, 1).contiguous()
    return px[:3], (px[3] if C == 4 else None), depth

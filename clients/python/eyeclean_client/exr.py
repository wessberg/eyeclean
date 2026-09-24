"""Read EyeClean's output EXRs with numpy only (no torch, no OpenEXR library).

Supports what EyeClean writes: single-part scanline files, NO_COMPRESSION / ZIPS / ZIP, HALF / FLOAT / UINT channels,
any dataWindow.  Returns channels as (H, W) arrays of the file's own dtype plus the data and display windows.
This is file decoding (zlib + OpenEXR's byte pre-filter), done in the caller's process.
"""
from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass, field
from typing import Dict, Tuple

MAGIC = 20000630
_LINES = {0: 1, 2: 1, 3: 16}
_NAMES = {0: "NO_COMPRESSION", 1: "RLE", 2: "ZIPS", 3: "ZIP", 4: "PIZ", 5: "PXR24", 6: "B44", 7: "B44A", 8: "DWAA", 9: "DWAB"}


class ExrReadError(ValueError):
    pass


@dataclass
class Exr:
    channels: Dict[str, "object"] = field(default_factory=dict)   # name -> numpy (H, W)
    data_window: Tuple[int, int, int, int] = (0, 0, 0, 0)          # xmin, ymin, xmax, ymax (inclusive, as in the file)
    display_window: Tuple[int, int, int, int] = (0, 0, 0, 0)
    attributes: Dict[str, object] = field(default_factory=dict)

    @property
    def box(self) -> Tuple[int, int, int, int]:
        """[x0, y0, x1, y1) in full-frame pixels, the convention of patches/index.json's data_window."""
        x0, y0, x1, y1 = self.data_window
        return x0, y0, x1 + 1, y1 + 1


def _cstr(b: bytes, o: int):
    e = b.index(b"\0", o)
    return b[o:e].decode("latin-1"), e + 1


def read_exr(path) -> Exr:
    import numpy as np

    with open(path, "rb") as f:
        b = f.read()
    magic, ver = struct.unpack_from("<ii", b, 0)
    if magic != MAGIC:
        raise ExrReadError(f"{path}: not an OpenEXR file")
    if ver & 0x1A00:
        raise ExrReadError(f"{path}: tiled, deep or multipart EXR is not supported by this reader")
    o, attrs, chans = 8, {}, []
    while b[o] != 0:
        name, o = _cstr(b, o)
        typ, o = _cstr(b, o)
        (size,) = struct.unpack_from("<i", b, o)
        o += 4
        data = b[o:o + size]
        o += size
        if typ == "chlist":
            q = 0
            while data[q] != 0:
                cn, q = _cstr(data, q)
                pt, _, xs, ys = struct.unpack_from("<iB3xii", data, q)
                q += 16
                chans.append((cn, pt))
        elif typ == "box2i":
            attrs[name] = struct.unpack("<4i", data)
        elif typ == "compression":
            attrs[name] = data[0]
        elif typ == "string":
            attrs[name] = data.decode("utf-8", "replace")
        elif typ == "float":
            attrs[name] = struct.unpack("<f", data)[0]
    o += 1
    comp = attrs.get("compression", 0)
    if comp not in _LINES:
        raise ExrReadError(f"{path}: compression {_NAMES.get(comp, comp)} is not supported (EyeClean writes none / zips / zip)")
    dw, disp = attrs["dataWindow"], attrs["displayWindow"]
    W, H = dw[2] - dw[0] + 1, dw[3] - dw[1] + 1
    lpc = _LINES[comp]
    n = (H + lpc - 1) // lpc
    table = struct.unpack_from(f"<{n}Q", b, o)
    chans.sort(key=lambda c: c[0])
    nbytes = {0: 4, 1: 2, 2: 4}
    dtypes = {0: np.uint32, 1: np.float16, 2: np.float32}
    row = sum(nbytes[pt] * W for _, pt in chans)
    payload = np.empty((H, row), np.uint8)
    for off in table:
        y, size = struct.unpack_from("<ii", b, off)
        k = (y - dw[1]) // lpc
        lines = min(lpc, H - k * lpc)
        raw = b[off + 8: off + 8 + size]
        expected = lines * row
        if comp and size != expected:
            d = np.frombuffer(zlib.decompress(raw), np.uint8).astype(np.int32)
            if d.size != expected:
                raise ExrReadError(f"{path}: chunk at y {y} inflates to {d.size} bytes, expected {expected}")
            d[1:] -= 128
            t = (np.cumsum(d) & 255).astype(np.uint8)
            half = (t.size + 1) // 2
            u = np.empty_like(t)
            u[0::2], u[1::2] = t[:half], t[half:]
        else:
            u = np.frombuffer(raw, np.uint8)
        payload[k * lpc: k * lpc + lines] = u.reshape(lines, row)
    out = Exr(data_window=tuple(dw), display_window=tuple(disp),
              attributes={k: v for k, v in attrs.items() if k not in ("dataWindow", "displayWindow")})
    c = 0
    for name, pt in chans:
        w = nbytes[pt] * W
        out.channels[name] = np.ascontiguousarray(payload[:, c:c + w]).view(dtypes[pt]).reshape(H, W)
        c += w
    return out

"""Minimal strict OpenEXR adapter: scanline HALF/FLOAT channels, uncompressed or ZIP / ZIPS.

Layout follows the OpenEXR file-layout document (magic, version flags, attribute header, offset
table, chunks of ``[part?][y][size][channel rows...]``).  The CPU parses/writes headers, offset
tables and raw bytes, and runs zlib (inflate / deflate: byte I/O); channel (de)interleaving, the ZIP
byte predictor and even/odd split, dtype conversion and packing of pixel payloads run on CUDA.
Supported:

* ``read_exr`` (strict user-input adapter): single-part, scanline, NO_COMPRESSION, ZIPS (1 line per
  chunk) or ZIP (16 lines per chunk), HALF/FLOAT, x/y sampling 1, any data window.  Multipart,
  tiled, deep and other compressions (PIZ, DWAA/B, B44, PXR24, RLE, HTJ2K) are rejected with a
  conversion hint.
* ``read_render_exr`` (declared renderer I/O boundary for our own Blender outputs): additionally
  accepts scanline multipart files, which Blender 5.2 writes for multilayer EXR.
* ``write_exr``: single-part scanline writer (NO_COMPRESSION by default, or lossless ZIPS / ZIP) with
  cropped ``dataWindow`` and full-frame ``displayWindow``.
"""

from __future__ import annotations

import os
import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from ..gpu.audit import boundary
from .rawbytes import bytes_to_cuda, cuda_to_bytes

MAGIC = 20000630
FLAG_TILED = 0x200
FLAG_LONG_NAMES = 0x400
FLAG_DEEP = 0x800
FLAG_MULTIPART = 0x1000
PIXEL_UINT, PIXEL_HALF, PIXEL_FLOAT = 0, 1, 2
_PIXEL_BYTES = {PIXEL_UINT: 4, PIXEL_HALF: 2, PIXEL_FLOAT: 4}
_PIXEL_DTYPE = {PIXEL_UINT: torch.int32, PIXEL_HALF: torch.float16, PIXEL_FLOAT: torch.float32}
COMPRESSION_NAMES = {0: "NO_COMPRESSION", 1: "RLE", 2: "ZIPS", 3: "ZIP", 4: "PIZ", 5: "PXR24", 6: "B44", 7: "B44A", 8: "DWAA", 9: "DWAB", 10: "HTJ2K256", 11: "HTJ2K32"}
NO_COMPRESSION, ZIPS_COMPRESSION, ZIP_COMPRESSION = 0, 2, 3
LINES_PER_CHUNK = {NO_COMPRESSION: 1, ZIPS_COMPRESSION: 1, ZIP_COMPRESSION: 16}  # the compressions read and written here


class ExrError(ValueError):
    """Unsupported or malformed EXR (clear, non-zero CLI exit)."""


@dataclass
class Channel:
    name: str
    pixel_type: int
    x_sampling: int = 1
    y_sampling: int = 1
    p_linear: int = 0


@dataclass
class ExrHeader:
    attributes: dict[str, Any]
    channels: list[Channel]
    data_window: tuple[int, int, int, int]  # xmin, ymin, xmax, ymax (inclusive)
    display_window: tuple[int, int, int, int]
    compression: int
    line_order: int
    part_name: str | None = None

    @property
    def width(self) -> int:
        return self.data_window[2] - self.data_window[0] + 1

    @property
    def height(self) -> int:
        return self.data_window[3] - self.data_window[1] + 1


@dataclass
class ExrImage:
    header: ExrHeader
    channels: dict[str, torch.Tensor] = field(default_factory=dict)  # name -> (H, W) CUDA tensor

    @property
    def data_window(self) -> tuple[int, int, int, int]:
        return self.header.data_window

    @property
    def display_window(self) -> tuple[int, int, int, int]:
        return self.header.display_window


# ------------------------------------------------------------------ header parsing (CPU metadata)
def _cstr(buf: bytes, off: int) -> tuple[str, int]:
    end = buf.index(b"\0", off)
    return buf[off:end].decode("latin-1"), end + 1


def _parse_attr_value(typ: str, data: bytes) -> Any:
    if typ == "box2i":
        return struct.unpack("<4i", data)
    if typ == "box2f":
        return struct.unpack("<4f", data)
    if typ in ("compression", "lineOrder", "envmap", "deepImageState"):
        return data[0]
    if typ == "float":
        return struct.unpack("<f", data)[0]
    if typ == "double":
        return struct.unpack("<d", data)[0]
    if typ == "int":
        return struct.unpack("<i", data)[0]
    if typ == "v2f":
        return struct.unpack("<2f", data)
    if typ == "v2i":
        return struct.unpack("<2i", data)
    if typ == "v3f":
        return struct.unpack("<3f", data)
    if typ == "string":
        return data.decode("utf-8", errors="replace")
    if typ == "chlist":
        chans = []
        o = 0
        while data[o] != 0:
            name, o = _cstr(data, o)
            pt, plin, xs, ys = struct.unpack_from("<iB3xii", data, o)
            o += 16
            chans.append(Channel(name, pt, xs, ys, plin))
        return chans
    return ("<raw>", typ, len(data))


def _parse_header(buf: bytes, off: int) -> tuple[dict[str, tuple[str, Any]], int]:
    attrs: dict[str, tuple[str, Any]] = {}
    while True:
        if buf[off] == 0:
            return attrs, off + 1
        name, off = _cstr(buf, off)
        typ, off = _cstr(buf, off)
        (size,) = struct.unpack_from("<i", buf, off)
        off += 4
        attrs[name] = (typ, _parse_attr_value(typ, buf[off:off + size]))
        off += size


def _to_header(attrs: dict[str, tuple[str, Any]], path: str) -> ExrHeader:
    for req in ("channels", "compression", "dataWindow", "displayWindow", "lineOrder"):
        if req not in attrs:
            raise ExrError(f"{path}: missing required attribute {req!r}")
    part_name = attrs["name"][1] if "name" in attrs else None
    if "type" in attrs and attrs["type"][1] not in ("scanlineimage",):
        raise ExrError(f"{path}: part type {attrs['type'][1]!r} unsupported (only scanlineimage)")
    return ExrHeader(
        attributes={k: v[1] for k, v in attrs.items()},
        channels=attrs["channels"][1],
        data_window=tuple(attrs["dataWindow"][1]),
        display_window=tuple(attrs["displayWindow"][1]),
        compression=int(attrs["compression"][1]),
        line_order=int(attrs["lineOrder"][1]),
        part_name=part_name,
    )


def probe_exr(path: str | os.PathLike) -> dict:
    """Header-only probe (CPU metadata)."""
    path = str(path)
    with open(path, "rb") as f:
        head = f.read(1 << 20)
    magic, ver = struct.unpack_from("<ii", head, 0)
    if magic != MAGIC:
        raise ExrError(f"{path}: not an OpenEXR file")
    flags = ver & ~0xFF
    off = 8
    parts = []
    if flags & FLAG_MULTIPART:
        while head[off] != 0:
            a, off = _parse_header(head, off)
            parts.append(_to_header(a, path))
        off += 1
    else:
        a, off = _parse_header(head, off)
        parts.append(_to_header(a, path))
    return {"version": ver & 0xFF, "tiled": bool(flags & FLAG_TILED), "long_names": bool(flags & FLAG_LONG_NAMES),
            "deep": bool(flags & FLAG_DEEP), "multipart": bool(flags & FLAG_MULTIPART), "parts": parts, "header_end": off}


def _validate_part(h: ExrHeader, path: str) -> None:
    if h.compression not in LINES_PER_CHUNK:
        raise ExrError(
            f"{path}: compression {COMPRESSION_NAMES.get(h.compression, h.compression)} is unsupported by the strict "
            "GPU EXR adapter (supported: NO_COMPRESSION, ZIPS, ZIP); convert e.g. with "
            "`oiiotool in.exr --compression zip -o out.exr`"
        )
    if h.line_order not in (0, 1):
        raise ExrError(f"{path}: lineOrder {h.line_order} unsupported for scanline files")
    for c in h.channels:
        if c.pixel_type not in (PIXEL_HALF, PIXEL_FLOAT, PIXEL_UINT):
            raise ExrError(f"{path}: channel {c.name!r} pixel type {c.pixel_type} unsupported")
        if c.x_sampling != 1 or c.y_sampling != 1:
            raise ExrError(f"{path}: channel {c.name!r} has subsampling {c.x_sampling}x{c.y_sampling}; only unit sampling is supported")


# ------------------------------------------------------------------ reading
def _read_parts(path: str, allow_multipart: bool, device: torch.device, want: set[str] | None,
                boundary_name: str) -> list[ExrImage]:
    info = probe_exr(path)
    if info["tiled"]:
        raise ExrError(f"{path}: tiled EXR is unsupported; use scanline storage")
    if info["deep"]:
        raise ExrError(f"{path}: deep EXR is unsupported")
    if info["multipart"] and not allow_multipart:
        raise ExrError(f"{path}: multipart EXR is unsupported by the strict input adapter; provide single-part files")
    parts: list[ExrHeader] = info["parts"]
    for h in parts:
        _validate_part(h, path)
    size = os.path.getsize(path)
    blob = bytearray(size)
    with open(path, "rb") as f:
        if f.readinto(blob) != size:
            raise ExrError(f"{path}: short read")
    off = info["header_end"]
    # offset tables (CPU metadata): one uint64 per scanline chunk, per part
    tables = []
    for h in parts:
        lpc = LINES_PER_CHUNK[h.compression]
        n = (h.height + lpc - 1) // lpc
        if "chunkCount" in h.attributes and int(h.attributes["chunkCount"]) != n:
            raise ExrError(f"{path}: chunkCount {h.attributes['chunkCount']} != {n} chunks of {lpc} line(s) for "
                           f"{COMPRESSION_NAMES[h.compression]}")
        tables.append(struct.unpack_from(f"<{n}Q", blob, off))
        off += 8 * n
    prefix = 12 if info["multipart"] else 8
    images = []
    for pi, (h, table) in enumerate(zip(parts, tables)):
        chans = h.channels
        if want is not None:
            if not any((h.part_name or "") + "." + c.name in want or c.name in want for c in chans):
                continue
        W, H = h.width, h.height
        row_payload = sum(_PIXEL_BYTES[c.pixel_type] * W for c in chans)
        if h.compression != NO_COMPRESSION:
            payload = _zip_payload(blob, h, table, pi, prefix, row_payload, info["multipart"], device, boundary_name, path)
            images.append(_split_channels(h, payload, W, info["multipart"], want))
            continue
        row_bytes = prefix + row_payload
        start = min(table)
        # validate the (metadata) layout: H chunks of row_bytes, contiguous in file order
        sorted_offs = sorted(table)
        for k, o in enumerate(sorted_offs):
            if o != start + k * row_bytes:
                raise ExrError(f"{path}: part {pi}: non-contiguous chunk layout; unsupported by the strict adapter")
        if start + H * row_bytes > len(blob):
            raise ExrError(f"{path}: truncated pixel data")
        raw = memoryview(blob)[start:start + H * row_bytes]
        rows = bytes_to_cuda(raw, device, boundary_name).view(H, row_bytes)
        # validate chunk prefixes on the GPU: part number (multipart), y, and size
        pref = rows[:, :prefix].contiguous().view(torch.int32)  # (H, prefix/4)
        ys = pref[:, -2] - h.data_window[1]
        sizes = pref[:, -1]
        ok = (sizes == row_payload).all() & (ys >= 0).all() & (ys < H).all()
        if info["multipart"]:
            ok = ok & (pref[:, 0] == pi).all()
        # permutation check: every row index present exactly once
        seen = torch.zeros(H, dtype=torch.int32, device=device)
        seen.index_add_(0, ys.clamp(0, H - 1).long(), torch.ones_like(ys))
        ok = ok & (seen == 1).all()
        if not bool(ok.item()):
            raise ExrError(f"{path}: part {pi}: invalid scanline chunk headers")
        order = torch.empty(H, dtype=torch.long, device=device)
        order[ys.long()] = torch.arange(H, device=device)
        payload = rows[:, prefix:].index_select(0, order)
        images.append(_split_channels(h, payload, W, info["multipart"], want))
    return images


def _split_channels(h: ExrHeader, payload: torch.Tensor, W: int, multipart: bool, want: set[str] | None) -> ExrImage:
    """(H, row_payload) uint8 scanlines in y order -> native-dtype (H, W) channel views."""
    img = ExrImage(h)
    o = 0
    for c in h.channels:
        nb = _PIXEL_BYTES[c.pixel_type] * W
        key = c.name
        if multipart and h.part_name and not c.name.startswith(h.part_name + "."):
            key = f"{h.part_name}.{c.name}"
        if want is None or key in want or c.name in want:
            img.channels[key] = payload[:, o:o + nb].contiguous().view(_PIXEL_DTYPE[c.pixel_type])
        o += nb
    return img


def _unpredict_deinterleave(d: torch.Tensor) -> torch.Tensor:
    """Undo OpenEXR's ZIP pre-filter on (m, L) uint8 CUDA chunks: the byte predictor
    (t[i] = t[i-1] + d[i] - 128, mod 256), then the even/odd byte split (first half = even bytes)."""
    t = d.to(torch.int32)
    t[:, 1:] -= 128
    t = torch.cumsum(t, dim=1).bitwise_and_(255).to(torch.uint8)
    m, L = t.shape
    half = (L + 1) // 2
    out = torch.empty_like(t)
    out[:, 0::2] = t[:, :half]
    out[:, 1::2] = t[:, half:]
    return out


def _zip_payload(blob: bytearray, h: ExrHeader, table, pi: int, prefix: int, row_payload: int, multipart: bool,
                 device: torch.device, boundary_name: str, path: str) -> torch.Tensor:
    """ZIPS / ZIP scanline chunks -> (H, row_payload) uint8 CUDA scanlines in y order.

    CPU: chunk headers (metadata) and zlib inflation (byte I/O).  CUDA: the predictor and byte
    de-interleave, and the placement of every chunk's lines.  A chunk whose stored size equals its
    raw size is stored uncompressed and unfiltered, as the OpenEXR layout specifies."""
    lpc = LINES_PER_CHUNK[h.compression]
    H = h.height
    n = len(table)
    groups: dict[tuple[int, bool], list[tuple[int, bytes]]] = {}
    seen = [False] * n
    for off in table:
        if multipart:
            (part,) = struct.unpack_from("<i", blob, off)
            if part != pi:
                raise ExrError(f"{path}: part {pi}: chunk at {off} belongs to part {part}")
        y, size = struct.unpack_from("<ii", blob, off + prefix - 8)
        rel = y - h.data_window[1]
        if rel < 0 or rel % lpc or rel // lpc >= n:
            raise ExrError(f"{path}: part {pi}: invalid chunk y {y}")
        k = rel // lpc
        if seen[k]:
            raise ExrError(f"{path}: part {pi}: duplicate chunk for y {y}")
        seen[k] = True
        expected = min(lpc, H - k * lpc) * row_payload
        start = off + prefix
        if size < 0 or start + size > len(blob):
            raise ExrError(f"{path}: part {pi}: truncated chunk at y {y}")
        data = bytes(blob[start:start + size])
        stored = size == expected
        if not stored:
            try:
                data = zlib.decompress(data)
            except zlib.error as e:
                raise ExrError(f"{path}: part {pi}: corrupt ZIP chunk at y {y}: {e}") from None
            if len(data) != expected:
                raise ExrError(f"{path}: part {pi}: ZIP chunk at y {y} inflates to {len(data)} bytes, expected {expected}")
        groups.setdefault((expected, stored), []).append((k, data))
    if not all(seen):
        raise ExrError(f"{path}: part {pi}: missing scanline chunks")
    payload = torch.empty((H, row_payload), dtype=torch.uint8, device=device)
    for (length, stored), items in groups.items():
        d = bytes_to_cuda(b"".join(data for _, data in items), device, boundary_name).view(len(items), length)
        if not stored:
            d = _unpredict_deinterleave(d)
        lines = length // row_payload
        with boundary(boundary_name):                               # chunk indices (metadata)
            ks = torch.tensor([k for k, _ in items], dtype=torch.long).to(device)
        rows = (ks[:, None] * lpc + torch.arange(lines, device=device)[None, :]).flatten()
        payload[rows] = d.view(len(items) * lines, row_payload)
    return payload


def read_exr(path: str | os.PathLike, device: torch.device, want: set[str] | None = None) -> ExrImage:
    """Strict single-part input adapter. Returns native-dtype channels as (H, W) CUDA tensors."""
    return _read_parts(str(path), allow_multipart=False, device=device, want=want, boundary_name="exr_payload_upload")[0]


def read_render_exr(path: str | os.PathLike, device: torch.device, want: set[str] | None = None) -> dict[str, torch.Tensor]:
    """Renderer I/O boundary for our own uncompressed Blender outputs (multipart allowed).

    Returns {"<part>.<channel>": (H, W) CUDA tensor}; single-part files use bare channel names.
    """
    out: dict[str, torch.Tensor] = {}
    for img in _read_parts(str(path), allow_multipart=True, device=device, want=want, boundary_name="render_buffer_upload"):
        out.update(img.channels)
    return out


# ------------------------------------------------------------------ writing
def _attr(name: str, typ: str, payload: bytes) -> bytes:
    return name.encode() + b"\0" + typ.encode() + b"\0" + struct.pack("<i", len(payload)) + payload


def _chlist(channels: list[Channel]) -> bytes:
    b = b""
    for c in channels:
        b += c.name.encode() + b"\0" + struct.pack("<iB3xii", c.pixel_type, c.p_linear, c.x_sampling, c.y_sampling)
    return b + b"\0"


def pack_scanlines(planes: list[torch.Tensor], y0: int) -> torch.Tensor:
    """GPU packing of channel planes (already in sorted channel order, native dtypes, (H, W))
    into NO_COMPRESSION scanline chunks: per row ``int32 y, int32 size, channel rows``."""
    H, W = planes[0].shape
    dev = planes[0].device
    rows = [p.contiguous().view(torch.uint8).view(H, -1) for p in planes]
    payload = torch.cat(rows, dim=1) if len(rows) > 1 else rows[0]
    size = payload.shape[1]
    pref = torch.stack([torch.arange(y0, y0 + H, device=dev, dtype=torch.int32),
                        torch.full((H,), size, device=dev, dtype=torch.int32)], dim=1)
    return torch.cat([pref.contiguous().view(torch.uint8).view(H, 8), payload], dim=1)


def _predict_interleave(raw: torch.Tensor) -> torch.Tensor:
    """OpenEXR's ZIP pre-filter on (m, L) uint8 CUDA chunks: even bytes then odd bytes, then the byte
    predictor d[i] = t[i] - t[i-1] + 128 (mod 256).  Inverse of ``_unpredict_deinterleave``."""
    t = torch.cat([raw[:, 0::2], raw[:, 1::2]], dim=1).to(torch.int32)
    d = t.clone()
    d[:, 1:] = (t[:, 1:] - t[:, :-1] + 128).bitwise_and_(255)
    return d.to(torch.uint8)


def _zip_chunks(planes: list[torch.Tensor], y0: int, compression: int, header_len: int) -> tuple[bytes, bytes]:
    """ZIPS / ZIP chunks: GPU packing and pre-filter; CPU zlib deflate (byte I/O).  A chunk that does
    not shrink is stored raw and unfiltered, as the OpenEXR layout specifies."""
    H, W = planes[0].shape
    rows = [p.contiguous().view(torch.uint8).view(H, -1) for p in planes]
    payload = torch.cat(rows, dim=1) if len(rows) > 1 else rows[0]
    row_payload = payload.shape[1]
    lpc = LINES_PER_CHUNK[compression]
    n = (H + lpc - 1) // lpc
    full = H // lpc
    parts = []
    if full:
        raw = payload[: full * lpc].reshape(full, lpc * row_payload)
        parts.append((raw, _predict_interleave(raw)))
    if full < n:
        raw = payload[full * lpc:].reshape(1, -1)
        parts.append((raw, _predict_interleave(raw)))
    chunks = []
    for raw, filt in parts:
        m, L = raw.shape
        rb, fb = cuda_to_bytes(raw.reshape(-1)), cuda_to_bytes(filt.reshape(-1))
        for i in range(m):
            c = zlib.compress(bytes(fb[i * L:(i + 1) * L]), 6)
            chunks.append(c if len(c) < L else bytes(rb[i * L:(i + 1) * L]))
    off = header_len + 8 * n
    offsets, body = [], []
    for k, c in enumerate(chunks):
        offsets.append(off)
        body.append(struct.pack("<ii", y0 + k * lpc, len(c)) + c)
        off += 8 + len(c)
    return struct.pack(f"<{n}Q", *offsets), b"".join(body)


def write_exr(
    path: str | os.PathLike,
    channels: dict[str, torch.Tensor],
    display_size: tuple[int, int] | None = None,
    origin: tuple[int, int] = (0, 0),
    pixel_type: int = PIXEL_FLOAT,
    attributes: dict[str, str] | None = None,
    overwrite_guard: set[str] | None = None,
    compression: int = NO_COMPRESSION,
) -> dict:
    """Write a single-part scanline EXR (NO_COMPRESSION by default; ZIPS / ZIP are lossless).

    channels: name -> (H, W) CUDA tensor (converted to FLOAT/HALF on the GPU).
    display_size: (width, height) of the full frame; dataWindow is the crop at ``origin``.
    """
    path = str(path)
    if overwrite_guard and os.path.abspath(path) in overwrite_guard:
        raise ExrError(f"refusing to overwrite protected input path {path}")
    if not channels:
        raise ExrError("no channels")
    names = sorted(channels)
    H, W = channels[names[0]].shape
    for n in names:
        t = channels[n]
        if t.shape != (H, W):
            raise ExrError(f"channel {n} shape {tuple(t.shape)} != {(H, W)}")
        if not t.is_cuda:
            raise ExrError(f"channel {n} is not a CUDA tensor (strict GPU packing)")
    dtype = _PIXEL_DTYPE[pixel_type]
    planes = [channels[n].to(dtype) for n in names]
    x0, y0 = origin
    dw = (x0, y0, x0 + W - 1, y0 + H - 1)
    if display_size is None:
        display_size = (x0 + W, y0 + H)
    disp = (0, 0, display_size[0] - 1, display_size[1] - 1)
    chans = [Channel(n, pixel_type) for n in names]
    attrs = [
        _attr("channels", "chlist", _chlist(chans)),
        _attr("compression", "compression", bytes([0])),
        _attr("dataWindow", "box2i", struct.pack("<4i", *dw)),
        _attr("displayWindow", "box2i", struct.pack("<4i", *disp)),
        _attr("lineOrder", "lineOrder", bytes([0])),
        _attr("pixelAspectRatio", "float", struct.pack("<f", 1.0)),
        _attr("screenWindowCenter", "v2f", struct.pack("<2f", 0.0, 0.0)),
        _attr("screenWindowWidth", "float", struct.pack("<f", 1.0)),
    ]
    if compression not in LINES_PER_CHUNK:
        raise ExrError(f"write_exr: compression {compression!r} unsupported (NO_COMPRESSION, ZIPS or ZIP)")
    attrs[1] = _attr("compression", "compression", bytes([compression]))
    for k, v in sorted((attributes or {}).items()):
        attrs.append(_attr(k, "string", str(v).encode("utf-8")))
    long_names = any(len(n) > 31 for n in names) or any(len(k) > 31 for k in (attributes or {}))
    version = 2 | (FLAG_LONG_NAMES if long_names else 0)
    header = struct.pack("<ii", MAGIC, version) + b"".join(attrs) + b"\0"
    if compression == NO_COMPRESSION:
        chunks = pack_scanlines(planes, y0)
        row_bytes = chunks.shape[1]
        base = len(header) + 8 * H
        offsets = struct.pack(f"<{H}Q", *(base + k * row_bytes for k in range(H)))  # chunk offsets (metadata)
        body = cuda_to_bytes(chunks.view(-1))
    else:
        offsets, body = _zip_chunks(planes, y0, compression, len(header))
    tmp = path + ".part"
    with open(tmp, "wb") as f:
        f.write(header)
        f.write(offsets)
        f.write(body)
        f.flush()
        os.fsync(f.fileno())  # survive power loss: never publish a file whose data is unflushed
    os.replace(tmp, path)
    return {"path": path, "data_window": dw, "display_window": disp, "channels": names,
            "pixel_type": {PIXEL_FLOAT: "FLOAT", PIXEL_HALF: "HALF", PIXEL_UINT: "UINT"}[pixel_type],
            "bytes": len(header) + len(offsets) + len(body)}


def write_rgb_exr(path: str | os.PathLike, rgb: torch.Tensor, alpha: torch.Tensor | None = None, **kw) -> dict:
    """rgb: (3, H, W) CUDA tensor -> R, G, B (+A) FLOAT channels."""
    ch = {"R": rgb[0], "G": rgb[1], "B": rgb[2]}
    if alpha is not None:
        ch["A"] = alpha
    return write_exr(path, ch, **kw)


def read_rgb_exr(path: str | os.PathLike, device: torch.device) -> tuple[torch.Tensor, torch.Tensor | None, ExrHeader]:
    img = read_exr(path, device)
    names = set(img.channels)
    if not {"R", "G", "B"} <= names:
        raise ExrError(f"{path}: expected R, G, B channels, found {sorted(names)}")
    rgb = torch.stack([img.channels[c].float() for c in ("R", "G", "B")])
    a = img.channels["A"].float() if "A" in names else None
    return rgb, a, img.header


def as_path(p: str | os.PathLike) -> Path:
    return Path(p)

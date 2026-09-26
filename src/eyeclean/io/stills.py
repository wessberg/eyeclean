"""Still images in and out: JPEG, PNG, WebP, TIFF, BMP (HEIC / AVIF: planned).

Reading:
  - the format is taken from the file's magic bytes, not its extension;
  - JPEG decodes on the GPU (nvJPEG); formats without a GPU decoder decode on the CPU through PyAV
    (`eyeclean.fallback.decode`) and are uploaded at a declared boundary.  The run report records which;
  - the EXIF orientation is applied (on the GPU), so faces are upright for detection;
  - colour comes from the file: an ICC matrix/TRC profile, a PNG cICP / sRGB chunk, or - untagged - sRGB, which is
    the documented convention for untagged images and is recorded as an assumption.
Writing (`write_still`): the same format when EyeClean can encode it (JPEG on the GPU, PNG), else PNG.  Outside the
edit support the ORIGINAL decoded code values are written back unchanged, so a lossless output (PNG) is bit-identical
there; a JPEG output is re-encoded (lossy) as a whole.  EXIF (orientation reset to 1, since the pixels are written
upright) and the ICC profile are carried over.  Parsing and container bytes are host I/O; all pixel work is CUDA.
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import torch

from ..color import icc as ICC
from ..gpu.audit import boundary
from ..gpu.runtime import GpuPolicyError
from .errors import InputError
from .rawbytes import cuda_to_bytes

STILL_EXTS = {".jpg", ".jpeg", ".jpe", ".jfif", ".png", ".webp", ".tif", ".tiff", ".bmp", ".heic", ".heif", ".avif", ".hif"}


def sniff(head: bytes) -> str | None:
    if head[:3] == b"\xff\xd8\xff":
        return "jpeg"
    if head[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "webp"
    if head[:4] in (b"II*\0", b"MM\0*"):
        return "tiff"
    if head[:2] == b"BM":
        return "bmp"
    if head[4:8] == b"ftyp":
        brand = head[8:12]
        if brand in (b"avif", b"avis"):
            return "avif"
        if brand in (b"heic", b"heix", b"hevc", b"hevx", b"mif1", b"msf1", b"heim", b"heis"):
            return "heif"
    return None


def is_still(path: str) -> bool:
    p = Path(path)
    if not p.is_file():
        return False
    with open(p, "rb") as f:
        fmt = sniff(f.read(16))
    return fmt is not None or p.suffix.lower() in STILL_EXTS


@dataclass
class StillInfo:
    path: str
    format: str                    # jpeg | png | webp | tiff | bmp | heif | avif
    width: int                     # after orientation
    height: int
    bit_depth: int
    orientation: int = 1           # EXIF 1..8, already applied to the pixels
    colour: str = ""               # human-readable: where the colour came from
    colour_assumed: bool = False
    decode: str = ""               # nvjpeg | pyav-cpu
    icc: bytes = b""
    exif: bytes = b""              # the APP1 payload (starting with b"Exif\0\0"), JPEG only
    profile: object = field(default=None, repr=False)   # IccProfile used for conversion
    has_gain_map: bool = False     # Ultra HDR / gain-map JPEG (the SDR base is processed; the gain map is not)

    def record(self) -> dict:
        return {"format": self.format, "width": self.width, "height": self.height, "bit_depth": self.bit_depth,
                "orientation_applied": self.orientation, "colour": self.colour, "colour_assumed": self.colour_assumed,
                "decode": self.decode, "icc_bytes": len(self.icc), "has_gain_map": self.has_gain_map}


# ----------------------------------------------------------------------------- container parsing (host bytes)
def _jpeg_segments(b: bytes):
    i = 2
    while i + 4 <= len(b) and b[i] == 0xFF:
        m = b[i + 1]
        if m in (0xD9, 0xDA):
            return
        if m == 0x01 or 0xD0 <= m <= 0xD7:
            i += 2
            continue
        L = struct.unpack(">H", b[i + 2:i + 4])[0]
        yield m, b[i + 4:i + 2 + L], i
        i += 2 + L


def _exif_orientation(exif: bytes) -> tuple[int, int | None]:
    """(orientation, byte offset of its value inside `exif`) from an APP1 payload starting with b'Exif\\0\\0'."""
    t = exif[6:]
    if len(t) < 8:
        return 1, None
    e = "<" if t[:2] == b"II" else ">"
    off = struct.unpack(e + "I", t[4:8])[0]
    if off + 2 > len(t):
        return 1, None
    n = struct.unpack(e + "H", t[off:off + 2])[0]
    for k in range(n):
        o = off + 2 + 12 * k
        if o + 12 > len(t):
            break
        tag, typ, cnt = struct.unpack(e + "HHI", t[o:o + 8])
        if tag == 0x0112 and typ == 3:
            v = struct.unpack(e + "H", t[o + 8:o + 10])[0]
            return (v if 1 <= v <= 8 else 1), 6 + o + 8
    return 1, None


def _jpeg_meta(b: bytes) -> dict:
    icc_chunks, exif, gain = {}, b"", False
    for m, data, _ in _jpeg_segments(b):
        if m == 0xE1 and data[:6] == b"Exif\0\0" and not exif:
            exif = data
        elif m == 0xE1 and b"hdrgm" in data[:4096]:
            gain = True
        elif m == 0xE2 and data[:12] == b"ICC_PROFILE\0":
            icc_chunks[data[12]] = data[14:]
        elif m == 0xE2 and data[:4] == b"MPF\0":
            gain = gain or b"hdrgm" in b
    icc = b"".join(icc_chunks[k] for k in sorted(icc_chunks))
    return {"exif": exif, "icc": icc, "gain_map": gain}


def _png_meta(b: bytes) -> dict:
    out = {"icc": b"", "srgb": False, "cicp": None, "bit_depth": 8, "color_type": 2}
    i = 8
    while i + 8 <= len(b):
        L, typ = struct.unpack(">I4s", b[i:i + 8])
        data = b[i + 8:i + 8 + L]
        if typ == b"IHDR":
            out["bit_depth"], out["color_type"] = data[8], data[9]
        elif typ == b"iCCP":
            z = data.index(b"\0")
            out["icc"] = zlib.decompress(data[z + 2:])  # host byte I/O: a few KB of profile, not pixels
        elif typ == b"sRGB":
            out["srgb"] = True
        elif typ == b"cICP":
            out["cicp"] = tuple(data[:4])
        elif typ in (b"IDAT", b"IEND"):
            break
        i += 12 + L
    return out


# nclx / cICP code points (ITU-T H.273) EyeClean understands for stills
_PRIMARIES = {1: ("BT.709", ((0.64, 0.33), (0.30, 0.60), (0.15, 0.06))), 9: ("BT.2020", ((0.708, 0.292), (0.170, 0.797), (0.131, 0.046))),
              12: ("Display P3", ((0.680, 0.320), (0.265, 0.690), (0.150, 0.060)))}
_SRGB_TRC = ICC.Trc("para", ptype=3, params=(2.4, 1 / 1.055, 0.055 / 1.055, 1 / 12.92, 0.04045))
_709_TRC = ICC.Trc("para", ptype=3, params=(1 / 0.45, 1 / 1.099, 0.099 / 1.099, 1 / 4.5, 0.081))
_TRANSFERS = {1: ("BT.709", _709_TRC), 6: ("BT.709", _709_TRC), 13: ("sRGB", _SRGB_TRC), 8: ("linear", ICC.Trc("identity"))}


def _profile_from_cicp(cicp) -> ICC.IccProfile | None:
    prim, trc = cicp[0], cicp[1]
    if prim in _PRIMARIES and trc in _TRANSFERS:
        name = f"{_PRIMARIES[prim][0]} primaries, {_TRANSFERS[trc][0]} transfer (cICP)"
        return ICC.primaries_profile(name, _PRIMARIES[prim][1], (0.3127, 0.3290), _TRANSFERS[trc][1])
    return None


def _choose_profile(icc: bytes, srgb: bool = False, cicp=None) -> tuple[ICC.IccProfile, str, bool]:
    if cicp is not None:
        p = _profile_from_cicp(cicp)
        if p is not None:
            return p, p.description, False
    if icc:
        try:
            p = ICC.parse_icc(icc)
            return p, f"ICC: {p.description}", False
        except ICC.IccError as e:
            p = ICC.srgb_profile()
            return p, f"unsupported ICC profile ({e}); sRGB assumed", True
    if srgb:
        p = ICC.srgb_profile()
        p.description = "sRGB"
        return p, "sRGB (tagged)", False
    return ICC.srgb_profile(), "untagged: sRGB assumed", True


# ----------------------------------------------------------------------------- orientation (GPU)
def orient(x: torch.Tensor, o: int) -> torch.Tensor:
    """Apply EXIF orientation o (1..8) to a (C,H,W) tensor so the result is upright."""
    if o == 2:
        return x.flip(-1)
    if o == 3:
        return x.flip(-1).flip(-2)
    if o == 4:
        return x.flip(-2)
    if o == 5:
        return x.transpose(-1, -2)
    if o == 6:
        return x.transpose(-1, -2).flip(-1)
    if o == 7:
        return x.transpose(-1, -2).flip(-1).flip(-2)
    if o == 8:
        return x.transpose(-1, -2).flip(-2)
    return x


# ----------------------------------------------------------------------------- reading
def read_still(path: str, dev: torch.device) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, StillInfo]:
    """-> (linear Rec.709 (3,H,W) FP32 CUDA, alpha (H,W) or None, the decoded code values (3,H,W) int32 CUDA upright,
    StillInfo).  The code values let the writer put unedited pixels back exactly."""
    b = Path(path).read_bytes()
    fmt = sniff(b[:16])
    if fmt is None:
        raise InputError(f"{path}: not a recognised image file")
    if fmt in ("heif", "avif"):
        raise InputError(f"{path}: {fmt.upper()} input is not supported yet (planned); convert to JPEG or PNG for now")
    alpha = None
    if fmt == "jpeg":
        meta = _jpeg_meta(b)
        from torchvision.io import ImageReadMode, decode_jpeg

        data = torch.frombuffer(bytearray(b), dtype=torch.uint8)
        try:
            with boundary("image_bitstream_decode"):
                codes = decode_jpeg(data, mode=ImageReadMode.RGB, device=dev)
            decode = "nvjpeg"
        except RuntimeError as e:  # e.g. CMYK or lossless JPEG, which nvJPEG does not handle
            if isinstance(e, GpuPolicyError):
                raise
            from ..fallback.decode import decode_still
            codes, _, _ = decode_still(path, dev)
            decode = "pyav-cpu"
        codes = codes.to(torch.int32)
        depth = 8
        orientation, _ = _exif_orientation(meta["exif"]) if meta["exif"] else (1, None)
        prof, colour, assumed = _choose_profile(meta["icc"])
        exif, icc, gain = meta["exif"], meta["icc"], meta["gain_map"]
    else:
        from ..fallback.decode import decode_still
        codes, alpha, depth = decode_still(path, dev)
        decode = "pyav-cpu"
        orientation, exif, gain = 1, b"", False
        pm = _png_meta(b) if fmt == "png" else {"icc": b"", "srgb": False, "cicp": None}
        prof, colour, assumed = _choose_profile(pm["icc"], pm.get("srgb", False), pm.get("cicp"))
        icc = pm["icc"]
    codes = orient(codes, orientation)
    if alpha is not None:
        alpha = orient(alpha[None], orientation)[0]
    maxv = float((1 << depth) - 1)
    lin = ICC.icc_to_linear709(codes.float() / maxv, prof)
    info = StillInfo(path, fmt, int(codes.shape[2]), int(codes.shape[1]), depth, orientation, colour, assumed, decode,
                     icc, exif, prof, gain)
    return lin, alpha, codes, info


# ----------------------------------------------------------------------------- writing
def output_format(info: StillInfo) -> str:
    return "jpeg" if info.format == "jpeg" else "png"


def output_suffix(info: StillInfo) -> str:
    return {"jpeg": ".jpg", "png": ".png"}[output_format(info)]


def _encode_codes(l709: torch.Tensor, info: StillInfo) -> torch.Tensor:
    maxv = (1 << info.bit_depth) - 1
    enc = ICC.linear709_to_icc(l709, info.profile)
    return (enc.clamp(0, 1) * maxv).round().to(torch.int32)


def _exif_upright(exif: bytes) -> bytes:
    """The EXIF payload with the orientation tag set to 1 (the pixels are written upright)."""
    o, at = _exif_orientation(exif)
    if at is None or o == 1:
        return exif
    e = "<" if exif[6:8] == b"II" else ">"
    return exif[:at] + struct.pack(e + "H", 1) + exif[at + 2:]


def _app(marker: int, payload: bytes) -> bytes:
    return bytes([0xFF, marker]) + struct.pack(">H", len(payload) + 2) + payload


def write_still(dst: Path, out_l709: torch.Tensor, support: torch.Tensor, codes: torch.Tensor, info: StillInfo,
                alpha: torch.Tensor | None = None, jpeg_quality: int = 95) -> dict:
    """Write the edited image: re-encoded code values inside `support`, the ORIGINAL code values outside it."""
    new = _encode_codes(out_l709, info)
    final = torch.where(support[None], new, codes)
    fmt = output_format(info)
    dst = Path(dst)
    if fmt == "jpeg":
        from torchvision.io import encode_jpeg

        u8 = final.clamp(0, 255).to(torch.uint8).contiguous()
        enc = encode_jpeg(u8, quality=jpeg_quality)
        body = bytes(cuda_to_bytes(enc.contiguous(), "image_bitstream_download"))
        # SOI, then our APP1 (EXIF) / APP2 (ICC) before the encoder's own segments
        head = b"\xff\xd8"
        if info.exif:
            head += _app(0xE1, _exif_upright(info.exif))
        if info.icc:
            chunks = [info.icc[i:i + 65519] for i in range(0, len(info.icc), 65519)]
            for k, c in enumerate(chunks):
                head += _app(0xE2, b"ICC_PROFILE\0" + bytes([k + 1, len(chunks)]) + c)
        rest = body[2:]
        if rest[:2] == b"\xff\xe0":  # drop the encoder's JFIF APP0 when an EXIF APP1 leads
            L = struct.unpack(">H", rest[2:4])[0]
            if info.exif:
                rest = rest[2 + L:]
        dst.write_bytes(head + rest)
    else:
        depth = 16 if info.bit_depth > 8 else 8
        maxv = (1 << depth) - 1
        planes = final if info.bit_depth == depth else (final.float() * maxv / ((1 << info.bit_depth) - 1)).round().to(torch.int32)
        ch = 3
        if alpha is not None:
            a = (alpha.float().clamp(0, 1) * maxv).round().to(torch.int32) if alpha.is_floating_point() else alpha.to(torch.int32)
            planes = torch.cat([planes, a[None]], 0)
            ch = 4
        H, W = planes.shape[1:]
        px = planes.permute(1, 2, 0)                                    # (H, W, C)
        if depth == 16:
            px = torch.stack([px // 256, px % 256], -1).reshape(H, W * ch * 2)
        else:
            px = px.reshape(H, W * ch)
        rows = torch.cat([torch.zeros(H, 1, dtype=px.dtype, device=px.device), px], 1).to(torch.uint8).contiguous()
        raw = bytes(cuda_to_bytes(rows, "image_bitstream_download"))
        ctype = 6 if ch == 4 else 2

        def chunk(t: bytes, d: bytes) -> bytes:
            return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)

        png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", W, H, depth, ctype, 0, 0, 0))
        if info.icc:
            png += chunk(b"iCCP", b"ICC profile\0\0" + zlib.compress(info.icc))
        elif not info.colour_assumed and info.profile is not None and info.profile.description == "sRGB":
            png += chunk(b"sRGB", b"\0")
        png += chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b"")
        dst.write_bytes(png)
    return {"path": str(dst), "format": fmt, "bytes": dst.stat().st_size,
            "note": "outside the edit support the original code values are written back" +
                    ("; JPEG re-encoding is lossy for the whole image" if fmt == "jpeg" else " (lossless: bit-identical there)")}

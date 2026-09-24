"""NVDEC/NVENC video I/O through PyNvVideoCodec 2.2.2 (device memory only).

Decoding: FFmpeg demux (CPU container/byte work) -> NVDEC -> decoded surface copied into
an owned CUDA tensor before the decoder may reuse it -> CUDA YCbCr->R'G'B' conversion with
explicit range/matrix and chroma siting.  Encoding (viewing previews only): CUDA R'G'B' ->
YCbCr 4:2:0 (NV12 / P010) -> NVENC -> FFmpegMuxer with explicit source timestamps.

Support is probed at runtime (GetDecoderCaps); unsupported codecs/profiles (e.g. ProRes, DNxHR,
RAW) raise InputError naming the detected format and the strict alternatives.  There is no
software decode/encode fallback.
"""

from __future__ import annotations

import gc
import math
import os
from dataclasses import asdict, dataclass, field
from fractions import Fraction

import torch
import torch.nn.functional as F

from ..color import transforms as CT
from ..gpu.audit import boundary
from ..gpu.runtime import ensure_cuda12_runtime_on_dll_path
from .errors import InputError

_NVC = None


def nvc():
    global _NVC
    if _NVC is None:
        ensure_cuda12_runtime_on_dll_path()
        import PyNvVideoCodec as m  # noqa: N813

        _NVC = m
    return _NVC


STRICT_ALTERNATIVES = ("a hardware-decodable H.264/HEVC/AV1 export (4:2:0 or 4:2:2, 8/10-bit), or an uncompressed "
                       "single-part scanline FLOAT/HALF EXR sequence with --fps or --timestamps")


@dataclass
class VideoInfo:
    path: str
    container: str
    codec: str
    width: int
    height: int
    chroma: str
    bit_depth_container: int
    fps: float | None
    fps_rational: tuple[int, int] | None
    avg_fps_rational: tuple[int, int] | None
    time_base: tuple[int, int]
    color_space_meta: str
    color_range_meta: str
    has_audio: bool
    nvdec_caps: dict = field(default_factory=dict)
    bit_depth: int | None = None  # from the decoded surface
    surface_format: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def probe(path: str) -> VideoInfo:
    m = nvc()
    try:
        d = m.CreateDemuxer(path)
    except Exception as e:  # container not understood
        raise InputError(f"{path}: cannot demux ({e}). Strict alternatives: {STRICT_ALTERNATIVES}") from e
    codec = str(d.GetNvCodecId()).split(".")[-1]
    chroma = str(d.ChromaFormat()).split(".")[-1]
    try:
        aid = d.GetAudioStreamId()
    except Exception:
        aid = -1
    fr = (d.RFrameRateNum(), d.RFrameRateDen()) if d.RFrameRateDen() else None
    afr = (d.AvgFrameRateNum(), d.AvgFrameRateDen()) if d.AvgFrameRateDen() else None
    info = VideoInfo(path=path, container=d.GetContainerName(), codec=codec, width=d.Width(), height=d.Height(), chroma=chroma,
                     bit_depth_container=d.BitDepth(), fps=float(d.FrameRate()) if d.FrameRate() else None,
                     fps_rational=fr, avg_fps_rational=afr, time_base=(d.GetTimebaseNum(), d.GetTimebaseDen()),
                     color_space_meta=str(d.ColorSpace()).split(".")[-1], color_range_meta=str(d.ColorRange()).split(".")[-1],
                     has_audio=aid is not None and aid >= 0)
    del d
    return info


def check_supported(info: VideoInfo) -> dict:
    m = nvc()
    if info.codec not in ("H264", "HEVC", "AV1", "VP9"):
        raise InputError(f"{info.path}: codec {info.codec} (container {info.container}) is not an NVDEC-supported route in "
                         f"strict mode. Strict alternatives: {STRICT_ALTERNATIVES}")
    cf = {"420": "420", "422": "422", "444": "444", "Monochrome": "Monochrome"}.get(info.chroma.replace("cudaVideoChromaFormat_", ""), info.chroma)
    caps = {}
    for bd in (8, 10, 12):
        try:
            caps[bd] = m.GetDecoderCaps(0, getattr(m.cudaVideoCodec, info.codec), m.cudaVideoChromaFormat.__members__[cf], bd)
        except Exception as e:
            caps[bd] = {"supported": 0, "error": str(e)}
    info.nvdec_caps = {str(k): v for k, v in caps.items()}
    if not any(c.get("supported") for c in caps.values()):
        raise InputError(f"{info.path}: NVDEC reports no support for {info.codec} {cf}; strict alternatives: {STRICT_ALTERNATIVES}")
    mx = max((c for c in caps.values() if c.get("supported")), key=lambda c: c.get("width_max", 0))
    if info.width > mx.get("width_max", 1 << 30) or info.height > mx.get("height_max", 1 << 30):
        raise InputError(f"{info.path}: {info.width}x{info.height} exceeds NVDEC limits {mx.get('width_max')}x{mx.get('height_max')}")
    return info.nvdec_caps


class _Raw:
    def __init__(self, ptr: int, shape: tuple, strides: tuple, typestr: str):
        self.__cuda_array_interface__ = {"shape": shape, "strides": strides, "typestr": typestr, "data": (ptr, False), "version": 3}


@dataclass
class DecodedPlanes:
    index: int
    pts: int
    y: torch.Tensor  # (H,W) int32 codes
    u: torch.Tensor  # chroma planes, native subsampled resolution
    v: torch.Tensor
    chroma: str
    bit_depth: int


class VideoReader:
    """Iterates decoded frames in display order with their stream PTS."""

    def __init__(self, path: str, device: torch.device, max_frames: int | None = None):
        self.info = probe(path)
        check_supported(self.info)
        m = nvc()
        self.dev = device
        self.max_frames = max_frames
        self._dmx = m.CreateDemuxer(path)
        self._dec = m.CreateDecoder(gpuid=device.index or 0, codec=self._dmx.GetNvCodecId(), usedevicememory=True)
        self._bit_depth: int | None = None

    def _planes(self, fr, W: int, H: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        fmt = str(fr.format).split(".")[-1]
        p0, p1 = fr.GetPtrToPlane(0), fr.GetPtrToPlane(1)
        if fmt == "NV12":
            y = torch.as_tensor(_Raw(p0, (H, W), (W, 1), "|u1"), device=self.dev).clone()
            uv = torch.as_tensor(_Raw(p1, (H // 2, W // 2, 2), (W, 2, 1), "|u1"), device=self.dev).clone()
            return y.to(torch.int32), uv[..., 0].to(torch.int32), uv[..., 1].to(torch.int32), "420"
        if fmt == "P016":
            y = torch.as_tensor(_Raw(p0, (H, W), (2 * W, 2), "<u2"), device=self.dev).clone()
            uv = torch.as_tensor(_Raw(p1, (H // 2, W // 2, 2), (2 * W, 4, 2), "<u2"), device=self.dev).clone()
            return y.to(torch.int32), uv[..., 0].to(torch.int32), uv[..., 1].to(torch.int32), "420"
        if fmt == "NV16":
            y = torch.as_tensor(_Raw(p0, (H, W), (W, 1), "|u1"), device=self.dev).clone()
            uv = torch.as_tensor(_Raw(p1, (H, W // 2, 2), (W, 2, 1), "|u1"), device=self.dev).clone()
            return y.to(torch.int32), uv[..., 0].to(torch.int32), uv[..., 1].to(torch.int32), "422"
        if fmt == "P216":
            y = torch.as_tensor(_Raw(p0, (H, W), (2 * W, 2), "<u2"), device=self.dev).clone()
            uv = torch.as_tensor(_Raw(p1, (H, W // 2, 2), (2 * W, 4, 2), "<u2"), device=self.dev).clone()
            return y.to(torch.int32), uv[..., 0].to(torch.int32), uv[..., 1].to(torch.int32), "422"
        if fmt in ("YUV444", "YUV444_16Bit"):
            p2 = fr.GetPtrToPlane(2)
            if fmt == "YUV444":
                mk = lambda p: torch.as_tensor(_Raw(p, (H, W), (W, 1), "|u1"), device=self.dev).clone().to(torch.int32)
            else:
                mk = lambda p: torch.as_tensor(_Raw(p, (H, W), (2 * W, 2), "<u2"), device=self.dev).clone().to(torch.int32)
            return mk(p0), mk(p1), mk(p2), "444"
        raise InputError(f"{self.info.path}: unsupported decoded surface format {fmt}")

    def __iter__(self):
        W, H = self.info.width, self.info.height
        n = 0
        for pkt in self._dmx:
            for fr in self._dec.Decode(pkt):
                y, u, v, ch = self._planes(fr, W, H)
                pts = int(fr.getPTS())
                torch.cuda.synchronize(self.dev)  # decoder surface no longer referenced after this point
                fmt = str(fr.format).split(".")[-1]
                if fmt in ("P016", "P216", "YUV444_16Bit"):
                    if self._bit_depth is None:
                        self._bit_depth = 10 if bool(((y & 63) == 0).all()) else (12 if bool(((y & 15) == 0).all()) else 16)
                    sh = 16 - self._bit_depth
                    y, u, v = y >> sh, u >> sh, v >> sh
                else:
                    self._bit_depth = 8
                self.info.bit_depth = self._bit_depth
                self.info.surface_format = fmt
                yield DecodedPlanes(n, pts, y, u, v, ch, self._bit_depth)
                n += 1
                if self.max_frames is not None and n >= self.max_frames:
                    return
        # flush
        try:
            flush = nvc().PacketData()
            for fr in self._dec.Decode(flush):
                y, u, v, ch = self._planes(fr, W, H)
                pts = int(fr.getPTS())
                torch.cuda.synchronize(self.dev)
                if self._bit_depth and self._bit_depth > 8:
                    sh = 16 - self._bit_depth
                    y, u, v = y >> sh, u >> sh, v >> sh
                yield DecodedPlanes(n, pts, y, u, v, ch, self._bit_depth or 8)
                n += 1
                if self.max_frames is not None and n >= self.max_frames:
                    return
        except Exception:
            pass


def planes_to_rgb(p: DecodedPlanes, full_range: bool, matrix: str) -> torch.Tensor:
    """Decoded YCbCr codes -> encoded R'G'B' (3,H,W) FP32 on CUDA (chroma siting: MPEG-2 default)."""
    H, W = p.y.shape
    if p.chroma == "420":
        c = CT.upsample_chroma_420(torch.stack([p.u, p.v]).float(), H, W)
    elif p.chroma == "422":
        c = F.interpolate(torch.stack([p.u, p.v]).float()[None], size=(H, W), mode="bilinear", align_corners=False)[0]
    else:
        c = torch.stack([p.u, p.v]).float()
    return CT.ycbcr_to_rgb(p.y.float(), c[0], c[1], p.bit_depth, full_range, matrix)


# ----------------------------------------------------------------------------- encoding
class _Plane:
    def __init__(self, t: torch.Tensor, typestr: str | None = None):
        cai = dict(t.__cuda_array_interface__)
        if typestr:
            cai["typestr"] = typestr
        self.__cuda_array_interface__ = cai
        self._t = t


class _EncFrame:
    """PyNvVideoCodec AppFrame protocol: .cuda() -> per-plane CUDA array views of ONE contiguous
    (1.5H, W) NV12/P010 allocation (NVENC requires the UV plane to follow the Y plane)."""

    def __init__(self, buf: torch.Tensor, H: int, W: int):
        ts = "|u2" if buf.dtype == torch.uint16 else None
        self._keep = buf
        self.planes = [_Plane(buf[:H].view(H, W, 1), ts), _Plane(buf[H:].view(H // 2, W // 2, 2), ts)]

    def cuda(self):
        return self.planes


class VideoWriter:
    """NVENC preview writer. Frames: display-referred R'G'B' (3,H,W) in [0,1] on CUDA.
    ``pts`` are presentation timestamps in ``time_base`` units (source timestamps are preserved)."""

    def __init__(self, path: str, width: int, height: int, fps: float, codec: str = "h264", bit_depth: int = 8,
                 time_base: tuple[int, int] = (1, 90000), qp: int = 20):
        m = nvc()
        self.W, self.H = width - width % 2, height - height % 2
        if self.W < 146 or self.H < 50:
            raise InputError(f"preview size {self.W}x{self.H} below NVENC minimum")
        self.codec, self.bits, self.time_base = codec, bit_depth, time_base
        fmt = "P010" if bit_depth == 10 else "NV12"
        self.enc = m.CreateEncoder(self.W, self.H, fmt, False, codec=codec, preset="P4", tuning_info="high_quality",
                                   rc="constqp", constqp=qp, fps=max(1, round(fps)), gop=max(1, round(fps)), bf=0)
        fr = Fraction(fps).limit_denominator(1001)
        self.mux = m.FFmpegMuxer(path, m.MEDIA_FORMAT.MP4, codec, self.W, self.H, fr.numerator, fr.denominator,
                                 time_base[0], time_base[1], extradata=self.enc.GetSequenceParams())
        self.path = path
        self.n = 0
        self.last_pts = None
        self.dur = max(1, int(round(time_base[1] / (time_base[0] * fps))))
        self.pts_written: list[int] = []
        self.bytes = 0

    def _pack(self, rgb: torch.Tensor):
        x = rgb[:, : self.H, : self.W].clamp(0, 1)
        ycc = CT.rgb_to_ycbcr(x, self.bits, full_range=False, matrix="bt709")
        c = CT.downsample_chroma_420(ycc[1:3])
        mx = (1 << self.bits) - 1
        y = ycc[0].round().clamp(0, mx)
        c = c.round().clamp(0, mx)
        uv = torch.stack([c[0], c[1]], -1).reshape(self.H // 2, self.W)
        buf = torch.cat([y, uv], 0)
        if self.bits == 8:
            return buf.to(torch.uint8).contiguous()
        return (buf.to(torch.int32) << 6).to(torch.uint16).contiguous()

    def write(self, rgb: torch.Tensor, pts: int | None = None) -> None:
        buf = self._pack(rgb)
        torch.cuda.synchronize()
        pts = self.n * self.dur if pts is None else int(pts)
        self.pts_written.append(pts)
        for pk in self.enc.Encode(_EncFrame(buf, self.H, self.W)):
            self._mux(pk)
        self.n += 1

    def _mux(self, pk) -> None:
        idx = getattr(self, "_mux_i", 0)  # no B-frames: packets leave NVENC in input order
        pts = self.pts_written[idx] if idx < len(self.pts_written) else (self.pts_written[-1] + self.dur)
        nxt = self.pts_written[idx + 1] if idx + 1 < len(self.pts_written) else pts + self.dur
        with boundary("codec_bitstream"):
            data = pk["data"]
        self.bytes += len(data)
        self.mux.Mux(data, pts, pts, max(1, nxt - pts), 0, 1 if pk["picture_type"] in (2, 3) else 0, idx)
        self._mux_i = idx + 1

    def close(self) -> dict:
        for pk in self.enc.EndEncode():
            self._mux(pk)
        self.mux.Finalize()
        del self.mux
        del self.enc
        gc.collect()
        return {"path": self.path, "frames": self.n, "codec": self.codec, "bit_depth": self.bits, "bytes": os.path.getsize(self.path),
                "time_base": list(self.time_base), "pts": self.pts_written, "encoder": "NVENC (PyNvVideoCodec)"}


def read_timestamps(path: str, device: torch.device, max_frames: int | None = None) -> tuple[list[int], VideoInfo]:
    """Decode-order independent: returns display-order PTS of all decoded frames (for tests)."""
    r = VideoReader(path, device, max_frames)
    pts = [p.pts for p in r]
    return pts, r.info

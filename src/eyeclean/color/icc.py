"""ICC profiles for still images: parse a matrix/TRC RGB profile and convert with it on the GPU.

Photos carry their colour as an ICC profile (sRGB, Display P3, Adobe RGB, ...), nearly always the matrix/TRC kind:
three colorant XYZ values in the D50 profile connection space and one tone curve per channel.  Parsing is byte
handling on the host (a few kilobytes of tags); every pixel operation - the tone curve, the matrix, the chromatic
adaptation - runs on CUDA.  LUT-based profiles (A2B0 only, no colorants) are reported as unsupported and the caller
falls back to sRGB, recording it.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass

import torch

# linear Rec.709 / sRGB primaries (D65) <- XYZ (D65)
_XYZ_TO_709 = ((3.2404542, -1.5371385, -0.4985314), (-0.9692660, 1.8760108, 0.0415560), (0.0556434, -0.2040259, 1.0572252))
# Bradford chromatic adaptation D50 -> D65 (the ICC PCS is D50)
_D50_TO_D65 = ((0.9555766, -0.0230393, 0.0631636), (-0.0282895, 1.0099416, 0.0210077), (0.0122982, -0.0204830, 1.3299098))


class IccError(ValueError):
    pass


@dataclass
class Trc:
    kind: str            # "identity" | "gamma" | "lut" | "para"
    gamma: float = 1.0
    lut: tuple = ()      # 0..1 samples for "lut"
    ptype: int = 0       # parametric function type 0..4
    params: tuple = ()   # g, a, b, c, d, e, f (as present)


@dataclass
class IccProfile:
    description: str
    colorants: tuple     # 3x3 rows = X, Y, Z of the R, G, B colorants (D50 PCS), i.e. RGB -> XYZ(D50)
    trcs: tuple          # (Trc R, Trc G, Trc B)
    version: str

    def to_linear709_matrix(self) -> tuple:
        """3x3: linear profile RGB -> linear Rec.709 (D65)."""
        return _matmul(_XYZ_TO_709, _matmul(_D50_TO_D65, self.colorants))

    @property
    def is_srgb_like(self) -> bool:
        m = self.to_linear709_matrix()
        ident = all(abs(m[i][j] - (1.0 if i == j else 0.0)) < 0.02 for i in range(3) for j in range(3))
        return ident and self.trcs[0].kind == "para" and self.trcs[0].ptype == 3 and abs(self.trcs[0].params[0] - 2.4) < 0.05


def _matmul(a, b):
    return tuple(tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)) for i in range(3))


def _s15f16(b: bytes, o: int) -> float:
    return struct.unpack(">i", b[o:o + 4])[0] / 65536.0


def _text(data: bytes) -> str:
    sig = data[:4]
    if sig == b"desc":
        n = struct.unpack(">I", data[8:12])[0]
        return data[12:12 + n].split(b"\0")[0].decode("latin-1", "replace")
    if sig == b"mluc":
        count, size = struct.unpack(">II", data[8:16])
        if count:
            length, offset = struct.unpack(">II", data[20:28])
            return data[offset:offset + length].decode("utf-16-be", "replace")
    if sig == b"text":
        return data[8:].split(b"\0")[0].decode("latin-1", "replace")
    return ""


def _trc(data: bytes) -> Trc:
    sig = data[:4]
    if sig == b"curv":
        n = struct.unpack(">I", data[8:12])[0]
        if n == 0:
            return Trc("identity")
        if n == 1:
            return Trc("gamma", gamma=struct.unpack(">H", data[12:14])[0] / 256.0)
        vals = struct.unpack(f">{n}H", data[12:12 + 2 * n])
        return Trc("lut", lut=tuple(v / 65535.0 for v in vals))
    if sig == b"para":
        ptype = struct.unpack(">H", data[8:10])[0]
        count = {0: 1, 1: 3, 2: 4, 3: 5, 4: 7}.get(ptype)
        if count is None:
            raise IccError(f"unsupported parametric curve type {ptype}")
        params = tuple(_s15f16(data, 12 + 4 * i) for i in range(count))
        return Trc("para", ptype=ptype, params=params)
    raise IccError(f"unsupported tone curve type {sig!r}")


def parse_icc(b: bytes) -> IccProfile:
    if len(b) < 132 or b[36:40] != b"acsp":
        raise IccError("not an ICC profile")
    if b[16:20] != b"RGB ":
        raise IccError(f"ICC colour space {b[16:20]!r} is not RGB")
    if b[20:24] != b"XYZ ":
        raise IccError(f"ICC PCS {b[20:24]!r} is not XYZ")
    version = f"{b[8]}.{b[9] >> 4}"
    n = struct.unpack(">I", b[128:132])[0]
    tags = {}
    for i in range(n):
        sig, off, size = struct.unpack(">4sII", b[132 + 12 * i:144 + 12 * i])
        tags[sig] = b[off:off + size]
    need = (b"rXYZ", b"gXYZ", b"bXYZ", b"rTRC", b"gTRC", b"bTRC")
    if not all(t in tags for t in need):
        raise IccError("ICC profile is not matrix/TRC (LUT-based profiles are not supported)")
    cols = [tuple(_s15f16(tags[t], 8 + 4 * k) for k in range(3)) for t in (b"rXYZ", b"gXYZ", b"bXYZ")]
    colorants = tuple(tuple(cols[j][i] for j in range(3)) for i in range(3))  # rows X, Y, Z; columns R, G, B
    trcs = tuple(_trc(tags[t]) for t in (b"rTRC", b"gTRC", b"bTRC"))
    desc = _text(tags.get(b"desc", b"")) or "unnamed ICC profile"
    return IccProfile(desc.strip(), colorants, trcs, version)


# ----------------------------------------------------------------------------- GPU application
def _para(x: torch.Tensor, t: Trc) -> torch.Tensor:
    p = t.params
    g = p[0]
    if t.ptype == 0:
        return x.clamp(min=0) ** g
    a, b_ = p[1], p[2]
    lin = (a * x + b_).clamp(min=0) ** g
    if t.ptype == 1:
        return torch.where(x >= -b_ / a, lin, torch.zeros_like(x))
    if t.ptype == 2:
        return torch.where(x >= -b_ / a, lin + p[3], torch.full_like(x, p[3]))
    if t.ptype == 3:
        return torch.where(x >= p[4], lin, p[3] * x)
    return torch.where(x >= p[4], lin + p[5], p[3] * x + p[6])


def trc_decode(v: torch.Tensor, t: Trc) -> torch.Tensor:
    """Encoded [0, 1] -> linear, one channel."""
    if t.kind == "identity":
        return v
    if t.kind == "gamma":
        return v.clamp(min=0) ** t.gamma
    if t.kind == "para":
        return _para(v, t)
    lut = torch.tensor(t.lut, device=v.device, dtype=torch.float32)
    x = v.clamp(0, 1) * (lut.numel() - 1)
    i0 = x.floor().long().clamp(max=lut.numel() - 2)
    w = x - i0
    return lut[i0] * (1 - w) + lut[i0 + 1] * w


def trc_encode(lin: torch.Tensor, t: Trc, n: int = 16384) -> torch.Tensor:
    """Linear -> encoded [0, 1]: inverts the curve by a dense monotone table (GPU searchsorted)."""
    if t.kind == "identity":
        return lin.clamp(0, 1)
    if t.kind == "gamma":
        return lin.clamp(0, 1) ** (1.0 / t.gamma)
    xs = torch.linspace(0, 1, n, device=lin.device, dtype=torch.float32)
    ys = torch.cummax(trc_decode(xs, t), 0).values  # monotone
    q = lin.clamp(float(ys[0]), float(ys[-1])).contiguous()
    idx = torch.searchsorted(ys, q.flatten()).clamp(1, n - 1).view_as(q)
    y0, y1 = ys[idx - 1], ys[idx]
    w = ((q - y0) / (y1 - y0).clamp(min=1e-12)).clamp(0, 1)
    return (xs[idx - 1] * (1 - w) + xs[idx] * w)


def _apply3(m, x: torch.Tensor) -> torch.Tensor:
    M = torch.tensor(m, device=x.device, dtype=torch.float32)
    return torch.einsum("ij,jhw->ihw", M, x)


def icc_to_linear709(v: torch.Tensor, prof: IccProfile) -> torch.Tensor:
    """(3,H,W) encoded [0, 1] in the profile's space -> linear Rec.709 (D65), CUDA."""
    lin = torch.stack([trc_decode(v[c], prof.trcs[c]) for c in range(3)])
    return _apply3(prof.to_linear709_matrix(), lin)


def linear709_to_icc(l709: torch.Tensor, prof: IccProfile) -> torch.Tensor:
    """Inverse of icc_to_linear709: linear Rec.709 -> encoded [0, 1] in the profile's space."""
    m = prof.to_linear709_matrix()
    inv = _inverse3(m)
    lin = _apply3(inv, l709)
    return torch.stack([trc_encode(lin[c], prof.trcs[c]) for c in range(3)])


def _inverse3(m):
    a, b, c = m[0]
    d, e, f = m[1]
    g, h, i = m[2]
    det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    if abs(det) < 1e-12:
        raise IccError("singular ICC colorant matrix")
    return ((( e * i - f * h) / det, -(b * i - c * h) / det, (b * f - c * e) / det),
            (-(d * i - f * g) / det, (a * i - c * g) / det, -(a * f - c * d) / det),
            (( d * h - e * g) / det, -(a * h - b * g) / det, (a * e - b * d) / det))


def srgb_profile() -> IccProfile:
    """The IEC 61966-2.1 sRGB curve and primaries as a profile (the convention for untagged images)."""
    trc = Trc("para", ptype=3, params=(2.4, 1 / 1.055, 0.055 / 1.055, 1 / 12.92, 0.04045))
    inv709 = _inverse3(_XYZ_TO_709)
    inv_ad = _inverse3(_D50_TO_D65)
    colorants = _matmul(inv_ad, inv709)  # RGB(709) -> XYZ D65 -> XYZ D50
    return IccProfile("sRGB (untagged: assumed)", colorants, (trc, trc, trc), "builtin")


def primaries_profile(description: str, xy: tuple, white_xy: tuple, trc: Trc) -> IccProfile:
    """A matrix/TRC profile from chromaticities (for nclx / cICP / VUI-tagged images)."""
    def xyz(x, y):
        return (x / y, 1.0, (1 - x - y) / y)
    W = xyz(*white_xy)
    P = [xyz(*p) for p in xy]
    Mp = tuple(tuple(P[j][i] for j in range(3)) for i in range(3))
    S = _apply_vec(_inverse3(Mp), W)
    M = tuple(tuple(Mp[i][j] * S[j] for j in range(3)) for i in range(3))  # RGB -> XYZ(white)
    # adapt to D50 if the white is D65 (the only white these codes use besides DCI)
    if abs(white_xy[0] - 0.3127) < 1e-3 and abs(white_xy[1] - 0.3290) < 1e-3:
        M = _matmul(_inverse3(_D50_TO_D65), M)
    return IccProfile(description, M, (trc, trc, trc), "builtin")


def _apply_vec(m, v):
    return tuple(sum(m[i][k] * v[k] for k in range(3)) for i in range(3))


def gamma_trc(g: float) -> Trc:
    return Trc("gamma", gamma=g) if not math.isclose(g, 1.0) else Trc("identity")

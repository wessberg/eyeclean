"""Still images: ICC colour, EXIF orientation, lossless PNG preservation, JPEG metadata (Round 55 phase A1)."""
import struct
import zlib

import pytest
import torch

from eyeclean.color import icc as ICC
from eyeclean.io import stills as ST


def _icc_bytes(trc: bytes, colorants) -> bytes:
    """A minimal RGB matrix/TRC ICC v2 profile with the given TRC tag body and D50 colorants."""
    def xyz(v):
        return b"XYZ \0\0\0\0" + b"".join(struct.pack(">i", round(c * 65536)) for c in v)
    desc = b"desc\0\0\0\0" + struct.pack(">I", 5) + b"test\0" + b"\0" * 67
    tags = [(b"desc", desc), (b"rXYZ", xyz(colorants[0])), (b"gXYZ", xyz(colorants[1])), (b"bXYZ", xyz(colorants[2])),
            (b"rTRC", trc), (b"gTRC", trc), (b"bTRC", trc), (b"wtpt", xyz((0.9642, 1.0, 0.8249)))]
    off = 128 + 4 + 12 * len(tags)
    table, data = b"", b""
    for sig, body in tags:
        body = body + b"\0" * (-len(body) % 4)
        table += sig + struct.pack(">II", off + len(data), len(body))
        data += body
    size = off + len(data)
    head = struct.pack(">I", size) + b"test" + bytes([2, 0x40, 0, 0]) + b"mntr" + b"RGB " + b"XYZ " + b"\0" * 12 + b"acsp"
    head = head + b"\0" * (128 - len(head))
    return head + struct.pack(">I", len(tags)) + table + data


# sRGB colorants adapted to D50 (the published sRGB ICC values)
SRGB_D50 = ((0.4361, 0.2225, 0.0139), (0.3851, 0.7169, 0.0971), (0.1431, 0.0606, 0.7141))
SRGB_PARA = b"para\0\0\0\0" + struct.pack(">HH", 3, 0) + b"".join(struct.pack(">i", round(v * 65536)) for v in
                                                                  (2.4, 1 / 1.055, 0.055 / 1.055, 1 / 12.92, 0.04045))


def test_icc_parse_srgb_profile_matches_the_builtin_srgb(dev):
    prof = ICC.parse_icc(_icc_bytes(SRGB_PARA, SRGB_D50))
    assert prof.is_srgb_like and prof.description == "test"
    v = torch.linspace(0, 1, 256, device=dev).view(1, 1, 256).expand(3, 1, 256).contiguous()
    a = ICC.icc_to_linear709(v, prof)
    b = ICC.icc_to_linear709(v, ICC.srgb_profile())
    assert float((a - b).abs().max()) < 2e-3
    assert abs(float(a[1, 0, 128]) - ((128 / 255 + 0.055) / 1.055) ** 2.4) < 2e-3


def test_icc_round_trip_gamma_and_lut_curves(dev):
    for trc in (b"curv\0\0\0\0" + struct.pack(">I", 1) + struct.pack(">H", int(2.2 * 256)),
                b"curv\0\0\0\0" + struct.pack(">I", 1024) + b"".join(struct.pack(">H", round(65535 * (i / 1023) ** 1.8)) for i in range(1024))):
        prof = ICC.parse_icc(_icc_bytes(trc, SRGB_D50))
        v = torch.rand(3, 16, 16, device=dev)
        back = ICC.linear709_to_icc(ICC.icc_to_linear709(v, prof), prof)
        assert float((back - v).abs().max()) < 2e-3


def test_display_p3_red_is_outside_rec709(dev):
    p3 = ICC.primaries_profile("P3", ((0.680, 0.320), (0.265, 0.690), (0.150, 0.060)), (0.3127, 0.3290), ST._SRGB_TRC)
    red = torch.tensor([1.0, 0.0, 0.0], device=dev).view(3, 1, 1)
    l709 = ICC.icc_to_linear709(red, p3)
    assert float(l709[0]) > 1.0 and float(l709[1]) < 0.0      # a P3 primary is out of the Rec.709 gamut
    white = ICC.icc_to_linear709(torch.ones(3, 1, 1, device=dev), p3)
    assert float((white - 1).abs().max()) < 1e-3                  # same D65 white


def test_orientation_matches_exif_semantics(dev):
    x = torch.arange(6, device=dev).view(1, 2, 3)
    assert torch.equal(ST.orient(x, 6), torch.rot90(x, -1, (1, 2)))  # 6: rotate 90 clockwise
    assert torch.equal(ST.orient(x, 8), torch.rot90(x, 1, (1, 2)))   # 8: rotate 90 counter-clockwise
    assert torch.equal(ST.orient(x, 3), torch.rot90(x, 2, (1, 2)))
    for o in range(1, 9):
        assert ST.orient(x, o).numel() == 6


def _exif(orientation: int) -> bytes:
    ifd = struct.pack("<H", 1) + struct.pack("<HHIHH", 0x0112, 3, 1, orientation, 0) + struct.pack("<I", 0)
    return b"Exif\0\0" + b"II*\0" + struct.pack("<I", 8) + ifd


def test_exif_orientation_read_and_reset():
    e = _exif(6)
    assert ST._exif_orientation(e)[0] == 6
    assert ST._exif_orientation(ST._exif_upright(e))[0] == 1


def _png(codes: torch.Tensor) -> bytes:
    H, W = codes.shape[1:]
    rows = b"".join(b"\0" + bytes(codes.permute(1, 2, 0)[y].reshape(-1).tolist()) for y in range(H))

    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


def test_png_round_trip_is_lossless_outside_the_support(dev, tmp_path):
    g = torch.Generator().manual_seed(0)
    codes = torch.randint(0, 256, (3, 24, 32), generator=g)
    src = tmp_path / "in.png"
    src.write_bytes(_png(codes))
    lin, alpha, dec, info = ST.read_still(str(src), dev)
    assert info.format == "png" and info.decode == "pyav-cpu" and info.colour_assumed and alpha is None
    assert torch.equal(dec.cpu(), codes.to(torch.int32))
    edited = lin.clone()
    edited[:, 5:10, 5:10] = 0.5
    support = torch.zeros(24, 32, dtype=torch.bool, device=dev)
    support[5:10, 5:10] = True
    out = ST.write_still(tmp_path / "out.png", edited, support, dec, info)
    assert out["format"] == "png"
    _, _, dec2, _ = ST.read_still(str(tmp_path / "out.png"), dev)
    assert torch.equal(dec2[:, ~support], dec[:, ~support])       # bit-identical outside the support
    assert not torch.equal(dec2[:, support], dec[:, support])


def test_jpeg_round_trip_keeps_exif_icc_and_writes_upright(dev, tmp_path):
    from torchvision.io import encode_jpeg

    img = torch.zeros(3, 16, 32, dtype=torch.uint8, device=dev)
    img[0, :, :8] = 255                                            # a red band on the stored image's left
    body = bytes(encode_jpeg(img, quality=95).cpu().numpy())
    icc = _icc_bytes(SRGB_PARA, SRGB_D50)
    app1 = b"\xff\xe1" + struct.pack(">H", len(_exif(6)) + 2) + _exif(6)
    app2 = b"\xff\xe2" + struct.pack(">H", len(icc) + 16) + b"ICC_PROFILE\0\x01\x01" + icc
    src = tmp_path / "in.jpg"
    src.write_bytes(body[:2] + app1 + app2 + body[2:])
    lin, _, codes, info = ST.read_still(str(src), dev)
    assert info.decode == "nvjpeg" and info.orientation == 6 and info.colour == "ICC: test"
    assert (info.width, info.height) == (16, 32)                   # rotated upright
    assert float(codes[0, :8, :].float().mean()) > 200             # the band is now along the top
    ST.write_still(tmp_path / "out.jpg", lin, torch.zeros(32, 16, dtype=torch.bool, device=dev), codes, info)
    _, _, codes2, info2 = ST.read_still(str(tmp_path / "out.jpg"), dev)
    assert info2.orientation == 1 and info2.colour == "ICC: test" and (info2.width, info2.height) == (16, 32)
    assert float((codes2.float() - codes.float()).abs().mean()) < 3.0


def test_unknown_bytes_are_refused(dev, tmp_path):
    p = tmp_path / "x.jpg"
    p.write_bytes(b"not an image at all")
    with pytest.raises(Exception):
        ST.read_still(str(p), dev)

"""EXR adapter vs the official OpenEXR library (offline test; OpenEXR is a dev dependency only)."""

import struct

import numpy as np
import OpenEXR
import pytest
import torch

from eyeclean.io.exr import PIXEL_HALF, ExrError, read_exr, read_render_exr, write_exr


def _special(t):
    t[0, 0] = -0.0
    t[0, 1] = float("inf")
    t[0, 2] = float("-inf")
    t[0, 3] = float("nan")
    t[0, 4] = 1e-40
    return t


def test_roundtrip_bit_exact_and_official_reader(dev, tmp):
    H, W = 37, 53
    g = torch.Generator(device=dev).manual_seed(1)
    ch = {n: _special(torch.randn((H, W), device=dev, generator=g) * 1e3) for n in ("R", "G", "B", "A", "mask.hard")}
    p = tmp / "a.exr"
    write_exr(p, ch, display_size=(3840, 2160), origin=(1234, 567), attributes={"eyeclean/frame": "7"})
    img = read_exr(p, dev)
    assert img.header.data_window == (1234, 567, 1234 + W - 1, 567 + H - 1)
    assert img.header.display_window == (0, 0, 3839, 2159)
    for n, t in ch.items():
        assert torch.equal(img.channels[n].view(torch.int32), t.view(torch.int32)), n
    with OpenEXR.File(str(p), separate_channels=True) as f:
        hd = f.header()
        assert str(hd["compression"]).endswith("NO_COMPRESSION")
        for n, t in ch.items():
            ref = np.ascontiguousarray(f.channels()[n].pixels).view(np.int32)
            assert np.array_equal(ref, t.cpu().numpy().view(np.int32)), n


def test_read_official_writer_output(dev, tmp):
    H, W = 20, 31
    rng = np.random.default_rng(3)
    r = (rng.standard_normal((H, W)) * 50).astype(np.float32)
    r[0, 0] = -0.0
    h = rng.standard_normal((H, W)).astype(np.float16)
    hdr = {"compression": OpenEXR.NO_COMPRESSION, "type": OpenEXR.scanlineimage}
    p = tmp / "b.exr"
    with OpenEXR.File(hdr, {"R": r, "Y": h}) as f:
        f.write(str(p))
    img = read_exr(p, dev)
    assert np.array_equal(img.channels["R"].cpu().numpy().view(np.int32), r.view(np.int32))
    assert img.channels["Y"].dtype == torch.float16
    assert np.array_equal(img.channels["Y"].cpu().numpy().view(np.int16), h.view(np.int16))


def test_half_writer_and_negative_hdr(dev, tmp):
    x = torch.tensor([[-2.5, 0.0, 65504.0, 1.0e-3]], device=dev)
    p = tmp / "h.exr"
    write_exr(p, {"Y": x}, pixel_type=PIXEL_HALF)
    img = read_exr(p, dev)
    assert img.channels["Y"].dtype == torch.float16
    assert torch.equal(img.channels["Y"], x.half())


def test_row_order_and_little_endian_layout(dev, tmp):
    y = torch.arange(12, dtype=torch.float32, device=dev).view(3, 4)
    p = tmp / "o.exr"
    write_exr(p, {"Y": y}, origin=(5, 9))
    from eyeclean.io.exr import probe_exr

    b = p.read_bytes()
    hdr_end = probe_exr(p)["header_end"]
    offs = struct.unpack_from("<3Q", b, hdr_end)  # offset table directly after the header
    assert offs[1] - offs[0] == 8 + 16
    # first chunk: little-endian int32 y == 9 (dataWindow ymin), size 16, then row 0 floats in order
    assert struct.unpack_from("<ii", b, offs[0]) == (9, 16)
    assert struct.unpack_from("<4f", b, offs[0] + 8) == (0.0, 1.0, 2.0, 3.0)
    assert struct.unpack_from("<i", b, offs[2])[0] == 11


def test_compressed_and_multipart_rejected(dev, tmp):
    # Round 52: ZIP / ZIPS are read (tests/unit/test_exr_zip.py); other compressions stay rejected
    r = np.ones((8, 8), np.float32)
    p = tmp / "piz.exr"
    with OpenEXR.File({"compression": OpenEXR.PIZ_COMPRESSION, "type": OpenEXR.scanlineimage}, {"R": r}) as f:
        f.write(str(p))
    with pytest.raises(ExrError, match="compression"):
        read_exr(p, dev)
    p2 = tmp / "mp.exr"
    parts = [OpenEXR.Part({"compression": OpenEXR.NO_COMPRESSION, "type": OpenEXR.scanlineimage}, {"R": r}, name="a"),
             OpenEXR.Part({"compression": OpenEXR.NO_COMPRESSION, "type": OpenEXR.scanlineimage}, {"R": r * 2}, name="b")]
    with OpenEXR.File(parts) as f:
        f.write(str(p2))
    with pytest.raises(ExrError, match="multipart"):
        read_exr(p2, dev)
    d = read_render_exr(p2, dev)  # declared renderer boundary accepts uncompressed multipart
    assert torch.equal(d["b.R"], torch.full((8, 8), 2.0, device=dev))

"""ZIP / ZIPS EXR reading and writing (Round 52: studio and blender-virtual-production inputs are ZIP).

The fixtures were encoded AND decoded by an independent implementation (OpenCV's OpenEXR,
tests/fixtures/exr_zip/make_fixtures.py); our GPU decode must equal OpenCV's decode exactly.
"""
import json
from pathlib import Path

import pytest
import torch

from eyeclean.io.exr import (NO_COMPRESSION, ZIP_COMPRESSION, ZIPS_COMPRESSION, PIXEL_FLOAT, PIXEL_HALF, ExrError, probe_exr,
                             read_exr, write_exr)
from eyeclean.io.rawbytes import bytes_to_cuda

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "exr_zip"
META = json.loads((FIX / "fixtures.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", sorted(META))
def test_zip_fixture_matches_independent_decoder(dev, name):
    m = META[name]
    img = read_exr(FIX / f"{name}.exr", dev)
    exp = bytes_to_cuda((FIX / f"{name}.expected.bin").read_bytes(), dev, "test_reference").view(torch.float32)
    exp = exp.view(len(m["channels"]), m["height"], m["width"])
    for i, c in enumerate(m["channels"]):
        got = img.channels[c].float()
        assert got.shape == (m["height"], m["width"])
        assert torch.equal(got, exp[i]), f"{name}: channel {c} differs from OpenCV's decode"


@pytest.mark.parametrize("compression", [NO_COMPRESSION, ZIPS_COMPRESSION, ZIP_COMPRESSION])
@pytest.mark.parametrize("pixel_type", [PIXEL_HALF, PIXEL_FLOAT])
@pytest.mark.parametrize("size", [(1, 1), (17, 5), (33, 40)])
def test_write_read_round_trip(dev, tmp, compression, pixel_type, size):
    H, W = size
    g = torch.Generator(device=dev).manual_seed(H * 1000 + W)
    noise = torch.randn((H, W), generator=g, device=dev) * 4
    ramp = torch.linspace(0, 1, H * W, device=dev).view(H, W)
    ch = {"R": noise, "G": ramp, "B": -ramp, "A": (ramp > 0.5).float()}
    p = tmp / f"rt_{compression}_{pixel_type}_{H}x{W}.exr"
    write_exr(p, ch, display_size=(W + 9, H + 7), origin=(4, 3), pixel_type=pixel_type, compression=compression)
    h = probe_exr(p)["parts"][0]
    assert h.compression == compression and h.data_window == (4, 3, 4 + W - 1, 3 + H - 1)
    img = read_exr(p, dev)
    dt = torch.float16 if pixel_type == PIXEL_HALF else torch.float32
    for c, v in ch.items():
        assert torch.equal(img.channels[c], v.to(dt)), f"channel {c}"


def test_zip_shrinks_smooth_images(dev, tmp):
    ramp = torch.linspace(0, 1, 64 * 64, device=dev).view(64, 64)
    a, b = tmp / "raw.exr", tmp / "zip.exr"
    write_exr(a, {"Y": ramp}, compression=NO_COMPRESSION)
    write_exr(b, {"Y": ramp}, compression=ZIP_COMPRESSION)
    assert b.stat().st_size < a.stat().st_size / 2


def test_unsupported_compression_names_a_conversion(dev, tmp):
    p = tmp / "piz.exr"
    write_exr(p, {"R": torch.zeros(2, 2, device=dev)})
    blob = bytearray(p.read_bytes())
    i = blob.index(b"compression\0compression\0") + len(b"compression\0compression\0") + 4
    blob[i] = 4                                                   # PIZ
    p.write_bytes(bytes(blob))
    with pytest.raises(ExrError, match="PIZ.*oiiotool"):
        read_exr(p, dev)


@pytest.mark.parametrize("compression", [ZIPS_COMPRESSION, ZIP_COMPRESSION])
def test_official_openexr_reads_our_zip_and_we_read_its_zip(dev, tmp, compression):
    """Cross-check against the reference OpenEXR library in both directions."""
    OpenEXR = pytest.importorskip("OpenEXR")
    import numpy as np  # the reference library's own array type (test-only, CPU)

    H, W = 29, 13
    g = torch.Generator(device=dev).manual_seed(7)
    ours = {"R": torch.randn((H, W), generator=g, device=dev), "G": torch.linspace(-2, 2, H * W, device=dev).view(H, W)}
    p = tmp / "ours.exr"
    write_exr(p, ours, pixel_type=PIXEL_FLOAT, compression=compression)
    with OpenEXR.File(str(p)) as f:
        ch = f.channels()
        for c, v in ours.items():
            ref = bytes_to_cuda(np.ascontiguousarray(ch[c].pixels, dtype="<f4").tobytes(), dev, "test_reference").view(torch.float32)
            assert torch.equal(ref.view(H, W), v), f"OpenEXR decodes our channel {c} differently"
    ramp = (np.arange(H * W, dtype=np.float32).reshape(H, W) / 7.0).astype(np.float32)
    q = tmp / "theirs.exr"
    comp = OpenEXR.ZIP_COMPRESSION if compression == ZIP_COMPRESSION else OpenEXR.ZIPS_COMPRESSION
    with OpenEXR.File({"compression": comp, "type": OpenEXR.scanlineimage}, {"Y": ramp}) as f:
        f.write(str(q))
    got = read_exr(q, dev).channels["Y"]
    exp = bytes_to_cuda(ramp.tobytes(), dev, "test_reference").view(torch.float32).view(H, W)
    assert torch.equal(got, exp)

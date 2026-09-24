import pytest
import torch

pytestmark = pytest.mark.codec


@pytest.mark.parametrize("codec,bits", [("h264", 8), ("hevc", 10)])
def test_nvenc_nvdec_vfr_roundtrip(dev, tmp, codec, bits):
    from eyeclean.io.video import VideoReader, VideoWriter, planes_to_rgb

    W, H = 640, 360
    pts = [0, 3750, 7500, 13000, 15000, 22500, 26250]
    w = VideoWriter(str(tmp / f"t.{codec}.mp4"), W, H, 24.0, codec, bits)
    frames = []
    for i, t in enumerate(pts):
        ys, xs = torch.meshgrid(torch.linspace(0, 1, H, device=dev), torch.linspace(0, 1, W, device=dev), indexing="ij")
        f = torch.stack([xs, ys, torch.full_like(xs, i / 7)]) * 0.8 + 0.1
        frames.append(f)
        w.write(f, t)
    info = w.close()
    assert info["frames"] == len(pts)
    r = VideoReader(str(tmp / f"t.{codec}.mp4"), dev)
    got = []
    for k, d in enumerate(r):
        got.append(d.pts)
        assert float((planes_to_rgb(d, False, "bt709") - frames[k]).abs().mean()) < 0.01
    assert got == pts
    assert r.info.bit_depth == bits


def test_unsupported_codec_rejected(tmp):
    from eyeclean.io.errors import InputError
    from eyeclean.io.video import check_supported, probe

    av = pytest.importorskip("av")
    import numpy as np

    p = tmp / "p.mov"
    c = av.open(str(p), "w")
    s = c.add_stream("prores_ks", rate=24)
    s.width, s.height, s.pix_fmt = 320, 240, "yuv422p10le"
    for i in range(2):
        for pk in s.encode(av.VideoFrame.from_ndarray(np.full((240, 320, 3), 80, np.uint8), format="rgb24")):
            c.mux(pk)
    for pk in s.encode():
        c.mux(pk)
    c.close()
    with pytest.raises(InputError, match="(?i)strict"):
        check_supported(probe(str(p)))

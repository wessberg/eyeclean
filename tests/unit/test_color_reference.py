"""CUDA colour transforms vs colour-science (independent CPU reference, offline test only)."""

import warnings

import numpy as np
import pytest
import torch

warnings.filterwarnings("ignore")
colour = pytest.importorskip("colour")

from eyeclean.color import transforms as CT  # noqa: E402


def test_transfer_functions(dev):
    x = np.linspace(0.0, 12.0, 4001)
    keep = np.abs(x - 0.018) > 1e-6  # BT.709 branches meet discontinuously at the published breakpoint
    got = CT.bt709_oetf(torch.tensor(x, device=dev, dtype=torch.float32)).cpu().numpy()
    assert np.abs(got - colour.models.oetf_BT709(x))[keep].max() < 1e-5
    v = np.linspace(0.0, 1.0, 4001)
    kv = np.abs(v - 0.081) > 1e-6
    assert np.abs(CT.bt709_inverse_oetf(torch.tensor(v, device=dev, dtype=torch.float32)).cpu().numpy()
                  - colour.models.oetf_inverse_BT709(v))[kv].max() < 1e-5
    assert np.abs(CT.srgb_eotf(torch.tensor(v, device=dev, dtype=torch.float32)).cpu().numpy() - colour.models.eotf_sRGB(v)).max() < 1e-5
    vd = colour.models.log_decoding_VLog(v)
    got = CT.vlog_decode(torch.tensor(v, device=dev, dtype=torch.float32)).cpu().numpy()
    assert (np.abs(got - vd) / np.maximum(np.abs(vd), 1e-3)).max() < 1e-5
    assert np.abs(CT.vlog_encode(torch.tensor(x, device=dev, dtype=torch.float32)).cpu().numpy() - colour.models.log_encoding_VLog(x)).max() < 1e-5


@pytest.mark.parametrize("src,m", [("ITU-R BT.709", CT.M_709_TO_AP1), ("V-Gamut", CT.M_VGAMUT_TO_AP1), ("ITU-R BT.2020", CT.M_2020_TO_AP1)])
def test_primaries_matrices(dev, src, m):
    rgb = torch.rand((3, 16, 16), device=dev) * 3 - 0.5
    ref = colour.RGB_to_RGB(np.moveaxis(rgb.cpu().numpy().astype(np.float64), 0, -1), colour.RGB_COLOURSPACES[src],
                            colour.RGB_COLOURSPACES["ACEScg"], "Bradford")
    got = np.moveaxis(CT.apply_matrix(rgb, m).cpu().numpy(), 0, -1)
    assert np.abs(got - ref).max() < 2e-6


def test_ycbcr_roundtrip_and_ranges(dev):
    rgb = torch.rand((3, 32, 32), device=dev)
    for bits in (8, 10):
        for full in (False, True):
            ycc = CT.rgb_to_ycbcr(rgb, bits, full, "bt709")
            back = CT.ycbcr_to_rgb(ycc[0], ycc[1], ycc[2], bits, full, "bt709")
            assert (back - rgb).abs().max() < 1e-5
    y = torch.tensor([[16.0, 235.0]], device=dev)
    c = torch.tensor([[128.0, 128.0]], device=dev)
    out = CT.ycbcr_to_rgb(y, c, c, 8, False, "bt709")  # (3,1,2): limited-range black and white
    assert torch.allclose(out[:, 0, 0], torch.zeros(3, device=dev), atol=1e-6)
    assert torch.allclose(out[:, 0, 1], torch.ones(3, device=dev), atol=1e-6)


def test_asinh_mapping_invertible(dev):
    x = torch.linspace(-0.5, 50.0, 2000, device=dev).view(1, 1, 1, -1)
    e = torch.tensor([0.37], device=dev)
    assert (CT.asinh_inverse(CT.asinh_forward(x, e), e) - x).abs().max() < 1e-4


def test_chroma_siting(dev):
    # co-sited horizontally: even luma columns reproduce chroma samples exactly
    c = torch.rand((1, 8, 16), device=dev)
    up = CT.upsample_chroma_420(c, 16, 32)
    # vertically centred: average of the two neighbouring chroma rows at luma rows 2k+1/2k+2 boundaries
    assert torch.allclose(up[0, 1::2, ::2][:-1], (0.75 * c[0, :-1] + 0.25 * c[0, 1:]), atol=1e-5)

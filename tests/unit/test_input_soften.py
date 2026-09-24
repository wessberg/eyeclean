from types import SimpleNamespace

import pytest
import torch

from eyeclean.gpu.imgops import CropTransform, extract_crop
from eyeclean.inference.pipeline import Pipeline

pytestmark = pytest.mark.gpu


def _crop(dev, frame, tf, sigma):
    return Pipeline._input_crop(SimpleNamespace(gate={"input_soften_sigma": sigma}), frame, tf)


def test_off_is_the_plain_crop_bit_for_bit(dev):
    g = torch.Generator(device=dev).manual_seed(0)
    frame = torch.rand(3, 120, 160, device=dev, generator=g)
    tf = CropTransform(80.3, 61.7, 4.6, 0.1, 64)
    assert torch.equal(_crop(dev, frame, tf, 0.0), extract_crop(frame, tf))


def test_constant_frame_stays_constant_and_zero_outside_the_frame(dev):
    frame = torch.full((3, 120, 160), 0.37, device=dev)
    inside = CropTransform(80.0, 60.0, 4.0, 0.0, 64)
    out = _crop(dev, frame, inside, 0.5)
    assert float((out - 0.37).abs().max()) < 1e-5
    edge = CropTransform(2.0, 60.0, 4.0, 0.0, 64)          # the left part of this crop lies outside the frame
    plain, soft = extract_crop(frame, edge), _crop(dev, frame, edge, 0.5)
    assert bool((soft[:, :, :8] == 0).all()) and bool((plain[:, :, :8] == 0).all())


def test_softening_damps_high_frequencies(dev):
    yy, xx = torch.meshgrid(torch.arange(120, device=dev), torch.arange(160, device=dev), indexing="ij")
    frame = ((xx + yy) % 2).float().expand(3, 120, 160).contiguous()   # a native-pixel checkerboard
    tf = CropTransform(80.0, 60.0, 4.0, 0.0, 64)
    assert float(_crop(dev, frame, tf, 0.5).std()) < 0.5 * float(extract_crop(frame, tf).std())

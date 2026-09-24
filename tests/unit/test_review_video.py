from pathlib import Path

import pytest
import torch

from eyeclean.inference.process import PRIVATE_DIRS, ROOT, ProcessArgs, _check_paths
from eyeclean.io.errors import InputError
from eyeclean.validation.review_video import _GLYPHS, _smooth, draw_text


def test_font_glyphs_are_5x7():
    assert all(len(v) == 35 for v in _GLYPHS.values())


def test_smooth_fills_gaps_and_averages():
    v = _smooth([None, 2.0, None, 4.0, None], 1)
    assert v[0] == pytest.approx(2.0) and v[-1] == pytest.approx(4.0)
    assert all(x is not None for x in v)


@pytest.mark.gpu
def test_draw_text_marks_pixels(dev):
    img = torch.zeros((3, 20, 80), device=dev)
    draw_text(img, "EYE 1", 1, 1, 2)
    assert float(img.sum()) > 0


@pytest.mark.parametrize("private", PRIVATE_DIRS)
def test_outputs_never_written_into_private_footage_folders(private):
    a = ProcessArgs(input="x.mp4", checkpoint="c", input_color=None, mode="conservative", allow_synthesis=False,
                    output=str(ROOT / private / "out"))
    with pytest.raises(InputError):
        _check_paths(a, [str(ROOT / "x.mp4")])


@pytest.mark.gpu
def test_review_refuses_private_output(dev, tmp):
    from eyeclean.validation.review_video import review

    with pytest.raises(ValueError):
        review("x.mp4", {}, ROOT / "own-footage" / "review_out")
    assert not (ROOT / "own-footage" / "review_out").exists()

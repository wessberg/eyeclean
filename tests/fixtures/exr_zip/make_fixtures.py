"""Generate the ZIP / ZIPS EXR fixtures with an INDEPENDENT encoder/decoder (OpenCV's OpenEXR).

Run once, with any Python that has opencv-python (built with OpenEXR) and numpy:
    OPENCV_IO_ENABLE_OPENEXR=1 python tests/fixtures/exr_zip/make_fixtures.py
Writes <name>.exr and <name>.expected.bin (OpenCV's own decode, float32 planes in R, G, B[, A] order)
plus fixtures.json.  The files are committed; the tests never need OpenCV.
"""
import json
import os
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import cv2  # noqa: E402
import numpy as np  # noqa: E402

OUT = Path(__file__).resolve().parent
rng = np.random.default_rng(5252)
H, W = 23, 37                                   # 23 lines: ZIP = one 16-line chunk + one 7-line chunk
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
smooth = np.stack([xx / W, yy / H, 0.25 + 0.0 * xx, 0.5 + 0.25 * np.sin(xx / 5.0)], -1)       # compresses
noise = rng.standard_normal((H, W, 4)).astype(np.float32) * 3.0                              # stored raw
CASES = [
    ("zip_half_rgba_smooth", smooth, cv2.IMWRITE_EXR_TYPE_HALF, cv2.IMWRITE_EXR_COMPRESSION_ZIP, 4),
    ("zips_half_rgba_smooth", smooth, cv2.IMWRITE_EXR_TYPE_HALF, cv2.IMWRITE_EXR_COMPRESSION_ZIPS, 4),
    ("zip_float_rgb_noise", noise, cv2.IMWRITE_EXR_TYPE_FLOAT, cv2.IMWRITE_EXR_COMPRESSION_ZIP, 3),
    ("zips_float_rgb_noise", noise, cv2.IMWRITE_EXR_TYPE_FLOAT, cv2.IMWRITE_EXR_COMPRESSION_ZIPS, 3),
    ("zip_float_rgb_smooth", smooth, cv2.IMWRITE_EXR_TYPE_FLOAT, cv2.IMWRITE_EXR_COMPRESSION_ZIP, 3),
]
meta = {}
for name, img, typ, comp, nc in CASES:
    rgb = img[..., :nc]
    bgr = rgb[..., [2, 1, 0, 3][:nc]] if nc == 4 else rgb[..., ::-1]
    cv2.imwrite(str(OUT / f"{name}.exr"), np.ascontiguousarray(bgr), [cv2.IMWRITE_EXR_TYPE, typ, cv2.IMWRITE_EXR_COMPRESSION, comp])
    back = cv2.imread(str(OUT / f"{name}.exr"), cv2.IMREAD_UNCHANGED).astype(np.float32)
    planes = [back[..., 2], back[..., 1], back[..., 0]] + ([back[..., 3]] if nc == 4 else [])
    np.ascontiguousarray(np.stack(planes)).astype("<f4").tofile(OUT / f"{name}.expected.bin")
    meta[name] = {"height": H, "width": W, "channels": ["R", "G", "B", "A"][:nc], "cv2": cv2.__version__}
(OUT / "fixtures.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
print("wrote", list(meta))

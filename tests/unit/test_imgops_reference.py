"""Ported CUDA image operators vs independent CPU references (offline only)."""

import math

import numpy as np
import torch

from eyeclean.gpu.imgops import CropTransform, box_blur, dilate, erode, extract_crop, feather_inside, inverse_warp


def _np_dilate(m, r):
    H, W = m.shape
    out = np.zeros_like(m)
    for y in range(H):
        for x in range(W):
            out[y, x] = m[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1].any()
    return out


def _np_erode(m, r):
    H, W = m.shape
    p = np.zeros((H + 2 * r, W + 2 * r), bool)
    p[r:-r, r:-r] = m
    out = np.zeros_like(m)
    for y in range(H):
        for x in range(W):
            out[y, x] = p[y:y + 2 * r + 1, x:x + 2 * r + 1].all()
    return out


def test_morphology_matches_cpu(dev):
    rng = np.random.default_rng(0)
    m = rng.random((24, 31)) > 0.6
    t = torch.tensor(m, device=dev)
    for r in (1, 2):
        assert np.array_equal(dilate(t, r).cpu().numpy(), _np_dilate(m, r))
        assert np.array_equal(erode(t, r).cpu().numpy(), _np_erode(m, r))


def test_box_blur_matches_cpu(dev):
    rng = np.random.default_rng(1)
    x = rng.random((17, 23)).astype(np.float32)
    r = 2
    p = np.pad(x, r, mode="edge")
    ref = np.zeros_like(x)
    for yy in range(x.shape[0]):
        for xx in range(x.shape[1]):
            ref[yy, xx] = p[yy:yy + 2 * r + 1, xx:xx + 2 * r + 1].mean()
    got = box_blur(torch.tensor(x, device=dev), r).cpu().numpy()
    assert np.abs(got - ref).max() < 1e-5


def test_feather_zero_outside_support(dev):
    m = torch.zeros((40, 40), dtype=torch.bool, device=dev)
    m[10:30, 12:28] = True
    a = feather_inside(m, 4)
    assert float(a[~m].abs().max()) == 0.0
    assert float(a[m].max()) == 1.0


def _np_bilinear(img, x, y):
    H, W = img.shape
    x0, y0 = math.floor(x), math.floor(y)
    fx, fy = x - x0, y - y0
    def px(yy, xx):
        return img[yy, xx] if 0 <= yy < H and 0 <= xx < W else img[min(max(yy, 0), H - 1), min(max(xx, 0), W - 1)]
    return (px(y0, x0) * (1 - fx) * (1 - fy) + px(y0, x0 + 1) * fx * (1 - fy) + px(y0 + 1, x0) * (1 - fx) * fy + px(y0 + 1, x0 + 1) * fx * fy)


def test_crop_transform_matches_cpu(dev):
    rng = np.random.default_rng(2)
    img = rng.random((60, 80)).astype(np.float32)
    tf = CropTransform(41.3, 29.7, 1.6, 0.3, 32)
    got = extract_crop(torch.tensor(img, device=dev)[None], tf, padding="border")[0].cpu().numpy()
    c, s = math.cos(tf.theta), math.sin(tf.theta)
    for v in (0, 7, 31):
        for u in (0, 13, 31):
            du, dv = (u + 0.5 - 16) / tf.scale, (v + 0.5 - 16) / tf.scale
            x, y = tf.cx + c * du - s * dv, tf.cy + s * du + c * dv
            assert abs(got[v, u] - _np_bilinear(img, x, y)) < 1e-4


def test_inverse_warp_roundtrip(dev):
    tf = CropTransform(100.0, 80.0, 1.0, 0.0, 64)
    crop = torch.rand((1, 64, 64), device=dev)
    box = tf.frame_box(200, 160)
    back = inverse_warp(crop, tf, box)
    x0, y0 = box[0], box[1]
    # identity-scale, zero rotation: frame pixel (x, y) maps to crop (x - cx + 31.5, ...)
    for (x, y) in ((100, 80), (90, 70), (120, 101)):
        u, v = x - 100 + 31.5, y - 80 + 31.5
        if float(u).is_integer() and float(v).is_integer():
            assert abs(float(back[0, y - y0, x - x0]) - float(crop[0, int(v), int(u)])) < 1e-5

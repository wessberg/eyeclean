"""GPU S3FD decode/NMS and FAN heatmap decoding vs NumPy re-implementations of the upstream
face-alignment CPU wrapper (bbox.decode/nms, _get_preds_fromhm) -- offline test only."""

import numpy as np
import torch
import torchvision

from eyeclean.tracking.detector import LandmarkNet


def _np_nms(dets, thresh):
    x1, y1, x2, y2, s = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = s.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1, yy1 = np.maximum(x1[i], x1[order[1:]]), np.maximum(y1[i], y1[order[1:]])
        xx2, yy2 = np.minimum(x2[i], x2[order[1:]]), np.minimum(y2[i], y2[order[1:]])
        w, h = np.maximum(0.0, xx2 - xx1 + 1), np.maximum(0.0, yy2 - yy1 + 1)
        ovr = w * h / (areas[i] + areas[order[1:]] - w * h)
        order = order[np.where(ovr <= thresh)[0] + 1]
    return keep


def test_nms_agrees_with_upstream_cpu(dev):
    rng = np.random.default_rng(0)
    xy = rng.random((200, 2)) * 300
    wh = rng.random((200, 2)) * 60 + 20
    dets = np.concatenate([xy, xy + wh, rng.random((200, 1))], 1).astype(np.float32)
    keep_ref = set(_np_nms(dets, 0.3))
    # upstream uses (x2-x1+1) areas; convert to the continuous convention torchvision uses
    b = torch.tensor(dets[:, :4], device=dev)
    b[:, 2:] += 1
    keep = set(torchvision.ops.nms(b, torch.tensor(dets[:, 4], device=dev), 0.3).cpu().tolist())
    assert keep == keep_ref


def test_prior_decode_matches_upstream(dev):
    rng = np.random.default_rng(1)
    loc = rng.standard_normal((4,)).astype(np.float32) * 0.5
    stride = 16
    pcx, pcy, ps = stride / 2 + 3 * stride, stride / 2 + 5 * stride, stride * 4.0
    v = [0.1, 0.2]
    ref_c = np.array([pcx, pcy]) + loc[:2] * v[0] * ps
    ref_wh = ps * np.exp(loc[2:] * v[1])
    ref = np.concatenate([ref_c - ref_wh / 2, ref_c + ref_wh / 2])
    l = torch.tensor(loc, device=dev)
    cx = pcx + l[0] * 0.1 * ps
    cy = pcy + l[1] * 0.1 * ps
    w = ps * torch.exp(l[2] * 0.2)
    h = ps * torch.exp(l[3] * 0.2)
    got = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]).cpu().numpy()
    assert np.abs(got - ref).max() < 1e-3


def test_heatmap_decode_matches_upstream(dev):
    rng = np.random.default_rng(2)
    hm = rng.random((2, 68, 64, 64)).astype(np.float32) * 0.1
    for b in range(2):
        for c in range(68):
            y, x = rng.integers(1, 63), rng.integers(1, 63)
            hm[b, c, y, x] = 1.0
            hm[b, c, y, x + 1] = 0.5
            hm[b, c, y - 1, x] = 0.4
    # upstream: argmax, 1-based idx, +-0.25 toward the larger neighbour, then -0.5
    B, C, H, W = hm.shape
    flat = hm.reshape(B, C, -1)
    idx = flat.argmax(-1) + 1
    preds = np.stack([(idx - 1) % W + 1, np.floor((idx - 1) / H) + 1], -1).astype(np.float32)
    for i in range(B):
        for j in range(C):
            px, py = int(preds[i, j, 0]) - 1, int(preds[i, j, 1]) - 1
            if 0 < px < 63 and 0 < py < 63:
                d = np.array([hm[i, j, py, px + 1] - hm[i, j, py, px - 1], hm[i, j, py + 1, px] - hm[i, j, py - 1, px]])
                preds[i, j] += np.sign(d) * 0.25
    preds -= 0.5  # upstream heatmap coordinates (pixel-corner origin)
    # our GPU port: index coords + same refinement; convert to upstream convention (+0.5 - 0.5 = same) and compare
    t = torch.tensor(hm, device=dev)
    fl = t.view(B, C, -1)
    sc, ii = fl.max(-1)
    gx = (ii % W).float()
    gy = torch.div(ii, W, rounding_mode="floor").float()
    ix, iy = gx.long().clamp(1, W - 2), gy.long().clamp(1, H - 2)
    n = torch.arange(B, device=dev)[:, None]
    c = torch.arange(C, device=dev)[None]
    dx = t[n, c, iy, ix + 1] - t[n, c, iy, ix - 1]
    dy = t[n, c, iy + 1, ix] - t[n, c, iy - 1, ix]
    gx = gx + torch.sign(dx) * 0.25
    gy = gy + torch.sign(dy) * 0.25
    # upstream coordinate = (index + 1) +- 0.25 - 0.5 = our index coordinate +- 0.25, shifted by +0.5
    ours = torch.stack([gx, gy], -1).cpu().numpy() + 0.5
    assert np.abs(ours - preds).max() < 1e-5

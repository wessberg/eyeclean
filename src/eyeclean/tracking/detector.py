"""CUDA face detection (S3FD) and 68-point landmarks (2DFAN4) with GPU pre/post-processing.

Reimplements the numerical wrapper of face-alignment on CUDA (the upstream api/utils use NumPy
and OpenCV crops/heatmap decoding on the CPU): resizing, mean subtraction, softmax, prior-box
decoding, non-maximum suppression (torchvision CUDA kernel), crop warping, arg-max and sub-pixel
refinement of heatmaps and the inverse transform all run on the device.  Offline tests compare
these ports against independent CPU references.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
import torchvision

from ..gpu.imgops import CropTransform, extract_crop, resize
from .networks import FAN, S3FD
from .weights import load as load_weights

CENTER_Y_OFFSET = 0.12
REFERENCE_SCALE = 195.0
EYE_IDX = {"image_left": list(range(36, 42)), "image_right": list(range(42, 48))}


class FaceDetector:
    def __init__(self, device: torch.device, max_side: int = 1280, score_thresh: float = 0.5, nms_iou: float = 0.3):
        self.dev = device
        self.net = S3FD().to(device).eval()
        self.net.load_state_dict(load_weights("s3fd", device))
        self.max_side, self.score_thresh, self.nms_iou = max_side, score_thresh, nms_iou

    @torch.no_grad()
    def detect(self, proxy: torch.Tensor, max_side: int | None = None) -> torch.Tensor:
        """proxy: (3,H,W) display-referred [0,1] RGB (CUDA). Returns (N,5) x0,y0,x1,y1,score (frame px)."""
        H, W = proxy.shape[-2:]
        ms = max_side or self.max_side
        s = min(1.0, ms / max(H, W))
        img = resize(proxy, (max(32, round(H * s)), max(32, round(W * s)))) if s < 1.0 else proxy
        sy, sx = img.shape[-2] / H, img.shape[-1] / W
        x = img.flip(0) * 255.0 - torch.tensor([104.0, 117.0, 123.0], device=self.dev).view(3, 1, 1)
        outs = self.net(x[None].float())
        boxes, scores = [], []
        for i in range(len(outs) // 2):
            cls = F.softmax(outs[2 * i], dim=1)[0, 1]
            reg = outs[2 * i + 1][0]
            stride = 2 ** (i + 2)
            m = cls > 0.05
            if not bool(m.any()):
                continue
            ys, xs = torch.nonzero(m, as_tuple=True)
            sc = cls[ys, xs]
            loc = reg[:, ys, xs]  # (4,N)
            pcx = stride / 2 + xs.float() * stride
            pcy = stride / 2 + ys.float() * stride
            ps = stride * 4.0
            cx = pcx + loc[0] * 0.1 * ps
            cy = pcy + loc[1] * 0.1 * ps
            w = ps * torch.exp(loc[2] * 0.2)
            h = ps * torch.exp(loc[3] * 0.2)
            boxes.append(torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1))
            scores.append(sc)
        if not boxes:
            return torch.zeros((0, 5), device=self.dev)
        b = torch.cat(boxes)
        sc = torch.cat(scores)
        keep = torchvision.ops.nms(b, sc, self.nms_iou)
        b, sc = b[keep], sc[keep]
        good = sc > self.score_thresh
        b, sc = b[good], sc[good]
        b = b / torch.tensor([sx, sy, sx, sy], device=self.dev)
        return torch.cat([b, sc[:, None]], 1)


class LandmarkNet:
    def __init__(self, device: torch.device):
        self.dev = device
        self.net = FAN(4).to(device).eval()
        self.net.load_state_dict(load_weights("fan2d", device))

    @staticmethod
    def crop_transform(box: torch.Tensor) -> CropTransform:
        x0, y0, x1, y1 = [float(v) for v in box[:4]]
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0 - (y1 - y0) * CENTER_Y_OFFSET
        scale = ((x1 - x0) + (y1 - y0)) / REFERENCE_SCALE
        return CropTransform(cx - 0.5, cy - 0.5, 256.0 / (200.0 * scale), 0.0, 256)

    @torch.no_grad()
    def __call__(self, proxy: torch.Tensor, boxes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns landmarks (N,68,2) in frame pixel-index coordinates and heatmap peak scores (N,68)."""
        if boxes.shape[0] == 0:
            return torch.zeros((0, 68, 2), device=self.dev), torch.zeros((0, 68), device=self.dev)
        tfs = [self.crop_transform(b) for b in boxes]
        crops = torch.stack([extract_crop(proxy, tf) for tf in tfs])
        hm = self.net(crops.float())  # (N,68,64,64)
        N, C, Hh, Wh = hm.shape
        flat = hm.view(N, C, -1)
        sc, idx = flat.max(-1)
        px = (idx % Wh).float()
        py = torch.div(idx, Wh, rounding_mode="floor").float()
        # quarter-pixel refinement toward the larger neighbour (same rule as upstream)
        ix, iy = px.long().clamp(1, Wh - 2), py.long().clamp(1, Hh - 2)
        n_idx = torch.arange(N, device=self.dev)[:, None]
        c_idx = torch.arange(C, device=self.dev)[None, :]
        dx = hm[n_idx, c_idx, iy, ix + 1] - hm[n_idx, c_idx, iy, ix - 1]
        dy = hm[n_idx, c_idx, iy + 1, ix] - hm[n_idx, c_idx, iy - 1, ix]
        inner = (px > 0) & (px < Wh - 1) & (py > 0) & (py < Hh - 1)
        px = px + torch.where(inner, torch.sign(dx) * 0.25, torch.zeros_like(px))
        py = py + torch.where(inner, torch.sign(dy) * 0.25, torch.zeros_like(py))
        # heatmap index -> crop pixel-centre coordinates (64 -> 256)
        u = (px + 0.5) * (256.0 / Wh) - 0.5
        v = (py + 0.5) * (256.0 / Hh) - 0.5
        out = torch.empty((N, C, 2), device=self.dev)
        for k, tf in enumerate(tfs):
            c, s_ = math.cos(tf.theta), math.sin(tf.theta)
            du = (u[k] + 0.5 - tf.size / 2.0) / tf.scale
            dv = (v[k] + 0.5 - tf.size / 2.0) / tf.scale
            out[k, :, 0] = tf.cx + c * du - s_ * dv
            out[k, :, 1] = tf.cy + s_ * du + c * dv
        return out, sc


def eye_rois_from_landmarks(lm: torch.Tensor) -> list[dict]:
    """Per face: two eye ROIs (centre, eye width, angle) from landmarks 36-41 / 42-47."""
    rois = []
    for f in range(lm.shape[0]):
        for name, idx in EYE_IDX.items():
            pts = lm[f, idx]
            ctr = pts.mean(0)
            a, b = pts[0], pts[3]
            width = torch.linalg.norm(b - a)
            ang = torch.atan2(b[1] - a[1], b[0] - a[0])
            rois.append({"face": f, "which": name, "center": ctr, "width": width, "angle": ang})
    return rois

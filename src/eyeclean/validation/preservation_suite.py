"""Compositor/preservation engineering checks (GPU numerical fixtures, not photoreal CG).

* 4K fixture: 3840x2160 FP32 frame of textured noise with special values (+-0, +-inf, NaN,
  denormals, HDR, negatives) in unedited pixels; random elliptical supports and signed deltas.
  Changed IEEE-754 bit patterns outside the final support must be exactly zero.
* strength 0 and empty support: bit-exact copies of the reference.
* gate after inverse warp: a crop-space delta that is non-zero everywhere is inverse-warped
  onto the frame; only pixels of the full-resolution hard support may change.
* alpha: source alpha returned bit-identically.
* bounded memory over a long stream: peak allocator memory stays flat across many frames.
"""

from __future__ import annotations

import math
import time

import torch

from ..gpu.imgops import CropTransform, feather_inside, inverse_warp
from ..inference.compositor import EyePatch, changed_bits_outside, changed_bits_total, composite


def fixture_4k(dev, seed: int = 7) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device=dev)
    g.manual_seed(seed)
    H, W = 2160, 3840
    base = torch.rand((3, H // 8, W // 8), device=dev, generator=g)
    tex = torch.nn.functional.interpolate(base[None], size=(H, W), mode="bicubic", align_corners=False)[0]
    ref = tex * 4.0 - 0.5 + torch.randn((3, H, W), device=dev, generator=g) * 0.05
    ref[0, 0, 0] = -0.0
    ref[1, 0, 1] = float("inf")
    ref[2, 0, 2] = float("-inf")
    ref[0, 0, 3] = float("nan")
    ref[1, 0, 4] = 1e-40  # denormal
    ref[2, 0, 5] = 6.0e4
    alpha = torch.rand((H, W), device=dev, generator=g)
    alpha[0, 0] = -0.0
    return ref.contiguous(), alpha.contiguous()


def _ellipse(h: int, w: int, dev) -> torch.Tensor:
    ys, xs = torch.meshgrid(torch.linspace(-1, 1, h, device=dev), torch.linspace(-1, 1, w, device=dev), indexing="ij")
    return (xs ** 2 / 0.8 + ys ** 2 / 0.5) < 1


def run_suite(dev) -> dict:
    t0 = time.time()
    ref, alpha = fixture_4k(dev)
    ref_bits = ref.clone()
    H, W = ref.shape[1:]
    g = torch.Generator(device=dev)
    g.manual_seed(11)
    patches = []
    for k in range(6):
        cx, cy = int(300 + k * 600), int(400 + (k % 3) * 600)
        h, w = 180, 260
        box = (cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2)
        hard = _ellipse(h, w, dev)
        delta = torch.randn((3, h, w), device=dev, generator=g) * 0.3
        delta[:, ~hard] = 1e6  # huge garbage outside the support must never leak
        patches.append(EyePatch(box, delta, hard, feather_inside(hard, 6)))
    res = composite(ref, patches, 1.0, alpha)
    outside = changed_bits_outside(ref, res.output, res.support)
    inside_changed = int(((ref.view(torch.int32) != res.output.view(torch.int32)).any(0) & res.support).sum())
    alpha_bits = int((alpha.view(torch.int32) != res.alpha.view(torch.int32)).sum())
    ref_untouched = bool((ref.view(torch.int32) == ref_bits.view(torch.int32)).all())
    r0 = composite(ref, patches, 0.0, alpha)
    strength_zero = changed_bits_total(ref, r0.output)
    empty = [EyePatch(p.box, p.delta, torch.zeros_like(p.hard), p.feather) for p in patches]
    r1 = composite(ref, empty, 1.0, alpha)
    empty_bits = changed_bits_total(ref, r1.output)
    r2 = composite(ref, [], 1.0, alpha)
    no_eye_bits = changed_bits_total(ref, r2.output)
    # gate after inverse warp: crop-space delta non-zero everywhere
    S = 256
    tf = CropTransform(1900.3, 1100.7, 1.7, 0.23, S)
    box = tf.frame_box(W, H)
    crop_delta = torch.ones((3, S, S), device=dev) * 0.5
    delta_full = inverse_warp(crop_delta, tf, box)
    hard_crop = torch.zeros((S, S), device=dev)
    hard_crop[96:160, 80:176] = 1
    hard_full = inverse_warp(hard_crop[None], tf, box)[0] > 0.5
    r3 = composite(ref, [EyePatch(box, delta_full, hard_full, feather_inside(hard_full, 3))], 1.0)
    warp_outside = changed_bits_outside(ref, r3.output, r3.support)
    support_within_hard = bool((r3.support[box[1]:box[3], box[0]:box[2]] <= hard_full).all()) and \
        int(r3.support.sum()) == int((r3.support[box[1]:box[3], box[0]:box[2]] & hard_full).sum())
    # bounded memory over a long stream of 4K frames (fresh patches each frame)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(dev)
    peaks = []
    for i in range(120):
        rr = composite(ref, patches, 0.5 + 0.004 * i, alpha)
        del rr
        if i % 20 == 19:
            torch.cuda.synchronize()
            peaks.append(torch.cuda.max_memory_allocated(dev) / 2**30)
    bounded = max(peaks) - min(peaks) < 0.05
    out = {
        "fixture": {"width": W, "height": H, "kind": "GPU-composited numerical compositor fixture (not photorealistic CG)",
                    "special_values": ["-0", "+inf", "-inf", "NaN", "denormal", "6e4", "negative"]},
        "outside_support_changed_bits": outside, "inside_support_changed_px": inside_changed, "assigned_px": res.assigned_pixels,
        "strength_zero_changed_bits": strength_zero, "empty_support_changed_bits": empty_bits, "no_eye_changed_bits": no_eye_bits,
        "alpha_changed_bits": alpha_bits, "reference_not_modified": ref_untouched,
        "gate_after_inverse_warp": {"outside_changed_bits": warp_outside, "support_within_hard_gate": support_within_hard,
                                    "support_px": int(r3.support.sum())},
        "bounded_memory": {"frames": 120, "peak_allocated_gib_samples": [round(p, 4) for p in peaks], "bounded": bounded},
        "seconds": round(time.time() - t0, 2),
    }
    out["passed"] = (outside == 0 and strength_zero == 0 and empty_bits == 0 and no_eye_bits == 0 and alpha_bits == 0 and ref_untouched
                     and warp_outside == 0 and support_within_hard and bounded and inside_changed > 0)
    return out

"""Preservation compositor (spec section 4.3).

The output starts as a bit-exact copy of the FP32 working-space reference.  Candidate values
``C = B + strength * A * delta`` are computed only inside per-eye ROIs and written with a
boolean masked assignment restricted to the final full-resolution hard support ``H`` (and
finite candidates).  Strength zero and empty support bypass all arithmetic.  No blend such as
``B*(1-m) + P*m`` is ever evaluated over unedited pixels.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class EyePatch:
    """Full-resolution ROI correction for one eye (all CUDA tensors)."""

    box: tuple[int, int, int, int]  # x0, y0, x1, y1 (exclusive) in frame pixels
    delta: torch.Tensor  # (3, h, w) FP32 working-space correction proposal (pre-gating)
    hard: torch.Tensor  # (h, w) bool final hard support H (after inverse warp)
    feather: torch.Tensor  # (h, w) FP32 A in [0, 1]
    synth: torch.Tensor | None = None  # (h, w) bool synthesized (hybrid) support
    uncertainty: torch.Tensor | None = None  # (h, w) FP32
    track_id: int = -1
    eye: str = ""
    meta: dict = field(default_factory=dict)


@dataclass
class CompositeResult:
    output: torch.Tensor  # (3, H, W) FP32
    support: torch.Tensor  # (H, W) bool: pixels actually assigned
    alpha: torch.Tensor | None
    bypass: str | None  # 'strength_zero' | 'empty_support' | None
    assigned_pixels: int


def composite(reference: torch.Tensor, patches: list[EyePatch], strength: float,
              alpha: torch.Tensor | None = None) -> CompositeResult:
    if reference.dtype != torch.float32 or not reference.is_cuda or reference.dim() != 3 or reference.shape[0] != 3:
        raise ValueError("reference must be a (3, H, W) float32 CUDA tensor")
    H, W = reference.shape[1:]
    out = reference.clone()  # bit-exact copy; the reference itself is never modified
    alpha_out = alpha.clone() if alpha is not None else None
    support = torch.zeros((H, W), dtype=torch.bool, device=reference.device)
    if strength == 0.0:
        return CompositeResult(out, support, alpha_out, "strength_zero", 0)
    live = [p for p in patches if p.hard.numel() > 0]
    if not live:
        return CompositeResult(out, support, alpha_out, "empty_support", 0)
    any_support = torch.stack([p.hard.any() for p in live]).any()
    if not bool(any_support.item()):
        return CompositeResult(out, support, alpha_out, "empty_support", 0)
    s = float(strength)
    for p in live:
        x0, y0, x1, y1 = p.box
        if not (0 <= x0 < x1 <= W and 0 <= y0 < y1 <= H):
            raise ValueError(f"patch box {p.box} outside frame {W}x{H}")
        hard = p.hard
        A = p.feather * hard.to(p.feather.dtype)  # re-apply H after feathering / warping
        ref_roi = reference[:, y0:y1, x0:x1]
        cand = ref_roi + (s * A).unsqueeze(0) * p.delta
        valid = hard & torch.isfinite(cand).all(dim=0)
        roi = out[:, y0:y1, x0:x1]  # view into out
        roi[:, valid] = cand[:, valid]  # masked assignment only inside the support
        support[y0:y1, x0:x1] |= valid
    n = int(support.sum().item())
    return CompositeResult(out, support, alpha_out, None if n else "empty_support", n)


def changed_bits_outside(reference: torch.Tensor, output: torch.Tensor, support: torch.Tensor) -> int:
    """Number of pixels (any channel) whose IEEE-754 bit pattern differs outside ``support``."""
    diff = (reference.contiguous().view(torch.int32) != output.contiguous().view(torch.int32)).any(dim=0)
    return int((diff & ~support).sum().item())


def changed_bits_total(reference: torch.Tensor, output: torch.Tensor) -> int:
    return int((reference.contiguous().view(torch.int32) != output.contiguous().view(torch.int32)).any(dim=0).sum().item())


def signed_correction(reference: torch.Tensor, output: torch.Tensor, support: torch.Tensor,
                      box: tuple[int, int, int, int]) -> torch.Tensor:
    """out - reference within ``box``, exactly zero outside the support."""
    x0, y0, x1, y1 = box
    d = output[:, y0:y1, x0:x1] - reference[:, y0:y1, x0:x1]
    return torch.where(support[y0:y1, x0:x1].unsqueeze(0), d, torch.zeros_like(d))


def apply_signed_correction(reference: torch.Tensor, delta: torch.Tensor, support_roi: torch.Tensor,
                            box: tuple[int, int, int, int], out: torch.Tensor | None = None) -> torch.Tensor:
    """Reference ``apply-patches`` semantics: hard-select outside support, add the signed delta
    in the same working space inside it (the delta is already gated: no second alpha)."""
    out = reference.clone() if out is None else out
    x0, y0, x1, y1 = box
    roi = out[:, y0:y1, x0:x1]
    cand = reference[:, y0:y1, x0:x1] + delta
    valid = support_roi & torch.isfinite(cand).all(dim=0)
    roi[:, valid] = cand[:, valid]
    return out

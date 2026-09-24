"""Training objectives (FP32 loss math; network outputs may be BF16)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..gpu.imgops import dilate, erode
from ..models.inputs import denormalize


def charbonnier(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt(x * x + eps * eps)


def wmean(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return (x * w).sum() / w.sum().clamp(min=1.0)


def dice_loss(logits: torch.Tensor, target: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    p = torch.sigmoid(logits.float())
    inter = (p * target * w).sum((2, 3))
    den = ((p + target) * w).sum((2, 3))
    return (1 - (2 * inter + 1) / (den + 1)).mean()


def guard_loss(out: dict, gt: dict, valid: torch.Tensor) -> tuple[torch.Tensor, dict]:
    m = out["masks"].float()
    t = gt["masks"]
    w = valid.expand_as(t) * gt.get("mask_weight", 1.0)
    bce = F.binary_cross_entropy_with_logits(m, t, weight=w, reduction="sum") / w.sum().clamp(min=1)
    dice = dice_loss(m, t, w)
    pres = F.binary_cross_entropy_with_logits(out["present"].float(), gt["present"])
    loss = bce + 0.5 * dice + 0.2 * pres
    parts = {"bce": float(bce), "dice": float(dice), "present": float(pres)}
    if "refl_present" in out and "refl_present" in gt:  # v2 per-eye visible-reflection presence
        rw = gt["refl_present_w"]
        rp = (F.binary_cross_entropy_with_logits(out["refl_present"].float(), gt["refl_present"], weight=rw, reduction="sum")
              / rw.sum().clamp(min=1))
        loss = loss + 0.5 * rp
        parts["refl_present"] = float(rp)
    return loss, parts


def grad_xy(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return x[..., :, 1:] - x[..., :, :-1], x[..., 1:, :] - x[..., :-1, :]


def restore_loss(out: dict, gt: dict, b: dict, e: torch.Tensor, conf_target: str = "tolerance") -> tuple[torch.Tensor, dict]:
    """conf_target: 'tolerance' = |pred - tgt| < 0.02 (v1); 'improvement' (v2) = the correction removes at
    least half of the pixel's error or is within 0.02 - the event conservative editing actually needs."""
    res = out["residual"].float()
    src_n, tgt_n = gt["src_n"], gt["tgt_n"]
    pred_n = src_n + res
    valid = b["valid"]
    ocular = b["labels"][:, 0:1]
    pupil = b["labels"][:, 4:5]
    oc_d = dilate(ocular[:, 0] > 0.05, 3)[:, None].float()
    w_eye = valid * (1.0 + 9.0 * oc_d)
    band = (dilate(pupil[:, 0] > 0.5, 2) & ~erode(pupil[:, 0] > 0.5, 2))[:, None].float()
    w_eye = w_eye * (1.0 + 2.0 * band * oc_d)
    l_rec = wmean(charbonnier(pred_n - tgt_n).mean(1, keepdim=True), w_eye)
    # reconstruction averaged over the correction support itself (not diluted by unchanged pixels)
    sup_w = gt["support"] * valid
    l_rec_sup = wmean(charbonnier(pred_n - tgt_n).mean(1, keepdim=True), sup_w) if bool(sup_w.sum() > 0) else res.sum() * 0.0
    # complementary linear-space HDR check relative to exposure (FP32); the prediction is clamped
    # to the target's normalised range + margin before sinh so overshoots cannot explode
    hi = tgt_n.detach().amax((1, 2, 3), keepdim=True) + 0.25
    pred_lin = denormalize(torch.minimum(torch.maximum(pred_n, torch.full_like(pred_n, -0.25)), hi), e)
    rel = ((pred_lin - b["tgt"]).abs() / e.view(-1, 1, 1, 1)).mean(1, keepdim=True)
    l_lin = wmean(charbonnier(torch.clamp(rel, max=4.0)), valid * oc_d)
    gpx, gpy = grad_xy(pred_n)
    gtx, gty = grad_xy(tgt_n)
    wgx, wgy = (valid * oc_d)[..., :, 1:], (valid * oc_d)[..., 1:, :]
    l_grad = 0.5 * (wmean((gpx - gtx).abs().mean(1, keepdim=True), wgx) + wmean((gpy - gty).abs().mean(1, keepdim=True), wgy))
    # strong identity: no residual where the target equals the source; everywhere on negatives
    no_sup = valid * (1 - gt["support"])
    l_id = wmean(res.abs().mean(1, keepdim=True), no_sup)
    neg = b["negative"].float().view(-1, 1, 1, 1)
    l_neg = wmean(res.abs().mean(1, keepdim=True), valid * neg) if bool(b["negative"].any()) else res.sum() * 0.0
    sup_logit = out["support"].float()
    l_sup = F.binary_cross_entropy_with_logits(sup_logit, gt["support"], weight=valid, reduction="sum") / valid.sum() \
        + 0.5 * dice_loss(sup_logit, gt["support"], valid)
    w_syn = valid * (0.1 + gt["support"])
    l_syn = F.binary_cross_entropy_with_logits(out["synth"].float(), gt["synth"], weight=w_syn, reduction="sum") / w_syn.sum().clamp(min=1)
    err_p = (pred_n.detach() - tgt_n).abs().amax(1, keepdim=True)
    if conf_target == "improvement":
        err_s = (src_n - tgt_n).abs().amax(1, keepdim=True)
        ok = ((err_p <= 0.5 * err_s) | (err_p < 0.02)).float()
    else:
        ok = (err_p < 0.02).float()
    w_conf = valid * (0.2 + gt["support"])
    l_conf = F.binary_cross_entropy_with_logits(out["conf"].float(), ok, weight=w_conf, reduction="sum") / w_conf.sum().clamp(min=1)
    loss = (2.0 * l_rec_sup + l_rec + 0.5 * l_lin + 0.5 * l_grad + 1.0 * l_id + 2.0 * l_neg
            + 0.05 * l_sup + 0.05 * l_syn + 0.02 * l_conf)
    saturated = float((gt["synth"].sum((1, 2, 3)) >= 4).float().mean())
    return loss, {"rec_sup": float(l_rec_sup), "rec": float(l_rec), "lin": float(l_lin), "grad": float(l_grad), "id": float(l_id),
                  "neg": float(l_neg), "sup": float(l_sup), "syn": float(l_syn), "conf": float(l_conf),
                  "saturated_frac": saturated, "negative_frac": float(b["negative"].float().mean())}


def temporal_error_loss(pred_a: torch.Tensor, tgt_a: torch.Tensor, pred_b: torch.Tensor, tgt_b: torch.Tensor,
                        w: torch.Tensor) -> torch.Tensor:
    """Compare prediction *error* between consecutive frames (not raw RGB), over valid support."""
    return wmean(((pred_a - tgt_a) - (pred_b - tgt_b)).abs().mean(1, keepdim=True), w)

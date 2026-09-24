"""Loss terms for the FILL (batch-first: pred/truth (B,3,H,W), masks (B,1,H,W)), derived from what Round 20 actually measured on the user's plate.

The earlier restorer objective (native_objective, development round 8) is entirely
SYMMETRIC - every term is a Charbonnier of (pred - truth), so a region dug darker than the iris and
a region left too bright cost exactly the same.  Round 20 measured that this is the wrong shape:

    model                ABOVE    BELOW   %px<-.05   TEX
    Round 8 incumbent   0.0794  -0.0256     23.4%   1.70
    Round 19            0.0600  -0.0334     30.0%   1.57
    R39 (the target)    0.0175  -0.0179      6.5%   0.59

Both shipped models leave catchlight behind AND dig holes, and both spray roughly three times R39's
gradient energy inside the fill.  `catchlight_metric_fair.py` clamps the deficit to zero so the
published numbers never showed it, and the training loss could not see it either.

Three terms are added here, one per measured axis:

  excess       asymmetric: cost only where the prediction is BRIGHTER than the truth inside the
               support - the catchlight still there, which is the headline number.  Added late, in
               Round 21c; until then ABOVE had no dedicated term at all and removal was driven only
               by the symmetric reconstruction loss.
  undershoot   the mirror: cost only where the prediction is DARKER than the truth.  This is the
               hole-digging, the half of the error that both the acceptance metric and the
               symmetric loss were blind to.
  pupil        the pupil's level, weighted up.  The network fill drove the pupil to 0.0007 against
               R39's 0.0689 (near-black) while a membrane drove it to 0.1301 (iris-coloured); both
               are large level errors on a small number of pixels, which a mean over the whole
               support barely registers.
  texture      matches gradient ENERGY to the target rather than only penalising the gradient of the
               difference.  A fill can have the right mean gradient error while carrying far too
               much structure of its own, which is what TEX 1.57-1.73 versus 0.59 describes.
  chroma       colour error, because the three terms above and every acceptance axis are computed
               on LUMINANCE and are blind to a fill of the right brightness in the wrong hue.  Added
               in Round 21 after the trained fill measured worse on colour than the untouched
               source.  Weight defaults to 0.0 so a caller that predates it is unchanged.

Everything is measured in the same normalised space as the existing terms so the weights are
comparable, and each term is returned separately so a training run can be read per-axis.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _wmean(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return (x * w).sum() / w.sum().clamp(min=1.0)


def charbonnier(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt(x * x + eps * eps)


def undershoot_term(pred: torch.Tensor, truth: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    """Cost only where pred is darker than truth: the dug hole.

    Luminance is used rather than per-channel error so the term describes a LEVEL error, which is
    what the measurement does; a colour shift at the right level is the reconstruction terms' job.
    """
    w = torch.tensor([0.2126, 0.7152, 0.0722], device=pred.device, dtype=pred.dtype).view(1, -1, 1, 1)
    deficit = ((truth * w).sum(1, keepdim=True) - (pred * w).sum(1, keepdim=True)).clamp(min=0.0)
    return _wmean(charbonnier(deficit), support.float())


def excess_term(pred: torch.Tensor, truth: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    """Cost only where pred is BRIGHTER than truth: the catchlight still there.

    The mirror of `undershoot_term`, and the term that was missing.  Round 20 measured two failure
    directions - ABOVE (catchlight left behind) and BELOW (iris dug out) - and only BELOW ever got
    a dedicated term.  ABOVE was left to the symmetric reconstruction loss, so Round 21b, which cut
    `rec` to 18% and raised `undershoot` to 26%, gave the objective a brake on over-removal and no
    accelerator for removal at all.  Measured on held-out pairs against their own ground truth, that
    model removed only 51.4% of the excess while keeping texture (TEX 0.99 vs truth 1.07), colour
    (0.0569 vs 0.0617) and holes (0.4%) essentially perfect: it stopped halfway because nothing
    pushed it further.

    Luminance, like its mirror, because this is a LEVEL error; colour is `chroma_term`'s job.
    """
    w = torch.tensor([0.2126, 0.7152, 0.0722], device=pred.device, dtype=pred.dtype).view(1, -1, 1, 1)
    excess = ((pred * w).sum(1, keepdim=True) - (truth * w).sum(1, keepdim=True)).clamp(min=0.0)
    return _wmean(charbonnier(excess), support.float())


def pupil_term(pred: torch.Tensor, truth: torch.Tensor, pupil: torch.Tensor) -> torch.Tensor:
    """Level error inside the pupil, where both failure directions have been observed."""
    if pupil.sum() < 1:
        return pred.sum() * 0.0
    w = torch.tensor([0.2126, 0.7152, 0.0722], device=pred.device, dtype=pred.dtype).view(1, -1, 1, 1)
    d = (pred * w).sum(1, keepdim=True) - (truth * w).sum(1, keepdim=True)
    return _wmean(charbonnier(d), pupil.float())


def texture_term(pred: torch.Tensor, truth: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    """Match gradient ENERGY, not just the gradient of the difference.

    |grad(pred)| - |grad(truth)| is positive when the fill invents structure and negative when it is
    a smooth wash; both were observed (network 1.57-1.73, membrane 0.37, target 0.59), so the term
    is symmetric in the mismatch even though `undershoot` above is deliberately one-sided.
    """
    def gmag(t):
        gx = t[..., :, 1:] - t[..., :, :-1]
        gy = t[..., 1:, :] - t[..., :-1, :]
        gx = F.pad(gx, (1, 0, 0, 0))
        gy = F.pad(gy, (0, 0, 1, 0))
        return torch.sqrt(gx * gx + gy * gy + 1e-8).mean(1, keepdim=True)

    return _wmean(charbonnier(gmag(pred) - gmag(truth)), support.float())


def chroma_term(pred: torch.Tensor, truth: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    """Colour error inside the support, independent of brightness.

    Every other term here, and every axis of `round20_fill.py`, is computed on luminance
    (0.2126 R + 0.7152 G + 0.0722 B).  A fill at exactly the right brightness in exactly the wrong
    hue therefore scores perfectly and is penalised only by the symmetric reconstruction term, which
    is spread over the whole crop.  Round 21 walked straight into that: its step-2000 checkpoint
    removes the catchlight and paints a purple-magenta mark in its place, measuring WORSE on colour
    than the untouched source (0.0862 against 0.0692, R39 0.0341) and degrading with training
    (0.0991 by step 3000).

    Opponent axes (R-G, G-B) are used rather than a chroma angle: they are zero for neutral grey,
    cheap, and differentiable everywhere - a hue angle is undefined as saturation goes to zero,
    which is exactly where a pupil sits.
    """
    def opp(x):
        return torch.stack([x[:, 0] - x[:, 1], x[:, 1] - x[:, 2]], dim=1)

    d = opp(pred) - opp(truth)
    return _wmean(charbonnier(d).mean(1, keepdim=True), support.float())


def fill_objective(pred: torch.Tensor, truth: torch.Tensor, support: torch.Tensor,
                   pupil: torch.Tensor, weights: dict | None = None) -> tuple[torch.Tensor, dict]:
    """Returns (extra_loss, parts).  Added to the existing reconstruction objective, not replacing
    it: the symmetric terms still carry the bulk of the signal, and these steer the three failure
    modes that were measured but previously unpenalised."""
    w = {"undershoot": 3.0, "pupil": 2.0, "texture": 1.0, "chroma": 0.0, "excess": 0.0,
         **(weights or {})}
    u = undershoot_term(pred, truth, support)
    x = excess_term(pred, truth, support)
    p = pupil_term(pred, truth, pupil)
    t = texture_term(pred, truth, support)
    c = chroma_term(pred, truth, support)
    total = (w["undershoot"] * u + w["pupil"] * p + w["texture"] * t + w["chroma"] * c
             + w["excess"] * x)
    return total, {"undershoot": float(u.detach()), "pupil_level": float(p.detach()),
                   "texture_energy": float(t.detach()), "chroma": float(c.detach()),
                   "excess": float(x.detach())}

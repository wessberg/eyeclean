"""Final full-resolution hard edit support H_t (spec 4.2).

H = user max-region  ∧  high-confidence tracked aperture  ∧  visible ocular surface without
lids/lashes/tear line (eroded at full resolution)  ∧  (selected reflection ∪ declared
reconstruction support)  ∧  per-eye validity (track, open eye, exposure, geometry, minimum size).
All masks are warped from crop space back to the frame ROI first; H is thresholded and
re-applied after warping and feathering.  Any failed validity condition yields zero support.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from ..gpu.imgops import CropTransform, dilate, erode, feather_inside, inverse_warp

DEFAULT_GATE = {
    "present_thresh": 0.5, "ocular_thresh": 0.5, "protected_thresh": 0.5, "support_thresh": 0.5, "reflection_thresh": 0.5,
    "conf_thresh": 0.5, "synth_thresh": 0.5, "min_reflection_area_crop_px": 12.0, "min_mean_correction": 0.01,
    "erode_frac": 0.05, "erode_min_px": 2, "feather_frac": 0.05,
    # where the lid-margin erosion and the feather ramp are computed; see full_res_gate
    "margin_space": "native", "erode_min_crop_px": 2, "feather_min_crop_px": 2,
    # Track hysteresis for the eye-level presence decision (off = bit-identical behaviour).
    # A reflection from a fixed light does not vanish for a few frames while the eye reopens after a
    # blink; measured on the development shot, 5 open frames after a blink failed only the mean-correction
    # test (0.03-0.079 vs 0.08) while the presence classifier stayed at 0.86-0.94.  When most open
    # frames of the same track so far (and the two look-ahead frames) passed the strict tests, a frame
    # is edited if it still has selected area, at least this fraction of the correction threshold and
    # at least this fraction of the classifier threshold.  Single stills (the validation harness, the
    # controlled no-reflection case) have no such history and are unaffected; closed eyes stay zero
    # through eye_validity regardless.
    "presence_hysteresis": False, "hysteresis_min_track_frac": 0.5, "hysteresis_min_history": 3,
    "hysteresis_corr_frac": 0.25, "hysteresis_present_frac": 0.5,
    # Fill mode inside the gate.
    #   "network"        the restorer's correction everywhere in H, as a crop-space delta warped to the frame.
    #   "network_value"  the same, except inside the hole (below), where the restorer's fill reaches the
    #                    frame as a VALUE.  A delta composites  ref + warp(pred - crop) , and warp(crop) is a
    #                    blurred ref, so the source's high-pass - the catchlight's sharp edges - survives
    #                    every composite; replacing the value removes it.
    # Earlier experimental modes (membrane, polar-copy and atlas fills, pupil-circle fills) were retired in
    # 0.3.0; a bundle that asks for one is refused (see RETIRED_GATE).
    "fill_mode": "network",
    # native pixels with at least this much coverage by the warped value core take the fill value
    "membrane_native_cover": 0.2,
    # highlight test used by hybrid mode's canonical pupil observations: display luminance above the local
    # background by this much marks a highlight (and its halo) that must not enter the atlas
    "membrane_core_excess": 0.10,
    # size-dependent crop-space margin (0 = off): irises below small_iris_px use erode_min_crop_px_small /
    # erode_frac_small instead of the general margin
    "small_iris_px": 0.0, "erode_min_crop_px_small": 4, "erode_frac_small": 0.0,
    # Restrict the editable surface to the iris/pupil (0 = off, legacy behaviour).  The softbox catchlight
    # this tool removes lies on the iris; a wet-eye glint on the SCLERA is a natural feature a hand fix
    # leaves in place.  With conf_thresh low enough to recover the catchlight, the restorer also fires on
    # the sclera glint and leaves a grey smudge there.  This is the anatomical constraint that says where a
    # softbox reflection can be, rather than a confidence threshold tuned to hide the symptom.
    "iris_only_thresh": 0.0,
    # The hole: the guard's reflection channel above hole_refl_thresh, dilated hole_dilate_px crop px.  It
    # is the value-fill region of fill_mode "network_value".
    "hole_refl_thresh": 0.5, "hole_dilate_px": 2,
    # The restorer's input: "source" (the crop as it is) or "hole" (a masked-inpainting restorer: the hole is blanked
    # with the eye's mean colour and given as the reflection channel, models.inputs.hole_restore_input, and the
    # restorer's residual applies to the blanked crop).  A restorer must be trained for the input it is given.
    "restore_input": "source",
    # Temporal union of the reflection channel over +-N frames of the same track (0 = off), warped into
    # the current crop.  Measured on the development shot: the per-frame gate switched the whole 50 px
    # fill on and off in runs of 6-15 frames while the eye was open (presence 0.2-0.3 on frames where the
    # catchlight is faint), and the edited outline moved 13% per frame - both read as flicker.
    "temporal_refl_window": 0,
    # Inside the hole, exempt from the iris-only rule pixels within this many crop px of the iris/pupil
    # (0 = off).  The top rows of a catchlight lying against the upper lid read as neither iris nor pupil
    # to the guard and were left unedited.  Sclera glints away from the iris stay excluded.
    "hole_iris_relax_px": 0,
    # Take the lid margin (erode_frac / erode_min_crop_px) from the lid side of the surface only (0 = off, legacy).
    # The margin is there for the guard's boundary error at the lids.  Eroding the iris-only surface as a whole also
    # erodes it at the limbus, where no lid is: a catchlight that reaches the iris edge keeps its outer band, and the
    # erosion undoes hole_iris_relax_px.  On: the ocular-minus-protected surface is eroded, then the iris-only
    # surface is applied unchanged.  crop margin_space only.
    "margin_lid_only": 0,
    # Exempt the hole from the lid margin (0 = off, legacy).  At photo scale the margin (7 crop px at a 93 px crop
    # iris) exceeds hole_lid_margin_px, so a catchlight lying against a lid keeps its lid-side band even though the
    # hole covers it.  1: hole pixels on the surface keep their full extent (including the hole_lid_margin_px
    # admissions).  2: only hole pixels inside the guard's own ocular-minus-protected surface.  crop margin_space only.
    "hole_skip_margin": 0,
    # Cap the value fill's luminance (0 = off).  Inside the value-fill core, a filled pixel brighter than the
    # q-quantile of the SOURCE luminance in a ring around the core is scaled down to it (one factor for all channels,
    # so colour is kept).  The restorer sees the catchlight in its input and leaves part of it in the fill; the cap
    # only darkens, and only inside the hole.  The ring runs from fill_luma_cap_ring x iris to twice that, outside the
    # core, to skip the catchlight's own halo.
    "fill_luma_cap_q": 0.0, "fill_luma_cap_ring": 0.1,
    # When a reflection is present, the (temporally unioned) hole on the eye surface is itself selected /
    # recoverable, so the hard support follows the track's reflection mask rather than the restorer's
    # per-frame confidence outline (which moved 13% per frame).  Off = legacy.
    "hole_defines_support": False,
    # The catchlight's top rows lie against the upper lid, where the guard's ocular / protected masks
    # exclude them, and that boundary is decided per frame, so the residual there blinks.
    # hole_lid_margin_px: inside the hole, admit pixels within this many crop px of the eye surface.
    # hole_surface_union: inside the hole, admit pixels that were eye surface in any frame of the temporal
    # window (warped).  0 / off = legacy.
    "hole_lid_margin_px": 0, "hole_surface_union": False,
    # Soften the render pass's input crop: the frame region feeding the crop is blurred by a Gaussian of
    # this many NATIVE px before the bilinear crop (guard and restorer both see it; tracking and the output
    # path do not).  The fill amplifies high-frequency input change ~2.8x; 0.5 px removes 42-47% of the
    # development shot's removable frame-to-frame change with brightness, colour, texture and removal at
    # the unsoftened level, where 0.8 px and up brighten the fill.  0 = off (legacy).
    "input_soften_sigma": 0.0,
}

# Gate settings retired in 0.3.0 (research modes no promoted model uses).  A bundle that sets one of them to
# anything but its neutral value is refused rather than silently rendered differently.
RETIRED_GATE = {
    "fill_mode": ("network", "network_value"),          # allowed values; anything else is retired
    "restore_input": ("source", "hole"),                 # "hole" reinstated in round 56 (masked-inpainting restorer)
    "value_core_source": ("hole",),
    "value_pupil_circle": (False, 0),
    "value_pupil_full_disc": (False, 0),
    "value_canonical_pupil": (False, 0),
    "fill_temporal_window": (0,),
}


def check_retired(g: dict) -> None:
    bad = {k: g[k] for k, allowed in RETIRED_GATE.items() if k in g and g[k] not in allowed}
    if bad:
        raise ValueError(f"gate settings retired in EyeClean 0.3.0: {bad}; use EyeClean <= 0.2 for this bundle")


@dataclass
class GateDecision:
    valid: bool
    reasons: list = field(default_factory=list)


def eye_validity(present: float, iris_px: float, min_iris_px: float, track_ok: bool, exposure_ok: bool, geometry_ok: bool,
                 g: dict) -> GateDecision:
    reasons = []
    if not track_ok:
        reasons.append("track_invalid")
    if present < g["present_thresh"]:
        reasons.append("eye_closed_or_not_visible")
    if iris_px < min_iris_px:
        reasons.append(f"iris_below_minimum_{min_iris_px:g}px")
    if not exposure_ok:
        reasons.append("exposure_invalid")
    if not geometry_ok:
        reasons.append("geometry_unreliable")
    return GateDecision(not reasons, reasons)


def crop_support(probs: torch.Tensor, out: dict, mode: str, g: dict, refl_present: float | None = None,
                 relaxed: bool = False, hole: torch.Tensor | None = None) -> dict:
    """Crop-space Boolean components. probs: (6,S,S) EyeGuard sigmoid; out: restore outputs for one eye.
    refl_present: EyeGuard's per-eye visible-reflection probability (v2 checkpoints) - when given it
    replaces the mean-correction heuristic in the eye-level "is there a reflection" decision.
    hole: (S,S) bool, the Round 23 hole (possibly a temporal union); with hole_iris_relax_px > 0 the
    iris-only rule is relaxed inside it near the iris/pupil."""
    ocular = probs[0] > g["ocular_thresh"]
    protected = probs[3] > g["protected_thresh"]
    refl = probs[4] > g["reflection_thresh"]
    sup = torch.sigmoid(out["support"][0].float()) > g["support_thresh"]
    conf = torch.sigmoid(out["conf"][0].float())
    synth = torch.sigmoid(out["synth"][0].float()) > g["synth_thresh"]
    surface = ocular & ~protected
    lid_surface = surface
    iot = float(g.get("iris_only_thresh", 0.0))
    if iot > 0.0:
        on_iris = (probs[1] > iot) | (probs[2] > iot)
        relax = int(g.get("hole_iris_relax_px", 0))
        if hole is not None and relax > 0:
            on_iris = on_iris | (hole & dilate(on_iris, relax))
        surface = surface & on_iris
    selected = (sup | refl) & surface
    recoverable = selected & (conf > g["conf_thresh"]) & ~synth
    if mode == "hybrid":
        # spec 7.5: hybrid *additionally* reconstructs what conservative abstains on (saturated or
        # low-confidence selected reflection) from the fixed canonical state; the pipeline limits
        # this candidate set to the canonical iris/pupil and leaves the rest abstained
        synth_sel = selected & ~recoverable
        editable = recoverable | synth_sel
    else:
        synth_sel = selected & synth
        editable = recoverable
    area = float((selected.float()).sum())
    # a reflection is "present" only if the selected area is non-trivial AND the network proposes a
    # non-trivial correction there (negatives are trained towards an exactly-zero residual)
    mag = out["residual"].float().abs().amax(0) if "residual" in out else None
    mean_corr = float(mag[selected].mean()) if (mag is not None and area > 0) else 0.0
    # Two independent pieces of evidence, both selected on validation (either threshold may be 0,
    # which disables that test).  Measured at zero false alarms on reflection-free eyes, the
    # restorer's mean correction is the stronger detector of the two - it is trained towards an
    # exactly-zero residual on negatives - so the classifier is not allowed to carry the decision
    # alone unless the measurement says it should.
    area_ok = area >= g["min_reflection_area_crop_px"]
    corr_ok = mean_corr >= g.get("min_mean_correction", 0.0)
    if refl_present is not None:
        present_refl = area_ok and corr_ok and refl_present > g.get("refl_present_thresh", 0.5)
    else:
        present_refl = area_ok and corr_ok
    strict_present = present_refl
    if relaxed and not present_refl:
        # track hysteresis (see DEFAULT_GATE): the caller has established that this eye's track shows the
        # reflection in most of its open frames; weaker but consistent evidence on this frame suffices
        mmc = float(g.get("min_mean_correction", 0.0))
        thr = float(g.get("refl_present_thresh", 0.5))
        corr_ok_r = mean_corr >= float(g.get("hysteresis_corr_frac", 0.25)) * mmc
        rp_ok_r = refl_present is None or refl_present > float(g.get("hysteresis_present_frac", 0.5)) * thr
        present_refl = area_ok and corr_ok_r and rp_ok_r
    return {"surface": surface, "selected": selected, "recoverable": recoverable, "synth": synth_sel if mode == "hybrid" else torch.zeros_like(synth_sel),
            "abstained": selected & ~editable, "editable": editable if present_refl else torch.zeros_like(editable),
            "conf": conf, "reflection_present": present_refl, "strict_present": strict_present,
            "synth_head": selected & synth, "protected": protected, "lid_surface": lid_surface,
            # the two eye-level statistics, so a parameter sweep can re-apply the presence decision
            # without recomputing the masks (it only ever zeroes the whole eye)
            "area": area, "mean_corr": mean_corr}


def cap_fill_luma(filled: torch.Tensor, ref: torch.Tensor, core: torch.Tensor, iris_px: float, q: float,
                  ring_frac: float = 0.1, working_color: str = "linear-acescg", min_ring_px: int = 12) -> torch.Tensor:
    """(3,H,W) working-space fill, (3,H,W) source, (H,W) bool value-fill core -> the fill with every core pixel's
    luminance capped at the q-quantile of the source luminance in a ring around the core (gate key fill_luma_cap_q).
    Pixels outside the core, and core pixels already under the cap, are returned unchanged.  GPU ops only."""
    if not bool(core.any()):
        return filled
    w = (0.2722, 0.6741, 0.0537) if working_color == "linear-acescg" else (0.2126, 0.7152, 0.0722)
    wt = torch.tensor(w, device=filled.device, dtype=filled.dtype).view(3, 1, 1)
    r = max(1, int(round(ring_frac * iris_px)))
    ring = dilate(core, 2 * r) & ~dilate(core, r)
    if int(ring.sum()) < min_ring_px:
        return filled
    cap = torch.quantile((ref * wt).sum(0)[ring].float(), q)
    lum = (filled * wt).sum(0)
    scale = torch.where(core & (lum > cap), cap / lum.clamp(min=1e-8), torch.ones_like(lum))
    return filled * scale[None]


def highlight_excess(crop_w: torch.Tensor, working_color: str = "linear-acescg") -> torch.Tensor:
    """Display-referred luminance of a working-space crop minus a local background (9 px minimum
    filter then 25 px mean): positive where a highlight rises above its surround.  GPU ops only."""
    from ..color import transforms as CT

    lin = CT.working_to_linear709(crop_w).clamp(0, 1) if working_color == "linear-acescg" else crop_w.clamp(0, 1)
    d = CT.bt709_oetf(lin)
    L = (0.2126 * d[0] + 0.7152 * d[1] + 0.0722 * d[2])[None, None]
    # local background = mean over a window that EXCLUDES pixels already found bright (two passes),
    # so a highlight does not lift its own background; window ~ a quarter of the crop's iris
    S = L.shape[-1]
    r = max(6, S // 20)
    box = lambda t: torch.nn.functional.avg_pool2d(torch.nn.functional.pad(t, (r, r, r, r), mode="replicate"), 2 * r + 1, stride=1)
    keep = torch.ones_like(L)
    bg = box(L)
    for _ in range(2):
        keep = ((L - bg) <= 0.06).float()
        bg = box(L * keep) / box(keep).clamp(min=1e-3)
    return (L - bg)[0, 0]


def full_res_gate(comp: dict, tf: CropTransform, box: tuple[int, int, int, int], iris_px: float, g: dict,
                  max_region: torch.Tensor | None) -> dict:
    """Warp crop-space components to the frame ROI, threshold, erode, intersect, feather, re-apply H.

    ``margin_space`` decides where the lid-margin erosion and the feather ramp are computed.
    ``native`` (legacy) does whole-pixel morphology on the warped masks; the margin is then quantised
    to whole frame pixels, which at a small eye is enormous - at an 18 px iris a 2 px erosion is 11%
    of the iris from every side and removes a catchlight lying against the upper lid entirely.
    ``crop`` computes the same *physical* margin (a fraction of the iris) in crop space, which is the
    masks' own resolution (~5x native for a 1080p face), and lets the inverse warp carry it back at
    sub-pixel precision.  For large eyes the two agree, because ``erode_frac * iris`` is scale
    invariant.  Either way ``A`` is re-multiplied by ``H`` and so is exactly zero outside it.
    """
    pd = int(g.get("protected_dilate_px", 0))
    fmode = g.get("feather_mode", "support")
    if g.get("margin_space", "native") == "crop":
        iris_c = max(1.0, iris_px * tf.scale)  # iris diameter in crop pixels
        mc = max(int(g.get("erode_min_crop_px", 2)), int(round(g["erode_frac"] * iris_c)))
        rc = max(int(g.get("feather_min_crop_px", 2)), int(round(g["feather_frac"] * iris_c)))
        if iris_px < float(g.get("small_iris_px", 0.0)):
            # Round 4: every lid-edge leak measured on validation was on irises >= 77 px, where the guard's
            # boundary error is several native pixels; at small eyes the crop is a 4-5x upsample and the
            # same fraction of the iris is a fraction of a native pixel of real precision, so a smaller
            # margin can be legal there.  Verified by the selection harness per setting, never assumed.
            mc = max(int(g.get("erode_min_crop_px_small", 4)), int(round(float(g.get("erode_frac_small", 0.0)) * iris_c)))
            rc = min(rc, mc)
        surf_c = comp["surface"]
        lid_c = comp.get("lid_surface") if int(g.get("margin_lid_only", 0)) else None
        if pd > 0 and "protected" in comp:
            pdm = dilate(comp["protected"], max(1, int(round(pd * tf.scale))))
            surf_c = surf_c & ~pdm
            lid_c = lid_c & ~pdm if lid_c is not None else None
        # lid_c | surf_c: the pipeline may admit hole pixels next to the lids into the surface after crop_support
        surf_ec = erode(surf_c, mc) if lid_c is None else erode(lid_c | surf_c, mc) & surf_c
        hsm = int(g.get("hole_skip_margin", 0))
        if hsm and comp.get("hole") is not None:
            keep = comp["hole"] & surf_c
            if hsm == 2 and "lid_surface" in comp:
                keep = keep & comp["lid_surface"]
            surf_ec = surf_ec | keep
        ed_c = comp["editable"] & surf_ec
        a_c = feather_inside(surf_ec if fmode == "surface" else ed_c, rc)
        w = inverse_warp(torch.stack([ed_c.float(), surf_ec.float(), comp["synth"].float(), 1 - comp["conf"],
                                      comp["selected"].float(), comp["abstained"].float(), a_c]), tf, box)
        ed, surf_e, syn, unc = w[0] > 0.5, w[1] > 0.5, w[2] > 0.5, w[3]
        selected, abstained = w[4] > 0.5, w[5] > 0.5
        H = ed & surf_e
        if max_region is not None:
            x0, y0, x1, y1 = box
            H = H & (max_region[y0:y1, x0:x1] > 0.5)
        A = w[6].clamp(0, 1) * H.float()  # finite support: zero wherever H is zero
        return {"hard": H, "feather": A, "synth": syn & H, "uncertainty": unc * H.float(),
                "selected": selected & surf_e, "abstained": abstained & surf_e & ~H}
    stack = torch.stack([comp["editable"].float(), comp["surface"].float(), comp["synth"].float(), 1 - comp["conf"],
                         comp["selected"].float(), comp["abstained"].float()])
    w = inverse_warp(stack, tf, box)
    ed, surf, syn, unc = w[0] > 0.5, w[1] > 0.5, w[2] > 0.5, w[3]
    selected, abstained = w[4] > 0.5, w[5] > 0.5
    if pd > 0 and "protected" in comp:
        # thin lashes / tear line inside the eye opening: exclude a full-resolution band around them
        surf = surf & ~dilate(inverse_warp(comp["protected"].float()[None], tf, box)[0] > 0.5, pd)
    m = max(int(g.get("erode_min_px", 1)), int(round(g["erode_frac"] * iris_px)))
    surf_e = erode(surf, m)  # keep the support strictly inside the ocular surface at full resolution
    H = ed & surf_e
    if max_region is not None:
        x0, y0, x1, y1 = box
        H = H & (max_region[y0:y1, x0:x1] > 0.5)
    r = max(1, int(round(g["feather_frac"] * iris_px)))
    if fmode == "surface":
        # ramp only against the ocular-surface boundary (lids/lashes); inside the eye the correction
        # itself goes smoothly to zero where a reflection ends, so reflection edges keep full strength
        A = feather_inside(surf_e, r) * H.float()
    else:
        A = feather_inside(H, r) * H.float()
    return {"hard": H, "feather": A, "synth": syn & H, "uncertainty": unc * H.float(),
            "selected": selected & surf_e, "abstained": abstained & surf_e & ~H}

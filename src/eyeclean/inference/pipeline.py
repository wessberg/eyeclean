"""Two-pass eye-reflection removal pipeline (all numerical work on CUDA).

Pass 1 (analysis, bounded memory): per frame full-frame face detection (S3FD) and landmarks (FAN)
on a separate display-referred detector proxy, eye ROIs, two EyeGuard refinements, cut detection
and tracking.  Hybrid mode additionally accumulates confidence-weighted visible-iris observations
per shot and track.  After the pass: crop parameters are smoothed per track and canonical iris
states are frozen.
Pass 2 (output): the source is decoded again with a bounded ring buffer (frame_buffer_count) and
a two-frame lookahead; per-track restore inputs and encoder features are cached per frame and a
centred five-frame window is fused.  Corrections are inverse-warped, gated at full resolution and
applied by the separate preservation compositor.  Prediction never receives render coordinates
or targets.
"""

from __future__ import annotations

import math
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field

import torch

from ..color import transforms as CT
from ..gpu.imgops import CropTransform, dilate, extract_crop, warp_crop_to_crop
from ..models.bundle import Bundle
from ..models.inputs import (clipped_mask, crop_exposure, denormalize, guard_input, hole_mask, hole_restore_input,
                             iris_geometry, normalize, restore_input)
from ..models.presence import presence_input
from ..tracking.detector import FaceDetector, LandmarkNet, eye_rois_from_landmarks
from ..tracking.tracker import CutDetector, EyeTracker
from . import canonical as CAN
from .compositor import EyePatch, changed_bits_outside, composite
from .gate import DEFAULT_GATE, cap_fill_luma, check_retired, crop_support, eye_validity, full_res_gate, highlight_excess

K_CROP = 2.75  # crop size in iris diameters (middle of the training range)


@dataclass
class EyeRec:
    frame: int
    shot: int
    track: int
    which: str
    cx: float
    cy: float
    diam: float
    angle: float
    present: float
    face_score: float
    iris_visible: float
    tf: CropTransform | None = None
    reasons: list = field(default_factory=list)
    # always null since 0.3.0 (the pupil-circle fills were retired); kept so tracks.json keeps its fields
    pcx: float | None = None
    pcy: float | None = None
    prad: float | None = None

    def as_dict(self) -> dict:
        d = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.__dict__.items() if k not in ("tf",)}
        if self.tf is not None:
            d["crop"] = {k: round(v, 4) if isinstance(v, float) else v for k, v in self.tf.as_dict().items()}
        return d


class Pipeline:
    def __init__(self, bundle: Bundle, device: torch.device, *, mode: str = "conservative", strength: float = 1.0,
                 working_color: str = "linear-acescg", max_faces: int = 4, min_iris_px: float = 8.0, buffer_frames: int = 7,
                 temporal: bool = True, max_region: torch.Tensor | None = None, log=print):
        if mode not in ("conservative", "hybrid"):
            raise ValueError(mode)
        if temporal and bundle.restore is None:
            raise ValueError("this bundle has no five-frame restorer; run it with restore_mode single")
        self.b, self.dev, self.mode, self.strength = bundle, device, mode, float(strength)
        self.working = working_color
        self.max_faces, self.min_iris = max_faces, float(min_iris_px)
        self.buffer_frames = max(5, buffer_frames)
        self.temporal = temporal
        self.S = bundle.crop_size
        self.gate = dict(DEFAULT_GATE, **bundle.gate)
        check_retired(self.gate)
        if not temporal and "conf_thresh_single" in self.gate:  # each restore path has its own calibrated threshold
            self.gate["conf_thresh"] = self.gate["conf_thresh_single"]
        self.det = FaceDetector(device)
        self.lmk = LandmarkNet(device)
        self.max_region = max_region
        self.log = log
        self.timing = defaultdict(float)
        self.records: dict[int, list[EyeRec]] = {}
        self.shots: dict[int, int] = {}
        self.canon: dict[str, CAN.CanonicalState] = {}
        self.canon_meta: dict[str, dict] = {}
        self.excess_faces = 0
        self.cuts: list[int] = []
        self._last_refl_present: float | None = None
        # Round 4 track hysteresis: per (shot, track) count of open frames flushed so far and how many of
        # them passed the strict eye-level presence tests (see gate.DEFAULT_GATE["presence_hysteresis"])
        self.track_tally: dict[tuple[int, int], dict] = {}

    # ------------------------------------------------------------------ helpers
    def _guard(self, crop: torch.Tensor):
        e = crop_exposure(crop[None])
        gi = guard_input(crop[None], e)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = self.b.guard(gi)
        probs = torch.sigmoid(o["masks"].float())
        # v2 checkpoints: per-eye visible-reflection probability (dedicated classifier if bundled,
        # else the guard's own head; None for older bundles -> v1 heuristic in the gate)
        if self.b.presence is not None:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                self._last_refl_present = float(torch.sigmoid(self.b.presence(presence_input(gi, probs)).float())[0])
        else:
            self._last_refl_present = float(torch.sigmoid(o["refl_present"].float())[0]) if "refl_present" in o else None
        return probs[0], float(torch.sigmoid(o["present"].float())[0]), e

    def _refine(self, ref: torch.Tensor, cx: float, cy: float, diam: float, ang: float):
        """Two EyeGuard passes: ROI crop -> iris geometry -> iris-centred crop."""
        S = self.S
        tf0 = CropTransform(cx, cy, S / (K_CROP * max(diam, 2.0)), ang, S)
        probs0, pres0, _ = self._guard(extract_crop(ref, tf0))
        gcx, gcy, gr, vis = iris_geometry(probs0[1:2][None], probs0[2:3][None], S)
        if float(vis) > 0.15:
            du, dv = float(gcx) + 0.5 - S / 2, float(gcy) + 0.5 - S / 2
            c, s = math.cos(ang), math.sin(ang)
            cx = cx + (c * du - s * dv) / tf0.scale
            cy = cy + (s * du + c * dv) / tf0.scale
            diam = 0.5 * diam + 0.5 * (2 * float(gr) / tf0.scale)
        tf1 = CropTransform(cx, cy, S / (K_CROP * max(diam, 2.0)), ang, S)
        crop1 = extract_crop(ref, tf1)
        probs1, pres1, e1 = self._guard(crop1)
        _, _, _, vis1 = iris_geometry(probs1[1:2][None], probs1[2:3][None], S)
        return cx, cy, diam, pres1, float(vis1), crop1, probs1, e1

    # ------------------------------------------------------------------ pass 1
    @torch.no_grad()
    def analyze(self, source) -> dict:
        tracker = EyeTracker()
        cuts = CutDetector()
        t0 = time.time()
        n = 0
        for fr in source:
            ts = time.time()
            proxy = CT.detector_proxy(fr.reference, self.working)
            is_cut, _ = cuts(proxy)
            if is_cut:
                shot = tracker.new_shot()
                if fr.index > 0:
                    self.cuts.append(fr.index)
            self.shots[fr.index] = tracker.shot
            boxes = self.det.detect(proxy)
            if boxes.shape[0] > self.max_faces:
                self.excess_faces = max(self.excess_faces, int(boxes.shape[0]) - self.max_faces)
                boxes = boxes[: self.max_faces]
            self.timing["detect"] += time.time() - ts
            ts = time.time()
            lm, _ = self.lmk(proxy, boxes[:, :4]) if boxes.shape[0] else (torch.zeros((0, 68, 2), device=self.dev), None)
            rois = eye_rois_from_landmarks(lm)
            self.timing["landmarks"] += time.time() - ts
            ts = time.time()
            obs = []
            for roi in rois:
                width = float(roi["width"])
                cx, cy = float(roi["center"][0]), float(roi["center"][1])
                ang = float(roi["angle"])
                if width < 3:
                    continue
                cx, cy, diam, pres, vis, crop, probs, e = self._refine(fr.reference, cx, cy, 0.42 * width, ang)
                obs.append({"center": (cx, cy), "width": width, "diam": diam, "angle": ang, "present": pres, "vis": vis,
                            "face_score": float(boxes[roi["face"], 4]), "which": roi["which"], "crop": crop, "probs": probs, "e": e})
            ids = tracker.update(fr.index, obs)
            recs = []
            for o, tid in zip(obs, ids):
                r = EyeRec(fr.index, tracker.shot, tid, o["which"], o["center"][0], o["center"][1], o["diam"], o["angle"],
                           o["present"], o["face_score"], o["vis"])
                recs.append(r)
                if self.mode == "hybrid" and o["present"] > 0.5:
                    self._observe_canonical(r, o, fr.white)
            self.records[fr.index] = recs
            self.timing["guard_analysis"] += time.time() - ts
            n += 1
        self.n_frames = n
        self._smooth_tracks()
        if self.mode == "hybrid":
            for key, st in self.canon.items():
                self.canon_meta[key] = CAN.freeze(st)
        return {"frames": n, "shots": len(set(self.shots.values())), "cuts": self.cuts, "tracks": tracker.next_id,
                "excess_faces": self.excess_faces, "seconds": round(time.time() - t0, 2)}

    def _observe_canonical(self, r: EyeRec, o: dict, white: float) -> None:
        key = f"shot{r.shot}/track{r.track}"
        st = self.canon.get(key) or CAN.CanonicalState.new(key, self.dev)
        self.canon[key] = st
        probs, crop, e = o["probs"], o["crop"], o["e"]
        clipped = clipped_mask(crop[None], white, self.working)[0, 0]
        w = probs[1] * (1 - probs[4]) * (1 - clipped) * (1 - probs[3]) * (probs[0] > 0.3).float()
        S = self.S
        gcx, gcy, gr, _ = iris_geometry(probs[1:2][None], probs[2:3][None], S)
        rp = float(torch.sqrt(probs[2].sum() / math.pi))
        # Round 20: a PUPIL observation must never include the catchlight or its halo - those pixels
        # are exactly what the fill is meant to replace, and averaging them in is what made the first
        # canonical pupil come out lighter than polar_copy's.  probs[4] masks the reflection itself;
        # the halo around it is not masked, so exclude bright pixels by the same highlight_excess
        # test the membrane uses to find the core.
        halo = (highlight_excess(crop, self.working) > float(self.gate.get("membrane_core_excess", 0.10))).float()
        pupil_w = probs[2] * (1 - probs[4]) * (1 - halo) * (1 - clipped)
        CAN.observe(st, crop / e.view(1, 1, 1), w, pupil_w, float(gcx), float(gcy), float(gr), max(rp, 1.0))

    def _smooth_tracks(self) -> None:
        by_track: dict[tuple, list[EyeRec]] = defaultdict(list)
        for recs in self.records.values():
            for r in recs:
                by_track[(r.shot, r.track)].append(r)
        for lst in by_track.values():
            lst.sort(key=lambda r: r.frame)
            for i, r in enumerate(lst):
                nb = [x for x in lst[max(0, i - 1): i + 2] if abs(x.frame - r.frame) <= 1]
                cx = sum(x.cx for x in nb) / len(nb)
                cy = sum(x.cy for x in nb) / len(nb)
                win = sorted(x.diam for x in lst[max(0, i - 3): i + 4])
                d = win[len(win) // 2]
                ang = sum(x.angle for x in nb) / len(nb)
                r.tf = CropTransform(cx, cy, self.S / (K_CROP * max(d, 2.0)), ang, self.S)
                r.diam = d

    # ------------------------------------------------------------------ pass 2
    def _input_crop(self, frame: torch.Tensor, tf: CropTransform) -> torch.Tensor:
        """The render pass's crop; with gate `input_soften_sigma` > 0 the frame region feeding it is
        blurred first (Round 43).  Outside the true frame the crop stays zero, as the plain crop's."""
        sigma = float(self.gate.get("input_soften_sigma", 0.0))
        if sigma <= 0 or tf.scale < 1.0:
            return extract_crop(frame, tf)
        from ..gpu.imgops import extract_crop as _crop, gaussian_blur, valid_crop

        H, W = frame.shape[1:]
        x0, y0, x1, y1 = tf.frame_box(W, H, pad=int(4 * sigma) + 4)
        roi = gaussian_blur(frame[:, y0:y1, x0:x1], sigma)
        out = _crop(roi, CropTransform(tf.cx - x0, tf.cy - y0, tf.scale, tf.theta, tf.size), "bilinear", "border")
        return out * valid_crop(H, W, tf, frame.device).gt(0.999).float()

    def _prepare(self, fr, rec: EyeRec) -> dict:
        crop = self._input_crop(fr.reference, rec.tf)
        probs, pres, e = self._guard(crop)
        rp = self._last_refl_present
        clip = clipped_mask(crop[None], fr.white, self.working)
        # the hole = the guard's reflection mask, dilated: the value-fill region of fill_mode "network_value"
        hole = hole_mask(probs[None], float(self.gate.get("hole_refl_thresh", 0.5)), int(self.gate.get("hole_dilate_px", 2)))[0, 0]
        if self.gate.get("restore_input", "source") == "hole":
            # a masked-inpainting restorer: the hole is blanked in its input and its residual applies to the blanked crop
            inp, base = hole_restore_input(crop[None], e, probs[None], clip, hole[None, None])
            inp, base = inp[0], base[0]
        else:
            inp, base = restore_input(crop[None], e, probs[None], clip)[0], crop
        net = self.b.restore if self.temporal else self.b.restore_single
        with torch.autocast("cuda", dtype=torch.bfloat16):
            feats = net.encode(inp[None]) if self.temporal else None
        out1 = stats = None
        if not self.temporal or self.gate.get("presence_hysteresis", False):
            # single-frame output now: reused by _restore in single mode (no extra cost) and, for the
            # track hysteresis, the frame's own strict eye-level presence statistics
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out1 = self.b.restore_single(inp[None])
            if self.gate.get("hole_source", "guard") == "union" and self.gate.get("restore_input", "source") == "source":
                # the restorer's own support map joins the guard's reflection channel: the hole, the selection, the
                # area test and the temporal union read the union; the restorer's input above is unchanged
                probs = probs.clone()
                probs[4] = torch.maximum(probs[4], torch.sigmoid(out1["support"][0, 0].float()))
                hole = hole_mask(probs[None], float(self.gate.get("hole_refl_thresh", 0.5)), int(self.gate.get("hole_dilate_px", 2)))[0, 0]
            c1 = crop_support(probs, {k: v[0] for k, v in out1.items()}, "conservative", self.gate, refl_present=rp)
            stats = {"open": pres > self.gate["present_thresh"], "strict": bool(c1["reflection_present"]), "area": c1["area"],
                     "mean_corr": c1["mean_corr"]}
        return {"crop": crop, "probs": probs, "present": pres, "e": e, "inp": inp, "feats": feats, "rec": rec, "white": fr.white,
                "refl_present": rp, "out1": out1, "stats": stats, "hole": hole, "base": base}

    def _restore(self, t: int, track: int, cache: dict) -> dict:
        center = cache[(t, track)]
        if not self.temporal:
            if center.get("out1") is not None:
                return center["out1"]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return self.b.restore_single(center["inp"][None])
        shot = center["rec"].shot
        items, valid = [], []
        for f in range(t - 2, t + 3):
            it = cache.get((f, track))
            ok = it is not None and it["rec"].shot == shot and it["present"] > self.gate["present_thresh"]
            items.append(it if ok else center)
            valid.append(bool(ok) or f == t)
        xs = torch.stack([it["inp"] for it in items])[None]
        per = [it["feats"] for it in items]
        v = torch.tensor(valid, device=self.dev)[None]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self.b.restore.forward_video(xs, v, cached=per)

    def _track_relaxed(self, rec: EyeRec, cache: dict | None) -> bool:
        """Track hysteresis: does this eye's track show the reflection in most of its open frames -
        the frames flushed so far plus the cached look-ahead window - with enough history?"""
        g = self.gate
        if not g.get("presence_hysteresis", False):
            return False
        tal = self.track_tally.get((rec.shot, rec.track), {"open": 0, "present": 0})
        n_open, n_pres = tal["open"], tal["present"]
        if cache is not None:
            for f in range(rec.frame + 1, rec.frame + 3):  # look-ahead frames, not yet in the tally
                it = cache.get((f, rec.track))
                if it is None or it["rec"].shot != rec.shot or not it.get("stats"):
                    continue
                if it["stats"]["open"]:
                    n_open += 1
                    n_pres += int(it["stats"]["strict"])
        return n_open >= int(g.get("hysteresis_min_history", 3)) and n_pres / max(n_open, 1) >= float(g.get("hysteresis_min_track_frac", 0.5))

    def _patch(self, fr, rec: EyeRec, prep: dict, out: dict, cache: dict | None = None) -> tuple[EyePatch | None, dict]:
        H, W = fr.reference.shape[1:]
        S = self.S
        crop, e, probs = prep["crop"], prep["e"], prep["probs"]
        src_n = normalize(prep.get("base", crop)[None], e)  # the blanked crop for restore_input "hole"
        pred_n = src_n + out["residual"].float()
        delta = (denormalize(pred_n, e) - crop[None])[0]
        relaxed = self._track_relaxed(rec, cache)
        hole = prep.get("hole")
        win = int(self.gate.get("temporal_refl_window", 0))
        if win > 0 and cache is not None and hole is not None:
            # Round 23: the reflection channel as the union over +-win frames of this track, each
            # warped into this frame's crop.  The selection, the presence area and the value-fill hole
            # all read the union, so a frame where the guard's mask flickers off does not switch the
            # whole fill off with it, and the edited outline follows the track instead of the frame.
            refl = probs[4:5].clone()
            for f in range(rec.frame - win, rec.frame + win + 1):
                it = cache.get((f, rec.track)) if f != rec.frame else None
                if it is None or it["rec"].shot != rec.shot or it["present"] <= self.gate["present_thresh"]:
                    continue
                refl = torch.maximum(refl, warp_crop_to_crop(it["probs"][4:5], it["rec"].tf, rec.tf))
            probs = probs.clone()
            probs[4:5] = refl
            hole = hole_mask(probs[None], float(self.gate.get("hole_refl_thresh", 0.5)), int(self.gate.get("hole_dilate_px", 2)))[0, 0]
        comp = crop_support(probs, {k: v[0] for k, v in out.items()}, self.mode, self.gate, refl_present=prep.get("refl_present"),
                            relaxed=relaxed, hole=(hole > 0.5) if hole is not None else None)
        if hole is not None and comp["reflection_present"]:
            # Round 23: the hole's rows against the lid (see gate.DEFAULT_GATE hole_lid_margin_px / hole_surface_union)
            hb = hole > 0.5
            extra = torch.zeros_like(hb)
            lm = int(self.gate.get("hole_lid_margin_px", 0))
            if lm > 0:
                extra |= hb & dilate(comp["surface"], lm)
            if self.gate.get("hole_surface_union", False) and cache is not None and win > 0:
                for f in range(rec.frame - win, rec.frame + win + 1):
                    it = cache.get((f, rec.track)) if f != rec.frame else None
                    if it is None or it["rec"].shot != rec.shot or it["present"] <= self.gate["present_thresh"]:
                        continue
                    pj = it["probs"]
                    sj = ((pj[0] > self.gate["ocular_thresh"]) & (pj[3] <= self.gate["protected_thresh"])).float()[None]
                    extra |= hb & (warp_crop_to_crop(sj, it["rec"].tf, rec.tf)[0] > 0.5)
            if bool(extra.any()):
                comp["surface"] = comp["surface"] | extra
        if self.gate.get("hole_defines_support", False) and hole is not None and comp["reflection_present"]:
            # Round 23: the hole on the eye surface is selected and recoverable in its own right, so the
            # hard support follows the track's (unioned) reflection mask, not the per-frame confidence.
            hs = (hole > 0.5) & comp["surface"]
            comp["selected"] = comp["selected"] | hs
            comp["recoverable"] = comp["recoverable"] | hs
            comp["editable"] = comp["editable"] | hs
            comp["abstained"] = comp["selected"] & ~comp["editable"]
        comp["hole"] = (hole > 0.5) if hole is not None else None  # read by full_res_gate's hole_skip_margin only
        info = {"reflection_present": comp["reflection_present"],
                "by_track_hysteresis": bool(comp["reflection_present"] and not comp["strict_present"])}
        membrane_native = None
        if self.gate.get("fill_mode", "network") == "network_value" and comp["reflection_present"] and hole is not None:
            # The restorer's own fill reaches the frame as a VALUE inside the hole, not as a crop-space
            # delta.  The delta path composites  ref + warp(pred - crop) , and warp(crop) is a blurred ref,
            # so the source's high-pass - the catchlight's sharp edges - survives every composite: even a
            # hand fix put through that path left the catchlight's edges behind, which is where every
            # learned fill plateaued.  Replacing the value inside the hole removes them.  Outside the hole
            # the delta path stands: the network's edit there is smooth.
            core = (hole > 0.5) & comp["recoverable"]
            if bool(core.any()):
                membrane_native = (denormalize(pred_n, e)[0], core)
                info["fill"] = "network_value"
                info["membrane_core_native_px"] = round(float(core.sum()) / (rec.tf.scale ** 2), 1)
            else:
                info["fill"] = "network"
        if prep.get("stats") is not None:  # the tally only ever learns from STRICT decisions
            tal = self.track_tally.setdefault((rec.shot, rec.track), {"open": 0, "present": 0})
            if prep["stats"]["open"]:
                tal["open"] += 1
                tal["present"] += int(prep["stats"]["strict"])
        canon_digest = None
        if self.mode == "hybrid" and bool(comp["synth"].any()):
            key = f"shot{rec.shot}/track{rec.track}"
            st = self.canon.get(key)
            use = torch.zeros_like(comp["synth"])
            if st is not None and st.frozen:
                gcx, gcy, gr, _ = iris_geometry(probs[1:2][None], probs[2:3][None], S)
                rp = float(torch.sqrt(probs[2].sum() / math.pi))
                img, iris_mask = CAN.render(st, S, float(gcx), float(gcy), float(gr), max(rp, 1.0), float(e))
                use = comp["synth"] & (iris_mask > 0.5)
                delta = torch.where(use[None], img - crop, delta)
                canon_digest = st.digest
            # synthesis only where the canonical iris/pupil applies; other candidates stay abstained
            comp["synth"] = use
            comp["editable"] = (comp["recoverable"] | use) if comp["reflection_present"] else torch.zeros_like(use)
            comp["abstained"] = comp["selected"] & ~comp["editable"]
        info["canonical_digest"] = canon_digest
        exposure_ok = bool(torch.isfinite(e).all()) and float(e) > 1e-5
        dec = eye_validity(prep["present"], rec.diam, self.min_iris, rec.tf is not None, exposure_ok, True, self.gate)
        rec.reasons = list(dec.reasons)
        box = rec.tf.frame_box(W, H)
        if box[2] <= box[0] or box[3] <= box[1]:
            dec.valid = False
            rec.reasons.append("roi_outside_frame")
        if not comp["reflection_present"]:
            rec.reasons.append("no_reflection_detected")
        elif info["by_track_hysteresis"]:
            rec.reasons.append("presence_by_track_hysteresis")
        if info.get("fill") == "network_value":
            rec.reasons.append(f"fill_{info['fill']}_{info['membrane_core_native_px']:g}px")
        from ..gpu.imgops import inverse_warp

        gate = full_res_gate(comp, rec.tf, box, rec.diam, self.gate, self.max_region)
        if not dec.valid:
            gate = {k: torch.zeros_like(v) for k, v in gate.items()}
        delta_full = inverse_warp(delta, rec.tf, box)
        if membrane_native is not None and dec.valid:
            filled_c, core_c = membrane_native
            x0, y0, x1, y1 = box
            # every native pixel the core touches (partial coverage included): at the core's edge the
            # crop-space delta leaves the highlight's own edge half-removed
            core_n = inverse_warp(core_c.float()[None], rec.tf, box)[0] > float(self.gate.get("membrane_native_cover", 0.2))
            filled_n = inverse_warp(filled_c, rec.tf, box)
            cap_q = float(self.gate.get("fill_luma_cap_q", 0.0))
            if cap_q > 0:
                filled_n = cap_fill_luma(filled_n, fr.reference[:, y0:y1, x0:x1], core_n, rec.diam, cap_q,
                                         float(self.gate.get("fill_luma_cap_ring", 0.1)), self.working)
            delta_full = torch.where(core_n[None], filled_n - fr.reference[:, y0:y1, x0:x1], delta_full)
            # a value fill needs no ramp: inside H the core takes the fill completely (the feather ramp at
            # the eroded lid boundary otherwise leaves the highlight's top row at ~60% of the fill)
            gate["feather"] = torch.where(core_n & gate["hard"], torch.ones_like(gate["feather"]), gate["feather"])
        patch = EyePatch(box=box, delta=delta_full, hard=gate["hard"], feather=gate["feather"], synth=gate["synth"],
                         uncertainty=gate["uncertainty"], track_id=rec.track, eye=rec.which,
                         meta={"selected": gate["selected"], "abstained": gate["abstained"], **info})
        return patch, info

    @torch.no_grad()
    def render(self, source, on_frame) -> dict:
        """Second pass. ``on_frame(fr, result, patches, recs)`` is called in display order."""
        t0 = time.time()
        ring: "OrderedDict[int, object]" = OrderedDict()
        cache: dict = {}
        stats = {"frames": 0, "outside_changed_bits": 0, "assigned_px": 0, "bypass": defaultdict(int)}

        def flush(t: int) -> None:
            fr = ring[t]
            patches = []
            ts = time.time()
            for rec in self.records.get(t, []):
                if (t, rec.track) not in cache:
                    continue
                out = self._restore(t, rec.track, cache)
                p, _ = self._patch(fr, rec, cache[(t, rec.track)], out, cache)
                if p is not None:
                    patches.append(p)
            self.timing["restore_gate"] += time.time() - ts
            ts = time.time()
            res = composite(fr.reference, patches, self.strength, fr.alpha)
            sup = res.support
            bad = changed_bits_outside(fr.reference, res.output, sup)
            if bad:
                raise RuntimeError(f"preservation violated on frame {t}: {bad} pixels changed outside the support")
            if fr.alpha is not None:
                stats["alpha_frames"] = stats.get("alpha_frames", 0) + 1
                a0 = fr.alpha.contiguous().view(torch.int16 if fr.alpha.element_size() == 2 else torch.int32)
                a1 = res.alpha.contiguous().view(torch.int16 if res.alpha.element_size() == 2 else torch.int32)
                if not bool((a0 == a1).all()):
                    raise RuntimeError(f"source alpha changed on frame {t}")
            self.timing["composite"] += time.time() - ts
            stats["frames"] += 1
            if stats["frames"] % 6 == 0:  # windowed peak allocator memory (bounded-memory evidence)
                torch.cuda.synchronize(self.dev)
                stats.setdefault("mem_samples", []).append([t, round(torch.cuda.memory_allocated(self.dev) / 2**30, 4),
                                                            round(torch.cuda.max_memory_allocated(self.dev) / 2**30, 4)])
                torch.cuda.reset_peak_memory_stats(self.dev)
            stats["assigned_px"] += res.assigned_pixels
            stats["bypass"][res.bypass or "none"] += 1
            ts = time.time()
            on_frame(fr, res, patches, self.records.get(t, []))
            self.timing["export"] += time.time() - ts
            for k in [k for k in cache if k[0] <= t - 2]:
                del cache[k]
            del ring[t]

        pending = []
        for fr in source:
            ring[fr.index] = fr
            ts = time.time()
            for rec in self.records.get(fr.index, []):
                cache[(fr.index, rec.track)] = self._prepare(fr, rec)
            self.timing["prepare"] += time.time() - ts
            pending.append(fr.index)
            while pending and (pending[0] + 2 <= fr.index):
                flush(pending.pop(0))
            if len(ring) > self.buffer_frames:
                raise RuntimeError("ring buffer overflow")
        while pending:
            flush(pending.pop(0))
        stats["seconds"] = round(time.time() - t0, 2)
        stats["bypass"] = dict(stats["bypass"])
        return stats

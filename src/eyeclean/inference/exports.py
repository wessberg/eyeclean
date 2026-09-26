"""Output writers for `eyeclean process`.

Authoritative layers (uncompressed FLOAT EXR, cropped dataWindow inside the full-frame
displayWindow, working-space values):
  patches/f<frame>_t<track>.exr              R,G,B = signed correction out - reference (0 outside support)
  patches/f<frame>_t<track>_replacement.exr  R,G,B,A straight alpha: A = hard support, RGB = output where A=1
  masks/f<frame>_t<track>.exr                hard, feather, synth, selected, abstained
  uncertainty/f<frame>_t<track>.exr          uncertainty (1 - raw reconstruction confidence; not a probability)
  master/frame_<frame>.exr                   optional full-frame output (+ original alpha)
  <name>.eyeclean.<jpg|png>                  still-image input: the repaired photo in the input's format and colour
                                             (original code values outside the support; see io/stills.py)
Viewing previews (NOT pixel-preservation masters): preview.mp4, comparison.mp4 (NVENC H.264).

patches/index.json (eyeclean.process/1) lists EVERY tracked eye of every frame with a ``status``:
  repaired   the final hard support is non-empty: the eye was edited (files listed)
  untouched  the eye was tracked and left bit-exact (``reasons`` says why, e.g. no_reflection_detected)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from ..color import transforms as CT
from ..io.exr import NO_COMPRESSION, ZIP_COMPRESSION, ZIPS_COMPRESSION, write_exr
from ..version import PROCESS_REPORT_SCHEMA

_COMPRESSION = {"none": NO_COMPRESSION, "zips": ZIPS_COMPRESSION, "zip": ZIP_COMPRESSION}
from ..io.stills import output_suffix, write_still
from ..io.video import VideoWriter
from .compositor import CompositeResult, EyePatch, signed_correction


def tight_box(mask: torch.Tensor) -> tuple[int, int, int, int] | None:
    ys = torch.nonzero(mask.any(1)).flatten()
    xs = torch.nonzero(mask.any(0)).flatten()
    if ys.numel() == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


class Exporter:
    def __init__(self, out: Path, W: int, H: int, fps: float, exports: set[str], working_color: str,
                 time_base=(1, 90000), master: bool = False, preview_codec: str = "h264", exr_compression: str = "none",
                 still=None):
        self.out, self.W, self.H, self.exports, self.working = out, W, H, exports, working_color
        self.still, self.photo = still, None
        self.cmp = _COMPRESSION[exr_compression]
        self.index: list[dict] = []
        self.master = master or "master" in exports
        for d in ("patches", "masks", "uncertainty"):
            (out / d).mkdir(parents=True, exist_ok=True)
        if self.master:
            (out / "master").mkdir(parents=True, exist_ok=True)
        self.preview = self.comparison = None
        self.preview_exposure = None
        if "preview" in exports:
            self.preview = VideoWriter(str(out / "preview.mp4"), W, H, fps, preview_codec, 8, time_base)
            s = min(1.0, 1280 / W, 720 / H)
            self.pw, self.ph = int(W * s) // 2 * 2, int(H * s) // 2 * 2
            self.comparison = VideoWriter(str(out / "comparison.mp4"), self.pw * 3, self.ph, fps, preview_codec, 8, time_base)
        self.files = []

    def display(self, x: torch.Tensor) -> torch.Tensor:
        l709 = CT.working_to_linear709(x, self.working)
        if self.preview_exposure is None:
            lum = 0.2126 * l709[0] + 0.7152 * l709[1] + 0.0722 * l709[2]
            q = torch.quantile(lum.flatten()[:: max(1, lum.numel() // 262144)], torch.tensor([0.5, 0.97], device=x.device))
            self.preview_exposure = float(torch.minimum(0.18 / q[0].clamp(min=1e-4), 0.85 / q[1].clamp(min=1e-4)))
        return CT.bt709_oetf(torch.clamp(l709 * self.preview_exposure, 0, 1))

    def on_frame(self, fr, res: CompositeResult, patches: list[EyePatch], recs) -> None:
        entry = {"frame": fr.index, "time_s": fr.time_s, "pts": fr.pts, "source": fr.source_name, "bypass": res.bypass,
                 "assigned_px": res.assigned_pixels, "eyes": []}
        rec_by_track = {r.track: r for r in recs}
        cmp = self.cmp
        for p in patches:
            r = rec_by_track.get(p.track_id)
            e = {"track": p.track_id, "eye": p.eye, "roi": list(p.box), "reasons": list(r.reasons) if r else [],
                 "support_px": int(p.hard.sum()), "synth_px": int(p.synth.sum()) if p.synth is not None else 0,
                 "selected_px": int(p.meta["selected"].sum()), "abstained_px": int(p.meta["abstained"].sum()),
                 "canonical_digest": p.meta.get("canonical_digest")}
            x0, y0, x1, y1 = p.box
            sup = res.support[y0:y1, x0:x1] & p.hard
            tb = tight_box(sup | p.meta["abstained"])
            if tb is not None and "patches" in self.exports:
                bx0, by0, bx1, by1 = tb
                gbox = (x0 + bx0, y0 + by0, x0 + bx1, y0 + by1)
                corr = signed_correction(fr.reference, res.output, res.support, gbox)
                stem = f"f{fr.index:06d}_t{p.track_id:03d}"
                attrs = {"eyeclean/frame": str(fr.index), "eyeclean/time_s": f"{fr.time_s:.9f}", "eyeclean/working_color": self.working,
                         "eyeclean/semantics": "signed correction: out - reference, working space, zero outside hard support"}
                sl = (slice(by0, by1), slice(bx0, bx1))
                write_exr(self.out / "patches" / f"{stem}.exr", {"R": corr[0], "G": corr[1], "B": corr[2]}, display_size=(self.W, self.H),
                          origin=(gbox[0], gbox[1]), attributes=attrs, compression=cmp)
                outv = res.output[:, gbox[1]:gbox[3], gbox[0]:gbox[2]]
                a = res.support[gbox[1]:gbox[3], gbox[0]:gbox[2]].float()
                write_exr(self.out / "patches" / f"{stem}_replacement.exr", {"R": outv[0] * a, "G": outv[1] * a, "B": outv[2] * a, "A": a},
                          display_size=(self.W, self.H), origin=(gbox[0], gbox[1]),
                          attributes={"eyeclean/semantics": "replacement patch, STRAIGHT (unassociated) alpha; A=hard support"},
                          compression=cmp)
                write_exr(self.out / "masks" / f"{stem}.exr",
                          {"hard": sup[sl].float(), "feather": p.feather[sl] * sup[sl].float(), "synth": p.synth[sl].float(),
                           "selected": p.meta["selected"][sl].float(), "abstained": p.meta["abstained"][sl].float()},
                          display_size=(self.W, self.H), origin=(gbox[0], gbox[1]), compression=cmp)
                write_exr(self.out / "uncertainty" / f"{stem}.exr", {"uncertainty": p.uncertainty[sl] * sup[sl].float()},
                          display_size=(self.W, self.H), origin=(gbox[0], gbox[1]),
                          attributes={"eyeclean/semantics": "1 - raw network reconstruction confidence (see MODEL_CARD calibration)"},
                          compression=cmp)
                e.update({"patch": f"patches/{stem}.exr", "replacement": f"patches/{stem}_replacement.exr", "mask": f"masks/{stem}.exr",
                          "uncertainty": f"uncertainty/{stem}.exr", "data_window": list(gbox)})
            e["status"] = "repaired" if e["support_px"] > 0 else "untouched"
            entry["eyes"].append(e)
        patched = {p.track_id for p in patches}
        for r in recs:                      # tracked eyes that produced no patch at all: listed, untouched
            if r.track not in patched:
                entry["eyes"].append({"track": r.track, "eye": r.which, "status": "untouched", "reasons": list(r.reasons) or ["not_processed"],
                                      "support_px": 0, "synth_px": 0, "selected_px": 0, "abstained_px": 0})
        if self.master:
            ch = {"R": res.output[0], "G": res.output[1], "B": res.output[2]}
            if res.alpha is not None:
                ch["A"] = res.alpha.float() if res.alpha.dtype != torch.float32 else res.alpha
            write_exr(self.out / "master" / f"frame_{fr.index:06d}.exr", ch, attributes={"eyeclean/semantics": "full-frame output master"},
                      compression=self.cmp)
        if self.still is not None and "photo" in self.exports:
            name = f"{Path(self.still.path).stem}.eyeclean{output_suffix(self.still)}"
            self.photo = write_still(self.out / name, CT.working_to_linear709(res.output, self.working), res.support, fr.codes,
                                     self.still, alpha=fr.alpha)
            self.photo["path"] = name
        if self.preview is not None:
            disp_out = self.display(res.output)
            self.preview.write(disp_out, fr.pts)
            disp_in = self.display(fr.reference)
            ov = disp_out.clone()
            sup = res.support.float()
            ov = ov * (1 - 0.5 * sup) + torch.tensor([0.0, 1.0, 0.3], device=ov.device).view(3, 1, 1) * 0.5 * sup
            tiles = [F.interpolate(t[None], size=(self.ph, self.pw), mode="bilinear", antialias=True, align_corners=False)[0]
                     for t in (disp_in, disp_out, ov)]
            self.comparison.write(torch.cat(tiles, 2), fr.pts)
        self.index.append(entry)

    def abort(self) -> None:
        """Close the preview writers of an incomplete run (no index is written)."""
        for w in (self.preview, self.comparison):
            if w is not None:
                try:
                    w.close()
                except Exception:  # noqa: BLE001
                    pass
        self.preview = self.comparison = None

    def close(self) -> dict:
        res = {}
        if self.preview is not None:
            res["preview"] = self.preview.close()
            res["comparison"] = self.comparison.close()
        (self.out / "patches" / "index.json").write_text(json.dumps({"schema": PROCESS_REPORT_SCHEMA, "frames": self.index, "width": self.W,
                                                                     "height": self.H, "working_color": self.working}, indent=1),
                                                         encoding="utf-8")
        return res

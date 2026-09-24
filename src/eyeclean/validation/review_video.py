"""Eye-focused review videos for processed footage (`eyeclean review-video`).

Real footage has no clean ground truth, so these videos are for human review, not measurement:

* ``split``   source | EyeClean output side by side, cropped around the tracked eyes (temporally
  smoothed framing), in real time and slowed down.
* ``grid``    optional comparison against an external eye-crop comparison video with a regular
  layout (rows of nearest-neighbour-magnified eye panels).  The external video's SOURCE row is
  matched exactly against the plate frame by frame to recover its crop windows, so EyeClean's
  output is shown with identical framing next to the external method's row.

Output pixels are shown in the input's own display encoding: pixels outside the EyeClean support
are the decoded source values themselves, pixels inside are the corrected working-space values
re-encoded with the inverse of the recorded input transform.  All image work runs on CUDA; the
source files are only read (NVDEC).  Everything is written under the given output directory.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from ..color import transforms as CT
from ..io.video import VideoReader, VideoWriter, planes_to_rgb
from .evaluate import reconstruct

# ---- 5x7 bitmap font (labels) --------------------------------------------------------------
_GLYPHS = {
    "A": ".###." "#...#" "#...#" "#####" "#...#" "#...#" "#...#","B": "####.#...##...#####.#...##...#####.", "C": ".###.#...##....#....#....#...#.###.",
    "D": "####.#...##...##...##...##...#####.", "E": "######....#....####.#....#....#####", "F": "######....#....####.#....#....#....",
    "G": ".###.#...##....#.####...##...#.###.", "H": "#...##...##...#######...##...##...#", "I": ".###...#....#....#....#....#...###.",
    "J": "..###...#....#....#....#.#..#..##..", "K": "#...##..#.#.#..##...#.#..#..#.#...#", "L": "#....#....#....#....#....#....#####",
    "M": "#...###.###.#.##.#.##...##...##...#", "N": "#...##...###..##.#.##..###...##...#", "O": ".###.#...##...##...##...##...#.###.",
    "P": "####.#...##...#####.#....#....#....", "Q": ".###.#...##...##...##.#.##..#..##.#", "R": "####.#...##...#####.#.#..#..#.#...#",
    "S": ".#####....#.....###.....#....#####.", "T": "#####..#....#....#....#....#....#..", "U": "#...##...##...##...##...##...#.###.",
    "V": "#...##...##...##...##...#.#.#...#..", "W": "#...##...##...##.#.##.#.##.#.#.#.#.", "X": "#...##...#.#.#...#...#.#.#...##...#",
    "Y": "#...##...#.#.#...#....#....#....#..", "Z": "#####....#...#...#...#...#....#####", "0": ".###.#...##..###.#.###..##...#.###.",
    "1": "..#...##....#....#....#....#...###.", "2": ".###.#...#....#...#...#...#...#####", "3": "#####" "...#." "..#.." "...#." "....#" "#...#" ".###.",
    "4": "...#...##..#.#.#..#.#####...#....#.", "5": "######....####.....#....##...#.###.", "6": "..##..#...#....####.#...##...#.###.",
    "7": "#####....#...#...#...#....#....#...", "8": ".###.#...##...#.###.#...##...#.###.", "9": ".###.#...##...#.####....#...#..##..",
    " ": "." * 35, "-": "...............#####...............", ".": "..........................##...##..",
    ":": ".....##...##..........##...##......", "(": "...#...#...#....#....#.....#.....#.", ")": ".#.....#.....#....#....#...#...#...",
    "/": ".........#...#...#...#...#.........", "|": "..#....#....#....#....#....#....#..", "+": ".......#....#..#####..#....#.......",
    "=": "..........#####.....#####..........", "%": "##...##..#...#...#...#...#..##...##", ",": "....................##....#...#....",
    "_": "..............................#####", "'": "..#....#...#........................"[:35],
}
_ATLAS: dict = {}


def _glyph(ch: str, dev) -> torch.Tensor:
    key = (ch, str(dev))
    if key not in _ATLAS:
        bits = _GLYPHS.get(ch, _GLYPHS[" "])
        _ATLAS[key] = torch.tensor([1.0 if c == "#" else 0.0 for c in bits], device=dev).view(7, 5)
    return _ATLAS[key]


def draw_text(img: torch.Tensor, text: str, x: int, y: int, scale: int = 2, color=(1.0, 1.0, 1.0)) -> None:
    """In-place: draw uppercase 5x7 text into img (3,H,W) at (x, y) (top-left)."""
    dev = img.device
    col = torch.tensor(color, device=dev).view(3, 1, 1)
    for i, ch in enumerate(text.upper()):
        g = _glyph(ch, dev).repeat_interleave(scale, 0).repeat_interleave(scale, 1)
        gx, gy = x + i * 6 * scale, y
        h, w = g.shape
        if gx + w > img.shape[2] or gy + h > img.shape[1]:
            break
        reg = img[:, gy:gy + h, gx:gx + w]
        img[:, gy:gy + h, gx:gx + w] = reg * (1 - g) + col * g


# ---- decoding / output reconstruction ------------------------------------------------------
_ENCODE = {"rec709": CT.bt709_oetf, "rec709-display": CT.bt1886_inverse_eotf, "srgb": CT.srgb_inverse_eotf}


def load_process(proc: Path) -> dict:
    proc = Path(proc)
    man = json.loads((proc / "source_manifest.json").read_text(encoding="utf-8"))
    index = {e["frame"]: e for e in json.loads((proc / "patches" / "index.json").read_text(encoding="utf-8"))["frames"]}
    tracks = json.loads((proc / "tracks.json").read_text(encoding="utf-8"))
    rep = json.loads((proc / "RUN_REPORT.json").read_text(encoding="utf-8"))
    return {"dir": proc, "manifest": man, "index": index, "tracks": tracks, "interp": man["interpretation"], "mode": rep.get("mode"),
            "allow_synthesis": rep.get("allow_synthesis")}


def output_display(rgb_enc: torch.Tensor, ref: torch.Tensor, proc: dict, k: int) -> tuple[torch.Tensor, dict]:
    """Display-encoded output of frame k: source code values outside the support, re-encoded
    corrected values inside it (same display transform as the input)."""
    r = reconstruct(proc["dir"], proc["index"], k, ref)
    ic, wc = proc["interp"]["input_color"], proc["interp"]["working_color"]
    enc = _ENCODE.get(ic)
    if enc is None:
        raise ValueError(f"review videos show display-encoded inputs only (got {ic}); use the process previews for linear/log inputs")
    sup = r["support"]
    if not bool(sup.any()):
        return rgb_enc, r
    out_enc = enc(CT.working_to_linear709(r["pred"], wc))
    return torch.where(sup[None], out_enc, rgb_enc), r


def frames(video: str, proc: dict, dev, max_frames: int | None = None):
    """Yield (k, pts, decoded R'G'B', working reference) with the process run's recorded interpretation."""
    it = proc["interp"]
    full = it["input_range"] == "full"
    for k, d in enumerate(VideoReader(video, dev)):
        if max_frames is not None and k >= max_frames:
            break
        rgb = planes_to_rgb(d, full, it["ycbcr_matrix"])
        yield k, d.pts, rgb, CT.to_working(rgb, it["input_color"], it["working_color"]).contiguous()


def _sample_box(img: torch.Tensor, cx: float, cy: float, w: float, h: float, Wo: int, Ho: int, mode: str = "bicubic") -> torch.Tensor:
    """Resample the frame box centred at (cx, cy) with size (w, h) (pixel-centre coordinates) to (3, Ho, Wo)."""
    C, H, W = img.shape
    xs = cx + (torch.arange(Wo, device=img.device, dtype=torch.float32) + 0.5) / Wo * w - w / 2
    ys = cy + (torch.arange(Ho, device=img.device, dtype=torch.float32) + 0.5) / Ho * h - h / 2
    gx = (xs + 0.5) / W * 2 - 1
    gy = (ys + 0.5) / H * 2 - 1
    grid = torch.stack(torch.meshgrid(gy, gx, indexing="ij")[::-1], -1)[None]
    return F.grid_sample(img[None], grid, mode=mode, padding_mode="border", align_corners=False)[0].clamp(0, 1)


def _smooth(v: list[float | None], radius: int) -> list[float]:
    """Fill gaps by holding neighbours, then centred moving average."""
    n = len(v)
    filled, last = [], None
    for x in v:
        last = x if x is not None else last
        filled.append(last)
    nxt = None
    for i in range(n - 1, -1, -1):  # leading gaps take the first known value
        if filled[i] is None:
            filled[i] = nxt
        else:
            nxt = filled[i]
    filled = [0.0 if x is None else x for x in filled]
    return [sum(filled[max(0, i - radius):i + radius + 1]) / len(filled[max(0, i - radius):i + radius + 1]) for i in range(n)]


def _eye_centres(proc: dict, n: int) -> tuple[list, list, float]:
    fr = {f["frame"]: f for f in proc["tracks"]["frames"]}
    cx, cy, dist = [], [], []
    for k in range(n):
        eyes = fr.get(k, {}).get("eyes", [])
        if eyes:
            cx.append(sum(e["cx"] for e in eyes) / len(eyes))
            cy.append(sum(e["cy"] for e in eyes) / len(eyes))
            if len(eyes) >= 2:
                dist.append(math.hypot(eyes[0]["cx"] - eyes[1]["cx"], eyes[0]["cy"] - eyes[1]["cy"]))
        else:
            cx.append(None)
            cy.append(None)
    d = sorted(dist)[len(dist) // 2] if dist else 120.0
    return cx, cy, d


# ---- split-screen --------------------------------------------------------------------------
def split_video(video: str, proc_dir: Path, out_dir: Path, dev, side_w: int = 960, side_h: int = 540, span: float = 2.0,
                smooth_s: float = 0.35, slow: int = 4, max_frames: int | None = None, label_out: str | None = None) -> dict:
    proc = load_process(proc_dir)
    n = proc["manifest"]["frame_count"]
    if max_frames:
        n = min(n, max_frames)
    fps = len(proc["manifest"]["timestamps_s"]) / max(1e-6, proc["manifest"]["timestamps_s"][-1] - proc["manifest"]["timestamps_s"][0]) \
        if len(proc["manifest"]["timestamps_s"]) > 1 else 24.0
    cx, cy, d = _eye_centres(proc, n)
    rad = max(1, int(round(smooth_s * fps)))
    cx, cy = _smooth(cx, rad), _smooth(cy, rad)
    w = span * d
    h = w * side_h / side_w
    bar = 40
    W, H = side_w * 2 + 8, side_h + bar
    tb = tuple(proc["manifest"].get("time_base") or (1, 90000))
    out_dir.mkdir(parents=True, exist_ok=True)
    rt = VideoWriter(str(out_dir / "eyes_split.mp4"), W, H, fps, "h264", 8, (1, 90000), qp=16)
    sl = VideoWriter(str(out_dir / f"eyes_split_{slow}x_slow.mp4"), W, H, fps / slow, "h264", 8, (1, 90000), qp=16)
    lab = label_out or ("EYECLEAN " + (proc["mode"] or "").upper() + (" (SYNTHESIS ALLOWED)" if proc["allow_synthesis"] else ""))
    ts = proc["manifest"]["timestamps_s"]
    edited = 0
    for k, pts, rgb, ref in frames(video, proc, dev, n):
        out, r = output_display(rgb, ref, proc, k)
        edited += int(r["support"].any())
        a = _sample_box(rgb.clamp(0, 1), cx[k], cy[k], w, h, side_w, side_h)
        b = _sample_box(out.clamp(0, 1), cx[k], cy[k], w, h, side_w, side_h)
        canvas = torch.zeros((3, H, W), device=dev)
        canvas[:, bar:, :side_w] = a
        canvas[:, bar:, side_w + 8:] = b
        draw_text(canvas, "SOURCE", 12, 12, 2)
        draw_text(canvas, lab, side_w + 20, 12, 2)
        draw_text(canvas, f"FRAME {k}", side_w - 12 * 11, 12, 2, (0.7, 0.7, 0.7))
        t90 = int(round((ts[k] - ts[0]) * 90000))
        rt.write(canvas, t90)
        sl.write(canvas, t90 * slow)
    rt.close()
    sl.close()
    return {"eyes_split": str(out_dir / "eyes_split.mp4"), "eyes_split_slow": str(out_dir / f"eyes_split_{slow}x_slow.mp4"),
            "frames": n, "frames_with_edits": edited, "crop_box_px": [round(w, 1), round(h, 1)], "eye_distance_px": round(d, 1)}


# ---- comparison against an external eye-crop video ------------------------------------------
def match_reference_crops(video: str, reference: str, dev, crop: int = 48, block: int = 10, label_px: int = 18, panels: int = 2,
                          source_row: int = 0, search: tuple[int, int, int, int] | None = None, max_frames: int | None = None) -> list[dict]:
    """Per-frame top-left of each ``crop``x``crop`` panel in the external video's SOURCE row, found
    by exact template matching (block centres of the nearest-neighbour magnification) against the
    plate.  Frames whose best match is poor are marked ``matched: False``."""
    hid = math.ceil(label_px / block)
    rs = crop * block
    out, prev = [], [None] * panels
    P, R = VideoReader(video, dev), VideoReader(reference, dev)
    for k, (dp, dr) in enumerate(zip(P, R)):
        if max_frames is not None and k >= max_frames:
            break
        plate = planes_to_rgb(dp, False, "bt709")
        ref = planes_to_rgb(dr, False, "bt709")
        H, W = plate.shape[1:]
        rec = {"frame": k, "eyes": []}
        for e in range(panels):
            panel = ref[:, source_row * rs:(source_row + 1) * rs, e * rs:(e + 1) * rs]
            tmpl = panel[:, block // 2::block, block // 2::block][:, hid:, :]
            th, tw = tmpl.shape[1:]

            def find(x0, x1, y0, y1):
                x0, y0 = max(0, x0), max(0, y0)
                x1, y1 = min(W - crop, x1), min(H - crop, y1)
                reg = plate[:, y0 + hid:y1 + crop, x0:x1 + crop]
                r2 = F.avg_pool2d((reg ** 2).sum(0, keepdim=True)[None], (th, tw), stride=1)[0, 0] * th * tw
                ssd = r2 - 2 * F.conv2d(reg[None], tmpl[None])[0, 0] + (tmpl ** 2).sum()
                i = int(torch.argmin(ssd))
                yy, xx = divmod(i, ssd.shape[1])
                return x0 + xx, y0 + yy, math.sqrt(max(0.0, float(ssd[yy, xx])) / (3 * th * tw))

            wide = search or (0, W - crop, 0, H - crop)
            if prev[e] is None:
                bx, by, rm = find(*wide)
            else:
                px, py = prev[e]
                bx, by, rm = find(px - 12, px + 13, py - 12, py + 13)
                if rm > 0.03:
                    b2 = find(*wide)
                    if b2[2] < rm:
                        bx, by, rm = b2
            prev[e] = (bx, by)
            rec["eyes"].append({"x": bx, "y": by, "rmse": rm, "matched": rm <= 0.03})
        out.append(rec)
    # unmatched windows: interpolate between the nearest matched frames (marked for the viewer)
    for e in range(panels):
        good = [i for i, r in enumerate(out) if r["eyes"][e]["matched"]]
        for i, r in enumerate(out):
            if r["eyes"][e]["matched"] or not good:
                continue
            lo = max([g for g in good if g < i], default=None)
            hi = min([g for g in good if g > i], default=None)
            a, b = (lo if lo is not None else hi), (hi if hi is not None else lo)
            t = 0.0 if a == b else (i - a) / (b - a)
            r["eyes"][e]["x"] = int(round(out[a]["eyes"][e]["x"] * (1 - t) + out[b]["eyes"][e]["x"] * t))
            r["eyes"][e]["y"] = int(round(out[a]["eyes"][e]["y"] * (1 - t) + out[b]["eyes"][e]["y"] * t))
            r["eyes"][e]["interpolated"] = True
    return out


def _nearest_panel(img: torch.Tensor, x: int, y: int, crop: int, block: int) -> torch.Tensor:
    c = img[:, y:y + crop, x:x + crop].clamp(0, 1)
    return c.repeat_interleave(block, 1).repeat_interleave(block, 2)


def grid_video(video: str, reference: str, procs: dict[str, Path], crops: list[dict], out_path: Path, dev, crop: int = 48, block: int = 10,
               label_px: int = 18, reference_rows: dict[str, int] | None = None, fps: float = 12.0, overlay_of: str | None = None) -> dict:
    """Rows: SOURCE (plate), the external video's rows (copied), then one row per EyeClean run, and
    optionally an overlay row (green = recoverable edit, magenta = synthesized) for ``overlay_of``."""
    reference_rows = reference_rows or {}
    rs = crop * block
    panels = len(crops[0]["eyes"])
    P = {name: load_process(p) for name, p in procs.items()}
    names = list(P)
    n_rows = 1 + len(reference_rows) + len(names) + (1 if overlay_of else 0)
    W, H = rs * panels, rs * n_rows
    wr = VideoWriter(str(out_path), W, H, fps, "h264", 8, (1, 90000), qp=16)
    first = P[names[0]]
    Rd = VideoReader(reference, dev)
    stats = {name: {"changed_px": 0, "frames": 0} for name in names}
    for (k, pts, rgb, ref), dr in zip(frames(video, first, dev, len(crops)), Rd):
        refimg = planes_to_rgb(dr, False, "bt709")
        canvas = torch.zeros((3, H, W), device=dev)
        rows = [("SOURCE", rgb, None)]
        outs = {}
        for name in names:
            o, r = output_display(rgb, ref, P[name], k)
            outs[name] = (o, r)
            stats[name]["changed_px"] += int(r["support"].sum())
            stats[name]["frames"] += 1
        interp_any = any(e.get("interpolated") for e in crops[k]["eyes"])
        row = 0
        for label, img, _ in rows:
            for e, c in enumerate(crops[k]["eyes"]):
                canvas[:, row * rs:(row + 1) * rs, e * rs:(e + 1) * rs] = _nearest_panel(img, c["x"], c["y"], crop, block)
            canvas[:, row * rs:row * rs + label_px, :] = 0
            draw_text(canvas, f"{label}   FRAME {k}" + ("   CROP INTERPOLATED" if interp_any else ""), 4, row * rs + 2, 2)
            row += 1
        for label, ri in reference_rows.items():
            canvas[:, row * rs:(row + 1) * rs, :] = refimg[:, ri * rs:(ri + 1) * rs, :W]
            row += 1
        for name in names:
            o, r = outs[name]
            for e, c in enumerate(crops[k]["eyes"]):
                canvas[:, row * rs:(row + 1) * rs, e * rs:(e + 1) * rs] = _nearest_panel(o, c["x"], c["y"], crop, block)
            canvas[:, row * rs:row * rs + label_px, :] = 0
            draw_text(canvas, f"{name}   FRAME {k}", 4, row * rs + 2, 2, (0.55, 1.0, 0.6))
            row += 1
        if overlay_of:
            o, r = outs[overlay_of]
            ov = o.clone()
            for m, col in ((r["support"] & ~r["synth"], (0.1, 1.0, 0.2)), (r["synth"], (1.0, 0.1, 0.9)), (r["abstained"], (1.0, 0.9, 0.1))):
                mm = m.float()[None]
                ov = ov * (1 - 0.6 * mm) + torch.tensor(col, device=dev).view(3, 1, 1) * 0.6 * mm
            for e, c in enumerate(crops[k]["eyes"]):
                canvas[:, row * rs:(row + 1) * rs, e * rs:(e + 1) * rs] = _nearest_panel(ov, c["x"], c["y"], crop, block)
            canvas[:, row * rs:row * rs + label_px, :] = 0
            draw_text(canvas, f"{overlay_of} MASKS: GREEN EDIT  MAGENTA SYNTH  YELLOW ABSTAIN", 4, row * rs + 2, 2, (1.0, 0.9, 0.4))
        wr.write(canvas, int(round(k / fps * 90000)))
    wr.close()
    return {"path": str(out_path), "frames": len(crops), "rows": ["SOURCE", *reference_rows, *names] + ([f"{overlay_of} masks"] if overlay_of else []),
            "changed_px_total": {k: v["changed_px"] for k, v in stats.items()}}


def review(video: str, procs: dict[str, Path], out_dir: Path, reference: str | None = None, reference_rows: dict[str, int] | None = None,
           crop: int = 48, block: int = 10, label_px: int = 18, source_row: int = 0, reference_fps: float = 12.0, slow: int = 4,
           max_frames: int | None = None, log=print) -> dict:
    """All review outputs for one input video: a split screen per process run and, with an external
    reference video, the matched-crop comparison grid plus no-reference crop descriptors."""
    import os

    from ..gpu.runtime import require_cuda
    from ..inference.process import PRIVATE_DIRS, ROOT

    dev = require_cuda("strict")
    out_dir = Path(out_dir).resolve()
    for private in PRIVATE_DIRS:
        if os.path.normcase(str(out_dir)).startswith(os.path.normcase(str((ROOT / private).resolve()))):
            raise ValueError(f"refusing to write under {ROOT / private} (private footage folder)")
    out_dir.mkdir(parents=True, exist_ok=True)
    res: dict = {"input": video, "process_runs": {k: str(v) for k, v in procs.items()}, "split": {}}
    for name, p in procs.items():
        slug = "".join(c if c.isalnum() else "_" for c in name.lower()).strip("_")
        log(f"[review] split screen for {name}")
        res["split"][name] = split_video(video, p, out_dir / slug, dev, slow=slow, max_frames=max_frames, label_out=name)
    if reference:
        log("[review] matching the reference video's SOURCE-row crops against the input")
        crops = match_reference_crops(video, reference, dev, crop, block, label_px, source_row=source_row, max_frames=max_frames)
        (out_dir / "reference_crops.json").write_text(json.dumps({"reference": reference, "crop": crop, "block": block, "frames": crops}),
                                                      encoding="utf-8")
        n_unm = sum(1 for r in crops for e in r["eyes"] if not e["matched"])
        log("[review] comparison grid")
        overlay = next((n for n in procs if "HYBRID" in n.upper()), next(iter(procs)))
        res["grid"] = grid_video(video, reference, procs, crops, out_dir / "compare_reference.mp4", dev, crop, block, label_px,
                                 reference_rows, reference_fps, overlay_of=overlay)
        res["grid"]["unmatched_eye_crops"] = n_unm
        res["grid"]["eye_crops"] = sum(len(r["eyes"]) for r in crops)
        log("[review] crop descriptors")
        res["crop_descriptors"] = crop_metrics(video, reference, procs, crops, dev, crop, block, label_px, reference_rows, source_row)
    (out_dir / "review.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


def crop_metrics(video: str, reference: str, procs: dict[str, Path], crops: list[dict], dev, crop: int = 48, block: int = 10,
                 label_px: int = 18, reference_rows: dict[str, int] | None = None, source_row: int = 0) -> dict:
    """No-reference descriptors on the matched eye crops (there is no clean ground truth):
    changed area and mean change vs the source, near-white highlight pixels remaining, and added
    temporal change (|frame-to-frame difference| beyond the source's own, over the crop).  The
    external rows are compared with the external SOURCE row decoded from the same video (both carry
    that video's compression)."""
    reference_rows = reference_rows or {}
    hid = math.ceil(label_px / block)
    rs = crop * block
    P = {name: load_process(p) for name, p in procs.items()}
    first = P[next(iter(P))]
    acc: dict = {}
    prev: dict = {}
    Rd = VideoReader(reference, dev)

    def upd(name, cur, src, key_prev):
        a = acc.setdefault(name, {"n": 0, "changed_frac": 0.0, "mean_abs_change": 0.0, "spot_px": 0.0, "spot_excess": 0.0,
                                  "added_temporal": 0.0, "nt": 0})
        diff = (cur - src).abs().amax(0)
        a["n"] += 1
        a["changed_frac"] += float((diff > 6 / 255).float().mean())
        a["mean_abs_change"] += float((cur - src).abs().mean())
        # small bright spots (catchlights, glints): luma well above the local 9x9 median
        luma = 0.2126 * cur[0] + 0.7152 * cur[1] + 0.0722 * cur[2]
        med = F.unfold(F.pad(luma[None, None], (4, 4, 4, 4), mode="replicate"), 9).median(1).values.view(luma.shape)
        ex = luma - med
        a["spot_px"] += float((ex > 0.15).float().sum())
        a["spot_excess"] += float(ex.clamp(min=0).sum())
        if key_prev in prev:
            pc, ps = prev[key_prev]
            a["added_temporal"] += float((cur - pc).abs().mean()) - float((src - ps).abs().mean())
            a["nt"] += 1
        prev[key_prev] = (cur, src)

    for (k, pts, rgb, ref), dr in zip(frames(video, first, dev, len(crops)), Rd):
        refimg = planes_to_rgb(dr, False, "bt709")
        for e, c in enumerate(crops[k]["eyes"]):
            if not c.get("matched", True):
                continue
            src_p = rgb[:, c["y"] + hid:c["y"] + crop, c["x"]:c["x"] + crop].clamp(0, 1)
            upd("SOURCE (plate)", src_p, src_p, ("src", e))
            ext_src = refimg[:, source_row * rs:(source_row + 1) * rs, e * rs:(e + 1) * rs][:, block // 2::block, block // 2::block][:, hid:, :]
            for label, ri in reference_rows.items():
                ext = refimg[:, ri * rs:(ri + 1) * rs, e * rs:(e + 1) * rs][:, block // 2::block, block // 2::block][:, hid:, :]
                upd(label, ext, ext_src, (label, e))
            for name, pr in P.items():
                o, _ = output_display(rgb, ref, pr, k)
                upd(name, o[:, c["y"] + hid:c["y"] + crop, c["x"]:c["x"] + crop].clamp(0, 1), src_p, (name, e))
    res = {}
    for name, a in acc.items():
        n = max(1, a["n"])
        res[name] = {"eye_crops": a["n"], "changed_fraction_of_crop_px": a["changed_frac"] / n, "mean_abs_change": a["mean_abs_change"] / n,
                     "bright_spot_px_per_crop": a["spot_px"] / n, "bright_spot_excess_per_crop": a["spot_excess"] / n,
                     "added_temporal_change_per_frame": (a["added_temporal"] / a["nt"]) if a["nt"] else None}
    return {"metrics": res, "crop_px": [crop - hid, crop], "notes": [
        "no clean ground truth exists for this footage: these are descriptors, not quality scores",
        "changed fraction counts crop pixels differing from the source by more than 6/255 (external rows vs the external SOURCE row, "
        "which carries the same video compression; EyeClean vs the decoded plate)",
        "bright spots: display luma more than 0.15 above the local 9x9 median inside the eye crops (catchlights, but also sclera/skin "
        "glints); fewer is not automatically better - a method can also darken or blur real detail",
        "added temporal change: mean |frame-to-frame change| of the method's crop minus that of the source crop (flicker added)"]}

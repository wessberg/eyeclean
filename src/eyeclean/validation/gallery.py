"""Held-out example gallery (runs/<profile>/examples/index.html) from actual checkpoint outputs.

Videos are NVENC H.264 viewing previews; stills are nvJPEG (GPU) encodes.  Predictions are the
process run's exported layers applied to the input (never the generator's clean render).  The
target column is ground truth shown for comparison only (demo cases only).
"""

from __future__ import annotations

import html
import json
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.io import encode_jpeg

from ..color import transforms as CT
from ..gpu.audit import boundary
from ..gpu.imgops import CropTransform, extract_crop
from ..io.exr import read_exr
from ..io.video import VideoWriter
from .evaluate import _eye_center, _read_rgb, reconstruct

CASE_TEXT = {
    "small_moving_softbox": "Small softbox moving across the cornea (recoverable reflection, moving highlight).",
    "saturated_pupil_crossing_no_donor": "Bright source saturating the cornea over the pupil boundary; static light and gaze, so no frame reveals the hidden iris (no donor).",
    "colored_multiple_lights": "Red, green and blue LEDs plus a softbox: coloured and multiple reflections.",
    "animated_flame_reflection": "Animated flame-like emissive source reflected in the eyes.",
    "dark_brown_iris": "Dark brown iris (low contrast between reflection-free iris and pupil).",
    "blue_green_irregular_iris": "Blue iris with crypt-rich irregular structure.",
    "blink_eyelash_occlusion": "Full blink with downward gaze: lashes occlude, closed frames must get zero support.",
    "small_eye_wide_shot": "Wide shot: iris only ~10-20 px; near/below the minimum processing size (passthrough expected when too small).",
    "no_reflection_negative": "Controlled negative: no designated ocular reflection exists; any edit is a false edit.",
    "non_eye_bright_highlights": "Bright background cards and strong skin highlights that must not be edited.",
    "head_motion": "Head turning with a static camera; tracking and temporal fusion under motion.",
    "shot_cut_identity_change": "Hard cut to a different person: tracks, temporal buffers and canonical states must reset.",
}


def _disp(x_work: torch.Tensor, expo: float) -> torch.Tensor:
    l709 = CT.working_to_linear709(x_work)
    return CT.bt709_oetf(torch.clamp(l709 * expo, 0, 1))


def _jpeg(img01: torch.Tensor, path: Path, q: int = 90) -> None:
    u8 = (img01.clamp(0, 1) * 255).round().to(torch.uint8).contiguous()
    data = encode_jpeg(u8, quality=q)
    with boundary("preview_png"):
        path.write_bytes(bytes(data.cpu().numpy()))


def _heat(d: torch.Tensor) -> torch.Tensor:
    v = d.clamp(0, 1)
    return torch.stack([v.clamp(0, 1) * 1.5, (v - 0.33).clamp(0, 1) * 1.5, (v - 0.66).clamp(0, 1) * 3]).clamp(0, 1)


def build_case_gallery(run_dir: Path, case_dir: Path, dev) -> dict:
    case = case_dir.name
    ev = run_dir / "eval_private" / case
    meta = json.loads((ev / "meta.json").read_text(encoding="utf-8"))
    proc = case_dir / "process_temporal"
    index = {e["frame"]: e for e in json.loads((proc / "patches" / "index.json").read_text(encoding="utf-8"))["frames"]}
    frames = meta["frames"]
    inp0 = CT.to_working(_read_rgb(Path(meta["input_dir"]) / "frame_000000.exr", dev), meta["input_color"])
    H, W = inp0.shape[1:]
    l709 = CT.working_to_linear709(inp0)
    lum = 0.2126 * l709[0] + 0.7152 * l709[1] + 0.0722 * l709[2]
    q = torch.quantile(lum.flatten()[:: max(1, lum.numel() // 262144)], torch.tensor([0.5, 0.97], device=dev))
    expo = float(torch.minimum(0.18 / q[0].clamp(min=1e-4), 0.85 / q[1].clamp(min=1e-4)))
    fps = meta.get("fps", 24.0)
    s = min(1.0, 1280 / W, 720 / H)
    pw, ph = int(W * s) // 2 * 2, int(H * s) // 2 * 2
    writers = {n: VideoWriter(str(case_dir / f"{n}.mp4"), pw, ph, fps) for n in ("input", "prediction", "target")}
    comp = VideoWriter(str(case_dir / "comparison.mp4"), pw * 2, ph * 2, fps)
    zoom_S = 256
    zoom = VideoWriter(str(case_dir / "comparison_zoom.mp4"), zoom_S * 3, zoom_S * 2, fps)
    mid = len(frames) // 2
    stills = {}
    for f in frames:
        k = f["index"]
        inp = CT.to_working(_read_rgb(Path(meta["input_dir"]) / f"frame_{k:06d}.exr", dev), meta["input_color"])
        tgt = CT.to_working(_read_rgb(ev / "target_exr" / f"frame_{k:06d}.exr", dev), meta["input_color"])
        r = reconstruct(proc, index, k, inp)
        di, dp, dt = _disp(inp, expo), _disp(r["pred"], expo), _disp(tgt, expo)
        diff = _heat(((r["pred"] - tgt).abs().mean(0) * expo * 8))
        ov = dp.clone()
        green = r["support"] & ~r["synth"]
        for m, c in ((green, (0.1, 1.0, 0.2)), (r["synth"], (1.0, 0.1, 0.9)), (r["abstained"], (1.0, 0.9, 0.1))):
            mm = m.float()[None]
            ov = ov * (1 - 0.6 * mm) + torch.tensor(c, device=dev).view(3, 1, 1) * 0.6 * mm
        rs = lambda t: F.interpolate(t[None], size=(ph, pw), mode="bilinear", antialias=True, align_corners=False)[0]
        writers["input"].write(rs(di), int(round(f["time_s"] * 90000)))
        writers["prediction"].write(rs(dp), int(round(f["time_s"] * 90000)))
        writers["target"].write(rs(dt), int(round(f["time_s"] * 90000)))
        grid = torch.cat([torch.cat([rs(di), rs(dp)], 2), torch.cat([rs(dt), rs(ov)], 2)], 1)
        comp.write(grid, int(round(f["time_s"] * 90000)))
        # eye zoom: centre between visible eyes (GT position used for display framing only)
        eyes = [e for e in f["eyes"].values() if e["in_frame"]]
        if eyes:
            cx = sum(_eye_center(e)[0] for e in eyes) / len(eyes)
            cy = sum(_eye_center(e)[1] for e in eyes) / len(eyes)
            span = max(abs(_eye_center(eyes[0])[0] - _eye_center(eyes[-1])[0]) + 3 * max(_eye_center(e)[2] for e in eyes), 24)
        else:
            cx, cy, span = W / 2, H / 2, min(W, H) / 3
        tf = CropTransform(cx, cy, zoom_S / span, 0.0, zoom_S)
        zi, zp, zt = extract_crop(di, tf), extract_crop(dp, tf), extract_crop(dt, tf)
        zo, zd = extract_crop(ov, tf), extract_crop(diff, tf)
        unc = torch.zeros_like(zi)
        zoom.write(torch.cat([torch.cat([zi, zp, zt], 2), torch.cat([zo, zd, (zp - zi).abs().mul(4).clamp(0, 1)], 2)], 1),
                   int(round(f["time_s"] * 90000)))
        if k == mid:
            sd = case_dir / "stills"
            sd.mkdir(exist_ok=True)
            for n, t in (("input", di), ("prediction", dp), ("target", dt), ("support_overlay", ov), ("error_heat", diff),
                         ("zoom_input", zi), ("zoom_prediction", zp), ("zoom_target", zt), ("zoom_overlay", zo), ("zoom_error", zd)):
                _jpeg(t, sd / f"{n}.jpg")
                stills[n] = f"stills/{n}.jpg"
    for w in writers.values():
        w.close()
    comp.close()
    zoom.close()
    hyb = case_dir / "process_hybrid"
    videos_extra = []
    if (hyb / "patches" / "index.json").exists():
        hidx = {e["frame"]: e for e in json.loads((hyb / "patches" / "index.json").read_text(encoding="utf-8"))["frames"]}
        hz = VideoWriter(str(case_dir / "comparison_zoom_hybrid.mp4"), zoom_S * 3, zoom_S, fps)
        for f in frames:
            k = f["index"]
            inp = CT.to_working(_read_rgb(Path(meta["input_dir"]) / f"frame_{k:06d}.exr", dev), meta["input_color"])
            r = reconstruct(hyb, hidx, k, inp)
            dp = _disp(r["pred"], expo)
            ov = dp.clone()
            for m, c in ((r["support"] & ~r["synth"], (0.1, 1.0, 0.2)), (r["synth"], (1.0, 0.1, 0.9)), (r["abstained"], (1.0, 0.9, 0.1))):
                mm = m.float()[None]
                ov = ov * (1 - 0.6 * mm) + torch.tensor(c, device=dev).view(3, 1, 1) * 0.6 * mm
            eyes = [e for e in f["eyes"].values() if e["in_frame"]]
            if eyes:
                cx = sum(_eye_center(e)[0] for e in eyes) / len(eyes)
                cy = sum(_eye_center(e)[1] for e in eyes) / len(eyes)
                span = max(abs(_eye_center(eyes[0])[0] - _eye_center(eyes[-1])[0]) + 3 * max(_eye_center(e)[2] for e in eyes), 24)
            else:
                cx, cy, span = W / 2, H / 2, min(W, H) / 3
            tf = CropTransform(cx, cy, zoom_S / span, 0.0, zoom_S)
            zi = extract_crop(_disp(inp, expo), tf)
            zh, zo = extract_crop(dp, tf), extract_crop(ov, tf)
            hz.write(torch.cat([zi, zh, zo], 2), int(round(f["time_s"] * 90000)))
            if k == mid:
                _jpeg(zh, case_dir / "stills" / "zoom_hybrid.jpg")
                _jpeg(zo, case_dir / "stills" / "zoom_hybrid_overlay.jpg")
                stills["zoom_hybrid"] = "stills/zoom_hybrid.jpg"
                stills["zoom_hybrid_overlay"] = "stills/zoom_hybrid_overlay.jpg"
        hz.close()
        videos_extra.append("comparison_zoom_hybrid.mp4")
    for d in ("patches", "masks", "uncertainty", "canonical"):
        src = proc / d
        if src.exists():
            dst = case_dir / d
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
    return {"case": case, "stills": stills,
            "videos": ["input.mp4", "prediction.mp4", "target.mp4", "comparison.mp4", "comparison_zoom.mp4"] + videos_extra}


def write_index(run_dir: Path, galleries: list[dict], evaluation: dict, report: dict | None) -> Path:
    ex = run_dir / "examples"
    cases = {c["case"]: c for c in evaluation.get("cases", [])}
    summ = evaluation.get("summary", {})
    st = report or {}
    rows = []
    for g in galleries:
        name = g["case"]
        short = name.split("_", 2)[-1] if name.startswith("demo_") else name
        c = cases.get(name, {})
        mt = c.get("models", {}).get("temporal", {})
        ms = c.get("models", {}).get("single", {})
        fmt = lambda v, p=3: "n/a" if v is None else (f"{v:.{p}f}" if isinstance(v, float) else str(v))
        pr = mt.get("pupil_rel") or []
        prs = sorted(pr)
        stats = (f"<table class=m><tr><th>masked MAE reduction</th><td>{fmt(mt.get('masked_mae_reduction'))}</td>"
                 f"<th>single-frame</th><td>{fmt(ms.get('masked_mae_reduction'))}</td></tr>"
                 f"<tr><th>recoverable coverage</th><td>{fmt(mt.get('coverage'))}</td><th>support px</th><td>{mt.get('support_px', 'n/a')}</td></tr>"
                 f"<tr><th>abstained px</th><td>{mt.get('abstained_px', 'n/a')}</td><th>synthesized px</th><td>{mt.get('synth_px', 'n/a')}</td></tr>"
                 f"<tr><th>off-eye changed px</th><td>{mt.get('offeye_px', 'n/a')}</td><th>pupil area err (median)</th>"
                 f"<td>{fmt(prs[len(prs)//2] if prs else None)}</td></tr></table>")
        stl = g["stills"]
        imgs = "".join(f"<figure><img src='{html.escape(name)}/{html.escape(stl[k])}' loading=lazy><figcaption>{k.replace('_', ' ')}</figcaption></figure>"
                       for k in ("zoom_input", "zoom_prediction", "zoom_target", "zoom_overlay", "zoom_error") if k in stl)
        rows.append(f"<section><h2>{html.escape(short.replace('_', ' '))}</h2><p>{html.escape(CASE_TEXT.get(short, ''))}</p>{stats}"
                    f"<video src='{html.escape(name)}/comparison_zoom.mp4' autoplay loop muted playsinline controls></video>"
                    f"<p class=cap>Eye zoom (display framing only). Top: input | prediction | target. Bottom: support overlay "
                    f"(<span class=g>green = recoverable edit</span>, <span class=p>magenta = synthesized</span>, "
                    f"<span class=y>yellow = abstained</span>) | |prediction-target| | |prediction-input|x4.</p>"
                    f"<div class=grid>{imgs}</div>"
                    + (f"<h3>Hybrid mode (--allow-synthesis)</h3><video src='{html.escape(name)}/comparison_zoom_hybrid.mp4' autoplay loop muted "
                       f"playsinline controls></video><p class=cap>input | hybrid output | overlay (magenta = synthesized from the frozen per-shot "
                       f"canonical iris; not a recovery of the hidden iris)</p>" if "comparison_zoom_hybrid.mp4" in g["videos"] else "")
                    + f"<details><summary>Full frame and files</summary><video src='{html.escape(name)}/comparison.mp4' loop muted controls></video>"
                    f"<p class=cap>Top: input | prediction. Bottom: target | overlay. Files: "
                    + " ".join(f"<a href='{html.escape(name)}/{v}'>{v}</a>" for v in g["videos"])
                    + f" · <a href='{html.escape(name)}/patches/'>patches/</a> (authoritative signed FLOAT EXR corrections)"
                    f" · <a href='{html.escape(name)}/masks/'>masks/</a> · <a href='{html.escape(name)}/uncertainty/'>uncertainty/</a></p></details></section>")
    agg = summ.get("models", {}).get("temporal", {})
    head = (f"<p><b>Engineering status:</b> {html.escape(str(st.get('engineering_status', 'see RUN_REPORT.json')))} · "
            f"<b>Synthetic quality:</b> {html.escape(str(st.get('synthetic_quality_status', 'see RUN_REPORT.json')))} · "
            f"<b>Real footage:</b> {html.escape(str(st.get('real_quality_status', 'not_tested')))}</p>"
            f"<p>Aggregate (held-out test identities, full-frame detection, conservative mode, five-frame model): masked MAE reduction "
            f"{agg.get('masked_mae_reduction')}, recoverable coverage {agg.get('coverage')}, temporal/single-frame error ratio "
            f"{summ.get('temporal_error_ratio_vs_single_frame')}, detection recall {summ.get('detection_recall')}.</p>")
    page = f"""<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>EyeClean held-out examples</title><style>
:root{{--bg:#f6f6f4;--fg:#1d1d1f;--mut:#5f6368;--card:#fff;--line:#ddd}}
@media (prefers-color-scheme:dark){{:root{{--bg:#131314;--fg:#e8e8e8;--mut:#a0a0a0;--card:#1d1d1f;--line:#333}}}}
body{{margin:0;padding:24px 16px;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,Segoe UI,sans-serif}}
main{{max-width:1100px;margin:auto}} section{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;margin:18px 0}}
video{{width:100%;max-width:100%;border-radius:6px;background:#000}} .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:8px}}
figure{{margin:0}} img{{width:100%;border-radius:4px}} figcaption,.cap{{color:var(--mut);font-size:13px}}
table.m{{border-collapse:collapse;margin:8px 0;font-size:13px}} table.m th{{text-align:left;font-weight:500;color:var(--mut);padding:2px 10px 2px 0}}
table.m td{{padding:2px 18px 2px 0;font-variant-numeric:tabular-nums}} .g{{color:#1a9e3a}} .p{{color:#b0189a}} .y{{color:#a88a00}}
.warn{{border-left:4px solid #c77800;padding:8px 12px;background:var(--card)}}</style></head><body><main>
<h1>EyeClean — held-out synthetic examples</h1>{head}
<p class=warn>All inputs are procedurally generated CG heads of held-out test identities, scenes and (where marked) light/material
families, processed by the trained checkpoint through independent full-frame detection. Targets are shown only for comparison and
were not readable by the prediction process. MP4/JPEG files are lossy <b>viewing previews</b>; the authoritative outputs are the
uncompressed FLOAT EXR correction layers. Synthetic results do not establish real-footage quality.</p>
{''.join(rows)}
<p class=cap>Links: <a href='../RUN_REPORT.json'>RUN_REPORT.json</a> · <a href='../MODEL_CARD.md'>MODEL_CARD.md</a> ·
<a href='../DATASET_CARD.md'>DATASET_CARD.md</a> · <a href='../evaluation/evaluation.json'>evaluation.json</a></p></main></body></html>"""
    p = ex / "index.html"
    p.write_text(page, encoding="utf-8")
    return p

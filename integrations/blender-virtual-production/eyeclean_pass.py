"""EyeClean eye pass for blender-virtual-production-experiment (reference adapter; copy into that product's scripts/).

Replaces the homegrown eye removal (eye_removal_gpu.py and friends) with EyeClean, run OUT OF PROCESS in its own
environment (EyeClean needs Python 3.12 + its own torch/CUDA build; this product's venvs are Python 3.13).

  python scripts/eyeclean_pass.py --shot music_videos/<video>/shots/<shot> --name eyeclean_r48

Reads   <shot>/key/fg/%06d.exr (linear Rec.709, ZIP HALF, as written by key_plate.py) and shot.json's exact frame rate.
Runs    `eyeclean process --job ... --progress jsonl` (engine located by EYECLEAN_CMD / EYECLEAN_PYTHON / PATH), with
        input_color = working_color = linear-rec709, so EyeClean's corrections are in the fg's own linear space.
Writes  <shot>/composite/<name>/engine/          EyeClean's output (float EXR corrections, masks, RUN_REPORT.json)
        <shot>/composite/<name>/{source,removal,weight}/%06d.png + removal.json
                                                  the layout composite_ratio.py --eye-removal, sclera_glint_removal.py
                                                  --run, import_overlap_eyes.py, the audits and review scripts read
        <shot>/logs/<name>.json                   evidence: command, engine version, model hash, timings
The PNG layout is 8-bit sRGB, as this product's eye passes have always written it.  The float corrections in engine/ are
exact; docs/integrations/blender-virtual-production.md shows a `composite_ratio.py --eye-patches` path that applies them
without the 8-bit round trip.

Eye sides: this product names eyes from the SUBJECT's view (limbus_track.json: "left" is image-right); EyeClean from
the image's (image_left / image_right).  Each EyeClean eye is matched to the nearest limbus-track centre of its frame when
the track exists, else image_right -> "left", image_left -> "right".
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import cv2  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
try:                                            # the product's own conventions when run from its scripts/
    from color_pipeline import linear_to_srgb   # noqa: E402
except ImportError:                             # standalone use: the same sRGB curve
    def linear_to_srgb(x):
        x = np.maximum(x, 0)
        return np.where(x <= .0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - .055)
try:
    from eyeclean_client import RunOutput, run  # noqa: E402
except ImportError:
    sys.path.insert(0, os.environ.get("EYECLEAN_CLIENT", str(HERE.parent / "clients" / "python")))
    from eyeclean_client import RunOutput, run  # noqa: E402

T0 = time.time()
SIDE_OF = {"image_right": "left", "image_left": "right"}    # subject-view names used by this product


def log(*a):
    print(time.strftime("%H:%M:%S"), "[eyeclean_pass]", *a, flush=True)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def to8(lin):
    return np.rint(np.clip(linear_to_srgb(lin), 0, 1) * 255).astype(np.uint8)


def load_limbus(shot: Path):
    p = shot / "relight" / "limbus_track.json"
    if not p.exists():
        return {}
    rows = json.loads(p.read_text(encoding="utf-8"))["rows"]
    return {int(r["frame"]): {e["side"]: e.get("smoothed_centre") or e.get("centre") or e.get("landmark_centre")
                              for e in r["eyes"]} for r in rows}


def side_of(eye, frame, limbus):
    cand = limbus.get(frame) or {}
    dw = eye.data_window or eye.raw.get("roi")
    if cand and dw:
        cx, cy = (dw[0] + dw[2]) / 2, (dw[1] + dw[3]) / 2
        best = min(((s, (c[0] - cx) ** 2 + (c[1] - cy) ** 2) for s, c in cand.items() if c), key=lambda t: t[1], default=None)
        if best is not None and best[1] < 40 ** 2:
            return best[0], "limbus_track"
    return SIDE_OF.get(eye.eye, eye.eye), "image_position"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shot", required=True)
    ap.add_argument("--name", default="eyeclean")
    ap.add_argument("--out", default=None, help="run directory (default <shot>/composite/<name>)")
    ap.add_argument("--model", default=None, help="EyeClean model name or .safetensors (default: EyeClean's default)")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--engine", default=None, help="engine command (default: EYECLEAN_CMD / EYECLEAN_PYTHON / PATH)")
    ap.add_argument("--no-layout", action="store_true", help="only run the engine (float EXRs), skip the PNG layout")
    ap.add_argument("--cancel-file", default=None, help="stop the engine cleanly when this file appears (e.g. a GPU-release flag)")
    a = ap.parse_args()
    shot = Path(a.shot).resolve()
    meta = json.loads((shot / "shot.json").read_text(encoding="utf-8"))
    fps = meta.get("source_frame_rate") or str(meta.get("fps", ""))
    start = int(meta.get("source_start_frame", 0) or 0)
    run_dir = Path(a.out).resolve() if a.out else shot / "composite" / a.name
    engine_dir = run_dir / "engine"
    fg = shot / "key" / "fg"
    job = {"schema": "eyeclean.job/1", "input": str(fg / "%06d.exr"), "output": str(engine_dir),
           "input_color": "linear-rec709", "working_color": "linear-rec709", "fps": fps, "export": ["patches", "report"],
           "exr_compression": "zip", "client": {"product": "blender-virtual-production-experiment", "shot": meta.get("shot_id"),
                                                 "run": a.name}}
    if a.model:
        job["model"] = a.model
    if a.max_frames:
        job["max_frames"] = a.max_frames
    def on_event(ev):
        k = ev.get("event")
        if k == "frame" and (ev.get("done", 0) % 100 == 0 or ev.get("done") == ev.get("total")):
            log(f"{ev['stage']} {ev['done']}/{ev.get('total') or '?'}")
        elif k in ("start", "stage", "warning", "error"):
            log(k, {x: ev[x] for x in ("stage", "message", "frames_expected") if x in ev})

    engine = a.engine.split() if a.engine else None
    run_dir.mkdir(parents=True, exist_ok=True)
    log("engine run ->", engine_dir)
    res = run(job, engine=engine, on_event=on_event, cancel=a.cancel_file, stderr_path=str(run_dir / "engine_stderr.txt"))
    out = RunOutput(res.output)
    rep = out.report
    log("engine done:", rep["eye_status_counts"], f"{rep['seconds']} s")
    evidence = {"command": " ".join(sys.argv), "engine_output": str(engine_dir), "eyeclean_version": rep["eyeclean_version"],
                "model": rep["model"], "fps_exact": rep.get("fps_exact"), "eye_status_counts": rep["eye_status_counts"],
                "engine_seconds": rep["seconds"]}
    if a.no_layout:
        (run_dir / "eyeclean_pass.json").write_text(json.dumps(evidence, indent=1), encoding="utf-8")
        return
    manifest = json.loads((engine_dir / "source_manifest.json").read_text(encoding="utf-8"))
    shas = {os.path.normcase(os.path.abspath(k)): v for k, v in manifest["sha256_before"].items()}
    limbus = load_limbus(shot)
    for sub in ("source", "removal", "weight"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    rows = []
    for fr in out.frames():
        i = fr.frame
        path = fg / f"{i:06d}.exr"
        lin = cv2.imread(str(path), -1)[..., ::-1].astype(np.float32)
        src8 = to8(lin)
        out8 = src8.copy()
        weight8 = np.zeros(lin.shape[:2], np.uint8)
        eyes_rec, changed = {}, 0
        for eye in fr.eyes:
            side, how = side_of(eye, i, limbus)
            rec = {"side": side, "repaired": eye.status == "repaired", "reason": None if eye.status == "repaired" else
                   (eye.reasons[0] if eye.reasons else "untouched"), "eyeclean": {"track": eye.track, "eye": eye.eye,
                   "status": eye.status, "reasons": eye.reasons, "support_px": eye.support_px, "side_from": how}}
            if eye.status == "repaired":
                x0, y0, x1, y1 = eye.data_window
                corr = out.read(eye.patch)
                mask = out.read(eye.mask)
                hard = mask.channels["hard"] > 0.5
                feather = mask.channels["feather"].astype(np.float32)
                c = np.stack([corr.channels[k].astype(np.float32) for k in "RGB"], -1)
                win = lin[y0:y1, x0:x1] + c
                o8 = to8(win)
                out8[y0:y1, x0:x1][hard] = o8[hard]
                w = np.where(hard, np.maximum(1, np.rint(np.clip(feather, 0, 1) * 255)), 0).astype(np.uint8)
                weight8[y0:y1, x0:x1] = np.maximum(weight8[y0:y1, x0:x1], w)
                changed += int(hard.sum())
            eyes_rec[side] = rec
        for side in ("left", "right"):
            eyes_rec.setdefault(side, {"side": side, "repaired": False, "reason": "not_tracked_by_eyeclean"})
        assert np.array_equal(out8[weight8 == 0], src8[weight8 == 0]), "removal differs from source outside the weight"
        cv2.imwrite(str(run_dir / "source" / f"{i:06d}.png"), src8[..., ::-1])
        cv2.imwrite(str(run_dir / "removal" / f"{i:06d}.png"), out8[..., ::-1])
        cv2.imwrite(str(run_dir / "weight" / f"{i:06d}.png"), weight8)
        rows.append({"frame": i, "source_frame": start + i,
                     "source_sha256": shas.get(os.path.normcase(os.path.abspath(path))) or sha(path),
                     "changed_pixels": int((weight8 > 0).sum()), "weighted_pixels": round(float(weight8.sum()) / 255, 3),
                     "eyes": [eyes_rec["left"], eyes_rec["right"]] + [v for k, v in eyes_rec.items() if k not in ("left", "right")]})
        if (len(rows) % 200) == 0:
            log("layout", len(rows))
    repaired = [r for r in rows if r["changed_pixels"]]
    record = {"rows": rows, "frames": len(rows), "frames_repaired": len(repaired), "frames_exact_source": len(rows) - len(repaired),
              "total_changed_pixels": int(sum(r["changed_pixels"] for r in rows)),
              "method": f"EyeClean {rep['eyeclean_version']} ({rep['model'].get('name') or rep['model']['path']}); "
                        "float corrections in engine/patches, this layout is their 8-bit sRGB rendering",
              "implementation": "eyeclean", "eyeclean": {"version": rep["eyeclean_version"], "model": rep["model"],
                                                         "engine_output": "engine", "run_report_sha256": sha(engine_dir / "RUN_REPORT.json")},
              "source_outside_weight_exact": True, "generated_material_used": False, "scene_reflection_added": False,
              "production_enabled": False, "artist_accepted": False, "code_sha256": sha(Path(__file__)),
              "seconds": round(time.time() - T0, 1)}
    (run_dir / "removal.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    logs = shot / "logs" if not a.out else run_dir
    logs.mkdir(parents=True, exist_ok=True)
    (logs / f"{a.name}.json").write_text(json.dumps(dict(evidence, layout_frames=len(rows), seconds=record["seconds"]), indent=1),
                                         encoding="utf-8")
    log("written", len(rows), "frames ->", run_dir)


if __name__ == "__main__":
    main()

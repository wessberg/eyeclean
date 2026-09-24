"""Special engineering runs for the build report (each returns {'ok': bool, 'evidence': [...], ...})."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch

from ..color import transforms as CT
from ..io.exr import read_exr, write_exr
from .build_demo import ROOT, _cli


def _first_case(cases: dict, avoid=("no_reflection", "shot_cut", "small_eye", "blink")) -> str:
    for c in cases:
        if not any(a in c for a in avoid):
            return c
    return next(iter(cases))


def _index(proc: Path) -> list[dict]:
    return json.loads((proc / "patches" / "index.json").read_text(encoding="utf-8"))["frames"]


def strength_zero_run(run_dir: Path, cases: dict, best: Path, forbid: str) -> dict:
    case = _first_case(cases)
    cdir = run_dir / "examples" / case
    out = run_dir / "validation" / "strength_zero"
    if out.exists():
        shutil.rmtree(out)
    r = _cli(["process", str(cdir / "input_exr" / "frame_%06d.exr"), "--checkpoint", str(best), "--input-color", "linear-rec709", "--fps", "24",
              "--mode", "conservative", "--strength", "0", "--output", str(out), "--export", "patches,report", "--forbid-read", forbid,
              "--no-profile", "--full-frame-master"], run_dir / "logs" / "special_strength_zero.log")
    if r["returncode"] != 0:
        return {"ok": False, "evidence": [r["log"]], "reason": "process failed"}
    idx = _index(out)
    bypass = all(f["bypass"] == "strength_zero" and f["assigned_px"] == 0 for f in idx)
    # masters must be bit-identical to the input reference
    dev = torch.device("cuda")
    same = True
    for f in idx[:6]:
        m = read_exr(out / "master" / f"frame_{f['frame']:06d}.exr", dev)
        ref = CT.to_working(torch.stack([read_exr(cdir / "input_exr" / f"frame_{f['frame']:06d}.exr", dev).channels[c].float() for c in "RGB"]),
                            "linear-rec709")
        got = torch.stack([m.channels[c] for c in "RGB"])
        same &= bool((got.view(torch.int32) == ref.view(torch.int32)).all())
    return {"ok": bypass and same, "frames": len(idx), "all_bypassed": bypass, "masters_bit_identical": same,
            "evidence": [str(out / "RUN_REPORT.json"), str(out / "patches" / "index.json"), r["log"]], "command": r["command"]}


def no_eye_run(run_dir: Path, best: Path, dev) -> dict:
    src = run_dir / "validation" / "no_eye_input"
    src.mkdir(parents=True, exist_ok=True)
    g = torch.Generator(device=dev)
    g.manual_seed(5)
    for k in range(6):
        base = torch.rand((3, 45, 80), device=dev, generator=g)
        img = torch.nn.functional.interpolate(base[None], size=(360, 640), mode="bicubic", align_corners=False)[0].clamp(0) * 0.6
        write_exr(src / f"frame_{k:06d}.exr", {"R": img[0], "G": img[1], "B": img[2]})
    out = run_dir / "validation" / "no_eye"
    if out.exists():
        shutil.rmtree(out)
    r = _cli(["process", str(src / "frame_%06d.exr"), "--checkpoint", str(best), "--input-color", "linear-rec709", "--fps", "24",
              "--output", str(out), "--export", "patches,report", "--no-profile", "--full-frame-master"], run_dir / "logs" / "special_no_eye.log")
    if r["returncode"] != 0:
        return {"ok": False, "evidence": [r["log"]], "reason": "process failed"}
    idx = _index(out)
    tr = json.loads((out / "tracks.json").read_text(encoding="utf-8"))
    no_eyes = all(len(f["eyes"]) == 0 for f in tr["frames"])
    passthrough = all(f["assigned_px"] == 0 for f in idx)
    same = True
    for k in range(6):
        m = read_exr(out / "master" / f"frame_{k:06d}.exr", dev)
        ref = CT.to_working(torch.stack([read_exr(src / f"frame_{k:06d}.exr", dev).channels[c].float() for c in "RGB"]), "linear-rec709")
        same &= bool((torch.stack([m.channels[c] for c in "RGB"]).view(torch.int32) == ref.view(torch.int32)).all())
    return {"ok": no_eyes and passthrough and same, "frames": 6, "eyes_detected": not no_eyes, "masters_bit_identical": same,
            "evidence": [str(out / "RUN_REPORT.json"), str(out / "tracks.json"), r["log"]], "command": r["command"]}


def alpha_run(run_dir: Path, cases: dict, best: Path, dev) -> dict:
    case = _first_case(cases)
    cdir = run_dir / "examples" / case
    src = run_dir / "validation" / "alpha_input"
    src.mkdir(parents=True, exist_ok=True)
    g = torch.Generator(device=dev)
    g.manual_seed(9)
    alphas = []
    for k in range(6):
        img = read_exr(cdir / "input_exr" / f"frame_{k:06d}.exr", dev)
        H, W = img.channels["R"].shape
        a = torch.rand((H, W), device=dev, generator=g)
        a[0, 0], a[0, 1], a[0, 2] = -0.0, 1.0, 0.5
        alphas.append(a)
        write_exr(src / f"frame_{k:06d}.exr", {"R": img.channels["R"].float(), "G": img.channels["G"].float(), "B": img.channels["B"].float(), "A": a})
    out = run_dir / "validation" / "alpha"
    if out.exists():
        shutil.rmtree(out)
    r = _cli(["process", str(src / "frame_%06d.exr"), "--checkpoint", str(best), "--input-color", "linear-rec709", "--fps", "24",
              "--output", str(out), "--export", "patches,report", "--no-profile", "--full-frame-master"], run_dir / "logs" / "special_alpha.log")
    if r["returncode"] != 0:
        return {"ok": False, "evidence": [r["log"]], "reason": "process failed"}
    same = True
    for k in range(6):
        m = read_exr(out / "master" / f"frame_{k:06d}.exr", dev)
        same &= bool((m.channels["A"].view(torch.int32) == alphas[k].view(torch.int32)).all())
    man = json.loads((out / "source_manifest.json").read_text(encoding="utf-8"))
    return {"ok": same and man.get("alpha_frames_verified_unchanged") == 6, "alpha_bits_identical_in_masters": same,
            "frames_verified_in_pipeline": man.get("alpha_frames_verified_unchanged"),
            "evidence": [str(out / "source_manifest.json"), str(out / "RUN_REPORT.json"), r["log"]], "command": r["command"]}


def long_stream_run(run_dir: Path, cases: dict, best: Path, repeats: int = 6) -> dict:
    case = _first_case(cases)
    cdir = run_dir / "examples" / case
    src = run_dir / "validation" / "long_stream_input"
    if src.exists():
        shutil.rmtree(src)
    src.mkdir(parents=True)
    frames = sorted((cdir / "input_exr").glob("frame_*.exr"))
    k = 0
    for _ in range(repeats):
        for f in frames:
            dst = src / f"frame_{k:06d}.exr"
            try:
                os.link(f, dst)  # hard link: no data copy, source untouched
            except OSError:
                shutil.copy(f, dst)
            k += 1
    out = run_dir / "validation" / "long_stream"
    if out.exists():
        shutil.rmtree(out)
    r = _cli(["process", str(src / "frame_%06d.exr"), "--checkpoint", str(best), "--input-color", "linear-rec709", "--fps", "24",
              "--output", str(out), "--export", "patches,report", "--no-profile"], run_dir / "logs" / "special_long_stream.log")
    if r["returncode"] != 0:
        return {"ok": False, "evidence": [r["log"]], "reason": "process failed"}
    rep = json.loads((out / "RUN_REPORT.json").read_text(encoding="utf-8"))
    ms = rep["render"].get("mem_samples", [])
    half = len(ms) // 2
    first = max((m[2] for m in ms[:half]), default=None)
    second = max((m[2] for m in ms[half:]), default=None)
    ok = first is not None and second is not None and second <= first * 1.05 + 0.1
    return {"ok": ok, "frames": rep["render"]["frames"], "peak_alloc_first_half_gib": first, "peak_alloc_second_half_gib": second,
            "mem_samples": ms, "reason": f"{rep['render']['frames']}-frame 1080p stream: peak allocator memory first half {first} GiB, second half {second} GiB",
            "evidence": [str(out / "RUN_REPORT.json"), r["log"]], "command": r["command"]}


def resume_test(cfg: dict, run_dir: Path) -> dict:
    """Interrupted vs uninterrupted short guard training in scratch checkpoint dirs (not counted
    toward the model's step budget): the resumed run must restore step, optimizer, scheduler and
    the data-sampler cursor exactly."""
    from ..training.trainer import resume_probe

    return resume_probe(cfg, run_dir)


def _encode_case(run_dir: Path, case: str, out: Path, codec: str, bits: int, dev, pts: list[int] | None = None) -> dict:
    from ..io.video import VideoWriter

    cdir = run_dir / "examples" / case
    frames = sorted((cdir / "input_exr").glob("frame_*.exr"))
    img0 = read_exr(frames[0], dev)
    H, W = img0.channels["R"].shape
    lin = torch.stack([img0.channels[c].float() for c in "RGB"])
    lum = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]
    q = torch.quantile(lum.flatten()[::16], torch.tensor([0.5, 0.97], device=dev))
    expo = float(torch.minimum(0.18 / q[0].clamp(min=1e-4), 0.85 / q[1].clamp(min=1e-4)))
    w = VideoWriter(str(out), W, H, 24.0, codec, bits, qp=16)
    for k, f in enumerate(frames):
        im = read_exr(f, dev)
        lin = torch.stack([im.channels[c].float() for c in "RGB"])
        w.write(CT.bt709_oetf(torch.clamp(lin * expo, 0, 1)), pts[k] if pts else None)
    return w.close() | {"exposure": expo}


def own_footage_tests(run_dir: Path, cases: dict, best: Path, dev) -> dict:
    case = _first_case(cases)
    od = run_dir / "own_footage_test"
    if od.exists():
        shutil.rmtree(od)
    od.mkdir(parents=True)
    res = {"case": case, "tests": {}}
    ev = []
    # a) H.264 8-bit with sidecar (colour + range declared in <file>.eyeclean.json)
    h264 = od / "clip_h264.mp4"
    res["tests"]["encode_h264"] = _encode_case(run_dir, case, h264, "h264", 8, dev)
    (od / "clip_h264.mp4.eyeclean.json").write_text(json.dumps({"schema_version": "1.0", "input_color": "rec709", "input_range": "limited",
                                                                 "working_color": "linear-acescg", "notes": "test clip encoded by NVENC; range declared"}), encoding="utf-8")
    r = _cli(["process", str(h264), "--checkpoint", str(best), "--mode", "conservative", "--output", str(od / "out_h264"),
              "--gpu-policy", "strict", "--export", "patches,preview,report", "--no-profile"], od / "process_h264.log")
    res["tests"]["process_h264_sidecar"] = r
    ev += [str(od / "out_h264" / "RUN_REPORT.json"), str(od / "out_h264" / "source_manifest.json")]
    # b) apply-patches reference compositor
    r2 = _cli(["apply-patches", str(h264), "--patches", str(od / "out_h264" / "patches"), "--output", str(od / "master_h264"),
               "--gpu-policy", "strict"], od / "apply_h264.log")
    res["tests"]["apply_patches"] = r2
    ap = od / "master_h264" / "apply_report.json"
    ev.append(str(ap))
    # c) HEVC Main10 with explicit CLI flags (precedence over no sidecar)
    hevc = od / "clip_hevc10.mp4"
    res["tests"]["encode_hevc10"] = _encode_case(run_dir, case, hevc, "hevc", 10, dev)
    r3 = _cli(["process", str(hevc), "--checkpoint", str(best), "--input-color", "rec709", "--input-range", "limited", "--mode", "conservative",
               "--output", str(od / "out_hevc10"), "--gpu-policy", "strict", "--export", "patches,preview,report", "--no-profile"], od / "process_hevc10.log")
    res["tests"]["process_hevc10_cli"] = r3
    ev.append(str(od / "out_hevc10" / "RUN_REPORT.json"))
    # d) hybrid with --allow-synthesis
    r4 = _cli(["process", str(h264), "--checkpoint", str(best), "--mode", "hybrid", "--allow-synthesis", "--output", str(od / "out_h264_hybrid"),
               "--gpu-policy", "strict", "--export", "patches,preview,report", "--no-profile"], od / "process_h264_hybrid.log")
    res["tests"]["process_hybrid"] = r4
    ev.append(str(od / "out_h264_hybrid" / "RUN_REPORT.json"))
    # e) negative expectations: hybrid without flag, auto range without metadata, unsupported codec
    r5 = _cli(["process", str(h264), "--checkpoint", str(best), "--mode", "hybrid", "--output", str(od / "neg1")], od / "neg_hybrid_noflag.log")
    res["tests"]["hybrid_requires_allow_synthesis"] = {"returncode": r5["returncode"], "expected": 4, "log": r5["log"]}
    nosc = od / "nosidecar"
    nosc.mkdir()
    shutil.copy(hevc, nosc / "clip.mp4")
    r6 = _cli(["process", str(nosc / "clip.mp4"), "--checkpoint", str(best), "--input-color", "rec709", "--output", str(od / "neg2")], od / "neg_auto_range.log")
    res["tests"]["auto_range_needs_metadata"] = {"returncode": r6["returncode"], "expected": 4, "log": r6["log"]}
    prores = od / "clip_prores.mov"
    made = _make_prores(prores)
    if made:
        r7 = _cli(["process", str(prores), "--checkpoint", str(best), "--input-color", "rec709", "--input-range", "limited",
                   "--output", str(od / "neg3")], od / "neg_prores.log")
        res["tests"]["prores_rejected"] = {"returncode": r7["returncode"], "expected": 4, "log": r7["log"],
                                           "message": Path(r7["log"]).read_text(encoding="utf-8", errors="replace")[-600:]}
    else:
        res["tests"]["prores_rejected"] = {"returncode": None, "expected": 4, "reason": "could not create a ProRes fixture"}
    # f) VFR timestamps through process
    vfr = od / "clip_vfr.mp4"
    n = len(list((run_dir / "examples" / case / "input_exr").glob("frame_*.exr")))
    pts = []
    t = 0
    for k in range(n):
        pts.append(t)
        t += 3750 if k % 3 else 5000
    res["tests"]["encode_vfr"] = _encode_case(run_dir, case, vfr, "h264", 8, dev, pts)
    r8 = _cli(["process", str(vfr), "--checkpoint", str(best), "--input-color", "rec709", "--input-range", "limited",
               "--output", str(od / "out_vfr"), "--export", "patches,preview,report", "--no-profile"], od / "process_vfr.log")
    res["tests"]["process_vfr"] = r8
    vfr_ok = False
    if r8["returncode"] == 0:
        man = json.loads((od / "out_vfr" / "source_manifest.json").read_text(encoding="utf-8"))
        rep = json.loads((od / "out_vfr" / "RUN_REPORT.json").read_text(encoding="utf-8"))
        vfr_ok = man["pts"] == pts and rep["previews"]["preview"]["pts"] == pts and man["frame_count"] == n
        res["tests"]["vfr_check"] = {"source_pts": pts, "manifest_pts": man["pts"], "preview_pts": rep["previews"]["preview"]["pts"], "ok": vfr_ok}
        ev += [str(od / "out_vfr" / "source_manifest.json"), str(od / "out_vfr" / "RUN_REPORT.json")]
    ap_ok = ap.exists() and json.loads(ap.read_text(encoding="utf-8")).get("verified")
    mp4_ok = r["returncode"] == 0 and r3["returncode"] == 0
    ok = (mp4_ok and r2["returncode"] == 0 and bool(ap_ok) and r4["returncode"] == 0 and r5["returncode"] == 4 and r6["returncode"] == 4
          and (res["tests"]["prores_rejected"]["returncode"] == 4) and vfr_ok)
    res.update(ok=ok, mp4_ok=mp4_ok, vfr_ok=vfr_ok, apply_verified=bool(ap_ok), evidence=ev + [str(od / "own_footage_tests.json")],
               reason="NVENC-encoded H.264 8-bit (sidecar) and HEVC Main10 (CLI flags) processed; apply-patches verified; hybrid with "
                      "--allow-synthesis; hybrid without flag, auto range without metadata and ProRes rejected with exit code 4; VFR PTS preserved")
    (od / "own_footage_tests.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    return res


def _make_prores(path: Path) -> bool:
    """Test FIXTURE only: a tiny ProRes file via PyAV's CPU encoder (never part of processing)."""
    try:
        import av
        import numpy as np

        c = av.open(str(path), "w")
        s = c.add_stream("prores_ks", rate=24)
        s.width, s.height, s.pix_fmt = 320, 240, "yuv422p10le"
        for i in range(3):
            fr = av.VideoFrame.from_ndarray(np.full((240, 320, 3), 60 + i * 20, np.uint8), format="rgb24")
            for p in s.encode(fr):
                c.mux(p)
        for p in s.encode():
            c.mux(p)
        c.close()
        return True
    except Exception:
        return False


def canonical_check(run_dir: Path, cases: dict) -> dict:
    out = {"runs": {}, "evidence": []}
    ok_all = True
    for case in cases:
        proc = run_dir / "examples" / case / "process_hybrid"
        if not (proc / "tracks.json").exists():
            continue
        tr = json.loads((proc / "tracks.json").read_text(encoding="utf-8"))
        idx = _index(proc)
        digests: dict[str, set] = {}
        shot_of_frame = {f["frame"]: f["shot"] for f in tr["frames"]}
        for f in idx:
            for e in f["eyes"]:
                if e.get("canonical_digest"):
                    key = f"shot{shot_of_frame.get(f['frame'])}/track{e['track']}"
                    digests.setdefault(key, set()).add(e["canonical_digest"])
        states = tr.get("canonical", {})
        frozen = all(v.get("frozen") for v in states.values()) and bool(states)
        constant = all(len(v) == 1 for v in digests.values())
        saved = all((proc / "canonical" / f"{k.replace('/', '_')}.json").exists() for k in states)
        shots = {k.split("/")[0] for k in states}
        ok = frozen and constant and saved
        ok_all &= ok
        out["runs"][case] = {"states": len(states), "frozen": frozen, "digest_constant_per_shot_track": constant, "rendered_tracks": len(digests),
                             "saved": saved, "shots_with_states": sorted(shots)}
        out["evidence"] += [str(proc / "tracks.json"), str(proc / "patches" / "index.json")]
    out["ok"] = ok_all and bool(out["runs"])
    out["reason"] = ("hybrid runs build one frozen polar atlas per shot/track in the analysis pass; every rendered frame of a shot uses the "
                     "same atlas digest; states are saved under canonical/ and refit only after cuts")
    return out


def blink_cut_check(run_dir: Path, cases: dict) -> dict:
    res = {"evidence": []}
    cut = next((c for c in cases if "shot_cut" in c), None)
    blink = next((c for c in cases if "blink" in c), None)
    ok = True
    if cut:
        proc = run_dir / "examples" / cut / "process_temporal"
        tr = json.loads((proc / "tracks.json").read_text(encoding="utf-8"))
        meta = json.loads((run_dir / "eval_private" / cut / "meta.json").read_text(encoding="utf-8"))
        true_cut = next((f["index"] for f in meta["frames"] if f["cut_part"] == 1), None)
        before = {e["track"] for f in tr["frames"] if f["frame"] < true_cut for e in f["eyes"]}
        after = {e["track"] for f in tr["frames"] if f["frame"] >= true_cut for e in f["eyes"]}
        c_ok = true_cut in tr["cuts"] and not (before & after)
        res["cut"] = {"true_cut_frame": true_cut, "detected_cuts": tr["cuts"], "tracks_before": sorted(before), "tracks_after": sorted(after), "ok": c_ok}
        ok &= c_ok
        res["evidence"].append(str(proc / "tracks.json"))
    if blink:
        proc = run_dir / "examples" / blink / "process_temporal"
        idx = {f["frame"]: f for f in _index(proc)}
        meta = json.loads((run_dir / "eval_private" / blink / "meta.json").read_text(encoding="utf-8"))
        closed = [f["index"] for f in meta["frames"] if all(e["openness"] < 0.2 for e in f["eyes"].values())]
        leaked = [k for k in closed if idx.get(k, {}).get("assigned_px", 0) > 0]
        b_ok = bool(closed) and not leaked
        res["blink"] = {"closed_frames": closed, "frames_with_support_while_closed": leaked, "ok": b_ok}
        ok &= b_ok
        res["evidence"].append(str(proc / "patches" / "index.json"))
    res["ok"] = ok and bool(cut) and bool(blink)
    res["reason"] = "cut detected at the true boundary with disjoint track ids before/after; zero support on frames where both eyes are closed"
    if not res["ok"]:
        bad = []
        if not res.get("cut", {}).get("ok", False):
            bad.append(f"cut: {res.get('cut')}")
        if not res.get("blink", {}).get("ok", False):
            bl = res.get("blink", {})
            bad.append(f"blink: support on closed-eye frames {bl.get('frames_with_support_while_closed')} of {bl.get('closed_frames')}")
        res["reason"] += " -- FAILED: " + "; ".join(bad)
    return res


def oracle_diagnostic(cfg: dict, run_dir: Path, best: Path, dev, n: int = 160) -> dict:
    """DIAGNOSTIC ONLY: test-split stills with generator (oracle) crops, bypassing detection."""
    from ..data.sampling import stable_seed
    from ..models.bundle import Bundle
    from ..models.inputs import clipped_mask, crop_exposure, guard_input, restore_input
    from ..training.data import CacheStore, TrainSampler, targets_from_batch

    b = Bundle(best, dev)
    store = CacheStore(run_dir, "test", kinds=("still",))
    smp = TrainSampler(store, dev, stable_seed(cfg["seed"], "oracle"), b.crop_size, 8.0, sim=cfg["data"].get("camera_sim"))
    buckets = {}
    for _ in range(n):
        bt = smp.batch(1)
        e = crop_exposure(bt["src"])
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            go = b.guard(guard_input(bt["src"], e))
        probs = torch.sigmoid(go["masks"].float())
        inp = restore_input(bt["src"], e, probs, clipped_mask(bt["src"], 1.0))
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = b.restore_single(inp)
        gt = targets_from_batch(bt, e)
        rec = gt["support"] * (1 - gt["synth"]) * bt["valid"]
        d = float(bt["native_iris"][0])
        bk = "iris<16px" if d < 16 else ("iris16-40px" if d < 40 else "iris>=40px")
        if bool(bt["negative"][0]):
            bk = "negative"
        a = buckets.setdefault(bk, {"ep": 0.0, "ei": 0.0, "px": 0.0, "n": 0, "neg_res": []})
        a["n"] += 1
        if bk == "negative":
            a["neg_res"].append(float(out["residual"].float().abs().mean()))
        else:
            pn = gt["src_n"] + out["residual"].float()
            a["ep"] += float(((pn - gt["tgt_n"]).abs().mean(1, keepdim=True) * rec).sum())
            a["ei"] += float(((gt["src_n"] - gt["tgt_n"]).abs().mean(1, keepdim=True) * rec).sum())
            a["px"] += float(rec.sum())
    res = {k: {"samples": v["n"], "masked_mae_reduction": (1 - v["ep"] / v["ei"]) if v["ei"] > 0 else None,
               "mean_abs_residual_on_negatives": (sum(v["neg_res"]) / len(v["neg_res"])) if v["neg_res"] else None} for k, v in buckets.items()}
    p = run_dir / "evaluation" / "oracle_crop_diagnostic.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"label": "DIAGNOSTIC: oracle generator crops, single-frame network only, no detection/gate", "buckets": res}, indent=1),
                 encoding="utf-8")
    return {"path": str(p), "buckets": res}


def real_footage(run_dir: Path, best: Path, cfg: dict, files: list[str] | None = None, input_color: str | None = None,
                 input_range: str | None = None) -> dict:
    """Bounded real-footage sample (no ground truth -> human review).  Files come from inputs/ or are
    named explicitly (e.g. own-footage/...); colour comes from a sidecar/metadata or explicit flags.
    Source files are only read; outputs go under <run>/real_footage/."""
    if files:
        media = [Path(f) for f in files]
        missing = [str(m) for m in media if not m.exists()]
        if missing:
            return {"status": "not_tested", "reason": f"named real-footage files not found: {missing}"}
    else:
        inputs = ROOT / "inputs"
        media = [p for p in inputs.rglob("*") if p.is_file() and not p.name.endswith(".eyeclean.json")] if inputs.exists() else []
    if not media:
        return {"status": "not_tested", "reason": "no files in inputs/ and none named with --real-footage"}
    n_max = cfg["validation"]["user_footage_sample_max_frames"]
    results = []
    for m in media:
        side = Path(str(m) + ".eyeclean.json")
        out = run_dir / "real_footage" / m.stem
        if out.exists():
            shutil.rmtree(out)
        args = ["process", str(m), "--checkpoint", str(best), "--output", str(out), "--mode", "conservative",
                "--max-frames", str(n_max), "--export", "patches,preview,report"]
        if input_color:
            args += ["--input-color", input_color]
        if input_range:
            args += ["--input-range", input_range]
        r = _cli(args, run_dir / "logs" / f"real_{m.stem}.log")
        row = {"file": str(m), "sidecar": side.exists(), "returncode": r["returncode"], "log": r["log"], "command": r["command"],
               "max_frames": n_max}
        if r["returncode"] == 0:
            try:
                from ..validation.review_video import split_video

                sv = split_video(str(m), out, out / "review", torch.device("cuda"), max_frames=n_max, label_out="EYECLEAN CONSERVATIVE")
                row["review_videos"] = [sv["eyes_split"], sv["eyes_split_slow"]]
                row["frames_with_edits"] = sv["frames_with_edits"]
                rep = json.loads((out / "RUN_REPORT.json").read_text(encoding="utf-8"))
                row["source_files_modified"] = rep.get("preservation", {}).get("source_files_modified")
            except Exception as e:  # review video is evidence for humans; its failure is recorded, not hidden
                row["review_video_error"] = f"{type(e).__name__}: {e}"[:400]
        results.append(row)
    processed = [x for x in results if x["returncode"] == 0]
    return {"status": "review_required" if processed else "not_tested", "results": results, "files": [str(m) for m in media],
            "reason": "no ground truth for real footage; outputs require human review" if processed else "no input could be processed with explicit colour settings"}

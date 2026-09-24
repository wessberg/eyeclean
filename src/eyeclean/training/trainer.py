"""`eyeclean train --stage guard|image|temporal`: bounded, resumable CUDA training.

AdamW, linear warmup then cosine decay, gradient clipping, EMA weights, BF16 network with FP32
losses, gradient accumulation to the configured effective batch, activation checkpointing.  On
CUDA OOM the microbatch is halved and accumulation doubled (never CPU, never smaller renders);
every adaptation is recorded.  Checkpoints carry model/EMA weights (safetensors) and a resume
state with optimizer, scheduler, RNG and data-sampler cursor.
"""

from __future__ import annotations

import copy
import json
import math
import os
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from ..data.sampling import stable_seed
from ..gpu.audit import GpuAudit, audited, profile_stage
from ..gpu.runtime import PeakMemoryMonitor, configure_memory_budget, require_cuda
from ..models.eyeguard import EyeGuard
from ..models.eyerestore import build_restore
from ..models.inputs import crop_exposure, clipped_mask, guard_input, restore_input
from ..models.naf import count_parameters
from ..validation.acceptance import freeze
from .data import CacheStore, TrainSampler, collate, targets_from_batch
from .losses import guard_loss, restore_loss, temporal_error_loss

STAGES = ("guard", "image", "temporal")


def _atomic_save_st(tensors: dict, path: Path, meta: dict | None = None) -> None:
    tmp = path.with_suffix(".part")
    save_file({k: v.detach().contiguous().cpu() for k, v in tensors.items()}, str(tmp), metadata={k: str(v) for k, v in (meta or {}).items()})
    with open(tmp, "rb+") as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _atomic_torch_save(obj, path: Path) -> None:
    tmp = path.with_suffix(".part")
    torch.save(obj, tmp)
    with open(tmp, "rb+") as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)


def lr_lambda(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    t = (step - warmup) / max(1, total - warmup)
    return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(1.0, t)))


def load_guard(run_dir: Path, dev) -> EyeGuard:
    p = run_dir / "checkpoints" / "guard" / "best_ema.safetensors"
    if not p.exists():
        raise RuntimeError(f"guard checkpoint missing: {p} (run `eyeclean train --stage guard` first)")
    sd = load_file(str(p), device=str(dev))
    g = EyeGuard(presence_head="refl_ev.weight" in sd).to(dev)
    g.load_state_dict(sd)
    g.eval()
    for q in g.parameters():
        q.requires_grad_(False)
    return g


@torch.no_grad()
def guard_probs(guard: EyeGuard, crop_w: torch.Tensor, e: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.autocast("cuda", dtype=torch.bfloat16):
        o = guard(guard_input(crop_w, e))
    return torch.sigmoid(o["masks"].float()), torch.sigmoid(o["present"].float())


def steps_ledger(run_dir: Path) -> dict:
    p = run_dir / "manifests" / "optimizer_steps.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _write_ledger(run_dir: Path, led: dict) -> None:
    p = run_dir / "manifests" / "optimizer_steps.json"
    p.write_text(json.dumps(led, indent=2), encoding="utf-8")


class _Val:
    """Fixed validation crops (deterministic seed) kept on the GPU in float16."""

    def __init__(self, store: CacheStore, dev, S: int, n: int, seed: int, min_iris: float, windows: int = 0, sim: dict | None = None,
                 vis: float | None = None):
        self.vis = vis  # v2 visibility threshold of the guard's reflection targets
        smp = TrainSampler(store, dev, seed, S, min_iris, sim=sim)
        self.items = []
        for _ in range(n):
            b = smp.batch(1)
            self.items.append({k: (v.half() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in b.items()})
        self.windows = []
        if windows and store.clips:
            for _ in range(windows):
                w = smp.clip_window(T=6)
                self.windows.append({k: (v.half() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in w.items()})

    @staticmethod
    def full(b: dict) -> dict:
        return {k: (v.float() if torch.is_tensor(v) and v.dtype == torch.float16 else v) for k, v in b.items()}


def _restore_forward(model, guard, b: dict):
    e = crop_exposure(b["src"])
    probs, present = guard_probs(guard, b["src"], e)
    inp = restore_input(b["src"], e, probs, clipped_mask(b["src"], 1.0))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(inp)
    return out, e, probs, present, inp


def _window_inputs(guard, w: dict):
    """w: collated T-frame window -> xs (1,T,C,S,S), e per frame, validity (1,T), gts."""
    e = crop_exposure(w["src"])
    probs, present = guard_probs(guard, w["src"], e)
    inp = restore_input(w["src"], e, probs, clipped_mask(w["src"], 1.0))
    valid = (w["inside"].bool() & (present > 0.5))[None]
    return inp[None], e, valid


@torch.no_grad()
def validate(stage: str, model, guard, val: _Val) -> dict:
    model.eval()
    if stage == "guard":
        inter = torch.zeros(6, device="cuda")
        union = torch.zeros(6, device="cuda")
        pres_ok = 0
        rp_ok = rp_n = rp_fp = rp_neg = 0
        for it in val.items:
            b = _Val.full(it)
            e = crop_exposure(b["src"])
            gt = targets_from_batch(b, e, vis=val.vis)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = model(guard_input(b["src"], e))
            p = (torch.sigmoid(o["masks"].float()) > 0.5).float() * b["valid"]
            t = (gt["masks"] > 0.5).float() * b["valid"]
            inter += (p * t).sum((0, 2, 3))
            union += ((p + t) > 0).float().sum((0, 2, 3))
            pres_ok += int(((torch.sigmoid(o["present"].float()) > 0.5).float() == gt["present"]).all())
            if "refl_present" in o and "refl_present" in gt:
                pr = (torch.sigmoid(o["refl_present"].float()) > 0.5).float()
                w = gt["refl_present_w"] > 0
                rp_ok += int(((pr == gt["refl_present"]) & w).sum())
                rp_n += int(w.sum())
                negm = w & (gt["refl_present"] == 0)
                rp_fp += int((negm & (pr == 1)).sum())
                rp_neg += int(negm.sum())
        iou = (inter / union.clamp(min=1)).tolist()
        score = sum(iou[:5]) / 5
        model.train()
        res = {"iou": dict(zip(("ocular", "iris", "pupil", "protected", "reflection", "clipped"), iou)),
               "present_acc": pres_ok / len(val.items), "score": score}
        if rp_n:
            res["refl_present_acc"] = rp_ok / rp_n
            res["refl_present_false_pos_on_negatives"] = (rp_fp / rp_neg) if rp_neg else None
            res["score"] = score + 0.2 * (rp_ok / rp_n)
        return res
    mae_p = mae_i = 0.0
    n_px = 0.0
    neg_res = []
    temporal_err = []
    for it in val.items:
        b = _Val.full(it)
        if stage == "temporal":
            xs_one, e, _ = _window_inputs(guard, {**b, "inside": torch.ones(1, device="cuda")})
            xs = xs_one.expand(1, 5, *xs_one.shape[2:]).contiguous()
            valid = torch.zeros(1, 5, dtype=torch.bool, device="cuda")
            valid[:, 2] = True
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model.forward_video(xs, valid)
        else:
            out, e, _, _, _ = _restore_forward(model, guard, b)
        gt = targets_from_batch(b, e)
        res = out["residual"].float()
        rec = gt["support"] * (1 - gt["synth"]) * b["valid"]
        if bool(b["negative"].any()):
            neg_res.append(float(res.abs().mean()))
        elif rec.sum() > 0:
            pred_n = gt["src_n"] + res
            mae_p += float(((pred_n - gt["tgt_n"]).abs().mean(1, keepdim=True) * rec).sum())
            mae_i += float(((gt["src_n"] - gt["tgt_n"]).abs().mean(1, keepdim=True) * rec).sum())
            n_px += float(rec.sum())
    if stage == "temporal":
        for wv in val.windows:
            w = _Val.full(wv)
            xs, e, valid = _window_inputs(guard, w)
            gts = targets_from_batch(w, e)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                oa = model.forward_video(xs[:, 0:5], valid[:, 0:5])
                ob = model.forward_video(xs[:, 1:6], valid[:, 1:6])
            pa = gts["src_n"][2:3] + oa["residual"].float()
            pb = gts["src_n"][3:4] + ob["residual"].float()
            wsup = ((gts["support"][2:3] + gts["support"][3:4]) > 0).float() * w["valid"][2:3] * w["valid"][3:4]
            if wsup.sum() > 0:
                temporal_err.append(float(temporal_error_loss(pa, gts["tgt_n"][2:3], pb, gts["tgt_n"][3:4], wsup)))
    red = 1 - mae_p / mae_i if mae_i > 0 else float("nan")
    neg = sum(neg_res) / len(neg_res) if neg_res else 0.0
    model.train()
    out = {"masked_mae_reduction": red, "neg_mean_abs_residual": neg, "score": (red if red == red else -1.0) - 10 * neg,
           "support_px": n_px}
    if temporal_err:
        out["temporal_error"] = sum(temporal_err) / len(temporal_err)
    return out


def sampler_digest(sampler) -> str:
    import hashlib

    st = sampler.state_dict()
    return hashlib.sha256(repr(st["py_random"]).encode() + bytes(st["cuda_gen"].cpu().tolist())).hexdigest()


def resume_probe(cfg: dict, run_dir: Path, steps: int = 20, stop: int = 10) -> dict:
    """Uninterrupted vs interrupted+resumed guard training in scratch dirs (not counted as model steps)."""
    base = run_dir / "validation" / "resume_probe"
    import shutil as _sh

    if base.exists():
        _sh.rmtree(base)
    a = train_stage(cfg, "guard", resume="never", max_steps_override=steps, ckpt_dir_override=base / "uninterrupted", count_steps=False, quiet=True)
    b1 = train_stage(cfg, "guard", resume="never", max_steps_override=steps, ckpt_dir_override=base / "interrupted", stop_at=stop,
                     count_steps=False, quiet=True)
    b2 = train_stage(cfg, "guard", resume="require", max_steps_override=steps, ckpt_dir_override=base / "interrupted", count_steps=False, quiet=True)
    wa = load_file(str(base / "uninterrupted" / "last.safetensors"))
    wb = load_file(str(base / "interrupted" / "last.safetensors"))
    maxdiff = max(float((wa[k].float() - wb[k].float()).abs().max()) for k in wa)
    res = {"steps": steps, "interrupted_at": b1["steps"], "resumed_to": b2["steps"], "sampler_digest_uninterrupted": a["sampler_digest"],
           "sampler_digest_resumed": b2["sampler_digest"], "data_cursor_identical": a["sampler_digest"] == b2["sampler_digest"],
           "weights_max_abs_diff": maxdiff, "note": "GPU kernels are not bitwise deterministic; data order/optimizer/scheduler state are restored exactly"}
    res["ok"] = res["data_cursor_identical"] and b1["steps"] == stop and b2["steps"] == steps and maxdiff < 5e-2
    (base / "resume_probe.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    res["evidence"] = [str(base / "resume_probe.json")]
    return res


def train_stage(cfg: dict, stage: str, resume: str = "auto", log=print, max_steps_override: int | None = None,
                ckpt_dir_override: Path | None = None, stop_at: int | None = None, count_steps: bool = True, quiet: bool = False) -> dict:
    if stage not in STAGES:
        raise ValueError(stage)
    if quiet:
        log = lambda *a, **k: None  # noqa: E731
    dev = require_cuda(cfg["gpu"]["policy"], cfg["gpu"]["device_index"])
    budget = configure_memory_budget(cfg["gpu"]["target_working_gib"], cfg["gpu"]["reserve_free_gib"])
    run_dir = Path(cfg["run_dir"]).resolve()
    frozen = freeze(run_dir)
    tr = cfg["training"]
    st = tr[stage]
    total_steps = int(max_steps_override or st["steps"])
    S = cfg["model"]["crop_size"]
    seed = stable_seed(cfg["seed"], "train", stage)
    torch.manual_seed(seed)
    ckdir = Path(ckpt_dir_override) if ckpt_dir_override else run_dir / "checkpoints" / stage
    ckdir.mkdir(parents=True, exist_ok=True)
    led = steps_ledger(run_dir) if count_steps else {stage: [0]}  # {stage: [steps of each run]}; repairs appended
    runs = led.setdefault(stage, [])
    fresh_run = not ((ckdir / "last_state.pt").exists() and resume in ("auto", "require"))
    if fresh_run and count_steps:
        runs.append(0)
    spent_elsewhere = sum(sum(v) for k, v in led.items()) - (runs[-1] if runs else 0)
    if count_steps and spent_elsewhere + total_steps > cfg["resources"]["max_total_optimizer_steps"]:
        raise RuntimeError(f"step cap: {spent_elsewhere} + {total_steps} > {cfg['resources']['max_total_optimizer_steps']}")
    min_iris = float(cfg["inference"]["minimum_iris_diameter_pixels"])
    store = CacheStore(run_dir, "train", kinds=("still", "clip"))
    vstore = CacheStore(run_dir, "validation", kinds=("still", "clip"))
    cam_sim = cfg["data"].get("camera_sim")  # v2 camera simulation (None = original prototype behaviour)
    sampler = TrainSampler(store, dev, seed, S, min_iris, sim=cam_sim)
    n_val = 48 if (cfg["profile"] != "smoke" and not quiet) else 16
    val = _Val(vstore, dev, S, n_val, stable_seed(cfg["seed"], "val"), min_iris, windows=(12 if stage == "temporal" else 0), sim=cam_sim,
               vis=cfg["model"].get("reflection_visibility"))
    guard = None
    if stage == "guard":
        model = EyeGuard(presence_head=bool(cfg["model"].get("reflection_presence", False))).to(dev)
    else:
        guard = load_guard(run_dir, dev)
        model = build_restore(cfg["model"], checkpoint=tr["activation_checkpointing"]).to(dev)
        if stage == "temporal":
            ip = run_dir / "checkpoints" / "image" / "best_ema.safetensors"
            if not ip.exists():
                raise RuntimeError("image-stage checkpoint missing; train --stage image first")
            model.load_state_dict(load_file(str(ip), device=str(dev)))
    n_params = count_parameters(model)
    from ..gpu.compile import probe_compile

    compile_info = probe_compile() if cfg["gpu"]["compile"] == "auto_cuda_only" else {"mode": "cuda_eager", "reason": "disabled by configuration"}
    # the compiled path is only used when the backend works on this host; otherwise CUDA eager (never CPU)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    if stage == "temporal":
        tparams = list(model.temporal.parameters())
        tids = {id(p) for p in tparams}
        groups = [{"params": [p for p in model.parameters() if id(p) not in tids], "lr": tr["learning_rate"] * 0.5},
                  {"params": tparams, "lr": tr["learning_rate"]}]
    else:
        groups = [{"params": list(model.parameters()), "lr": tr["learning_rate"]}]
    opt = torch.optim.AdamW(groups, lr=tr["learning_rate"], weight_decay=tr["weight_decay"], betas=(0.9, 0.99))
    warm = min(tr["warmup_steps"], max(1, total_steps // 5))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_lambda(s, warm, total_steps))
    micro, eff = st["microbatch"], st["effective_batch"]
    state = {"step": 0, "micro": micro, "accum": eff // micro, "best": None, "history": [], "adaptations": list(budget.adaptations),
             "seconds": 0.0}
    sp = ckdir / "last_state.pt"
    if sp.exists() and resume in ("auto", "require"):
        ck = torch.load(sp, map_location=dev, weights_only=True)
        model.load_state_dict(load_file(str(ckdir / "last.safetensors"), device=str(dev)))
        ema.load_state_dict(load_file(str(ckdir / "last_ema.safetensors"), device=str(dev)))
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        sampler.load_state_dict(ck["sampler"])
        torch.set_rng_state(ck["torch_rng"].cpu())
        torch.cuda.set_rng_state(ck["cuda_rng"].cpu())
        state = ck["state"]
        log(f"[train:{stage}] resumed at step {state['step']}")
    elif resume == "require":
        raise RuntimeError(f"--resume require but no state at {sp}")
    mon = PeakMemoryMonitor(dev)
    # audit/profile evidence is persisted in the resumable state so finished-and-resumed stages keep it
    audit_summary = state.get("gpu_audit")
    prof_info = state.get("profile")
    t_start = time.time()
    model.train()

    def save_last():
        _atomic_save_st(model.state_dict(), ckdir / "last.safetensors", {"stage": stage, "step": state["step"]})
        _atomic_save_st(ema.state_dict(), ckdir / "last_ema.safetensors", {"stage": stage, "step": state["step"]})
        _atomic_torch_save({"opt": opt.state_dict(), "sched": sched.state_dict(), "sampler": sampler.state_dict(),
                            "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(), "state": state}, sp)
        if count_steps:
            led[stage][-1] = state["step"]
            _write_ledger(run_dir, led)

    def one_micro():
        if stage == "guard":
            b = sampler.batch(state["micro"], negative_frac=0.2)
            e = crop_exposure(b["src"])
            gt = targets_from_batch(b, e, vis=cfg["model"].get("reflection_visibility"))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = model(guard_input(b["src"], e))
            return guard_loss(o, gt, b["valid"])
        if stage == "image":
            b = sampler.batch(state["micro"])
            out, e, _, _, _ = _restore_forward(model, guard, b)
            gt = targets_from_batch(b, e)
            return restore_loss(out, gt, b, e, conf_target=cfg["model"].get("conf_target", "tolerance"))
        # temporal: a 6-frame clip window -> two overlapping 5-frame predictions
        if sampler.r.random() < 0.3 or not store.clips:
            b = sampler.batch(1)
            xs_one, e, _ = _window_inputs(guard, {**b, "inside": torch.ones(1, device=dev)})
            xs = xs_one.expand(1, 5, *xs_one.shape[2:]).contiguous()
            valid = torch.zeros(1, 5, dtype=torch.bool, device=dev)
            valid[:, 2] = True
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model.forward_video(xs, valid)
            gt = targets_from_batch(b, e)
            return restore_loss(out, gt, b, e, conf_target=cfg["model"].get("conf_target", "tolerance"))
        w = sampler.clip_window(T=6)
        xs, e, valid = _window_inputs(guard, w)
        gts = targets_from_batch(w, e)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            enc = model.encode(xs[0])
        per = [[f[t:t + 1] for f in enc] for t in range(6)]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            oa = model.forward_video(xs[:, 0:5], valid[:, 0:5], cached=per[0:5])
            ob = model.forward_video(xs[:, 1:6], valid[:, 1:6], cached=per[1:6])
        sub = lambda d, i: {k: (v[i:i + 1] if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == 6 else v) for k, v in d.items()}
        ba, bb = sub(w, 2), sub(w, 3)
        ga, gb = targets_from_batch(ba, e[2:3]), targets_from_batch(bb, e[3:4])
        la, da = restore_loss(oa, ga, ba, e[2:3], conf_target=cfg["model"].get("conf_target", "tolerance"))
        lb, db = restore_loss(ob, gb, bb, e[3:4], conf_target=cfg["model"].get("conf_target", "tolerance"))
        pa = ga["src_n"] + oa["residual"].float()
        pb = gb["src_n"] + ob["residual"].float()
        wsup = ((ga["support"] + gb["support"]) > 0).float() * ba["valid"] * bb["valid"]
        lt = temporal_error_loss(pa, ga["tgt_n"], pb, gb["tgt_n"], wsup) if wsup.sum() > 0 else pa.sum() * 0
        da["temporal"] = float(lt)
        return 0.5 * (la + lb) + 0.5 * lt, da

    log_every = 25 if cfg["profile"] == "smoke" else 50
    stats = state.setdefault("sample_stats", {"micro_batches": 0, "saturated_frac_sum": 0.0, "negative_frac_sum": 0.0})

    def run_accum():
        lo, pa = None, {}
        for _ in range(state["accum"]):
            lo, pa = one_micro()
            (lo / state["accum"]).backward()
            if "saturated_frac" in pa:
                stats["micro_batches"] += 1
                stats["saturated_frac_sum"] += pa["saturated_frac"]
                stats["negative_frac_sum"] += pa["negative_frac"]
        return lo, pa

    while state["step"] < total_steps:
        if stop_at is not None and state["step"] >= stop_at:
            break  # simulated interruption (resume probe): state is saved below
        step_t0 = time.time()
        try:
            opt.zero_grad(set_to_none=True)
            parts = {}
            audit_this = state["step"] in (1, 2) and audit_summary is None
            prof_this = state["step"] == 3 and prof_info is None
            if audit_this:
                a = GpuAudit(strict=True)
                with audited(a):
                    loss, parts = run_accum()
                audit_summary = a.summary()
                audit_summary["stage"] = stage
                state["gpu_audit"] = audit_summary
            elif prof_this:
                with profile_stage(f"train_{stage}_step", run_dir / "profiles") as prof_info:
                    loss, parts = run_accum()
                state["profile"] = prof_info
            else:
                loss, parts = run_accum()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), tr["gradient_clip"])
            opt.step()
            sched.step()
            with torch.no_grad():
                d = tr["ema_decay"] if state["step"] > warm else 0.0
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.mul_(d).add_(pm.detach(), alpha=1 - d)
                for be, bm in zip(ema.buffers(), model.buffers()):
                    be.copy_(bm)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if state["micro"] == 1:
                raise
            state["micro"] //= 2
            state["accum"] *= 2
            msg = f"step {state['step']}: CUDA OOM -> microbatch {state['micro']}, accumulation {state['accum']}"
            state["adaptations"].append(msg)
            log(f"[train:{stage}] {msg}")
            continue
        state["step"] += 1
        state["seconds"] += time.time() - step_t0
        mon.sample()
        if state["step"] % log_every == 0 or state["step"] == 1:
            rec = {"step": state["step"], "loss": float(loss), "grad_norm": float(gn), "lr": sched.get_last_lr()[0], **parts,
                   "sec_per_step": round(time.time() - step_t0, 3)}
            state["history"].append(rec)
            log(f"[train:{stage}] " + " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()))
        if state["step"] % tr["validation_every_steps"] == 0 or state["step"] == total_steps:
            vm = validate(stage, ema, guard, val)
            vm["step"] = state["step"]
            state["history"].append({"val": vm})
            log(f"[train:{stage}] validation {json.dumps(vm)}")
            if state["best"] is None or vm["score"] > state["best"]["score"]:
                state["best"] = vm
                _atomic_save_st(ema.state_dict(), ckdir / "best_ema.safetensors", {"stage": stage, "step": state["step"], "score": vm["score"]})
                (ckdir / "best.json").write_text(json.dumps(vm, indent=2), encoding="utf-8")
        if state["step"] % tr["checkpoint_every_steps"] == 0 or state["step"] == total_steps:
            save_last()
    save_last()
    nmb = max(1, stats["micro_batches"])
    summary = {"stage": stage, "steps": state["step"], "parameters": n_params, "best": state["best"],
               "sampler_digest": sampler_digest(sampler),
               "sample_mix": {"saturated_fraction": stats["saturated_frac_sum"] / nmb, "negative_fraction": stats["negative_frac_sum"] / nmb,
                              "micro_batches": stats["micro_batches"],
                              "saturated_fraction_min": cfg["data"]["saturated_fraction_min"],
                              "meets_saturated_minimum": (stats["saturated_frac_sum"] / nmb >= cfg["data"]["saturated_fraction_min"]) if stats["micro_batches"] else None},
               "adaptations": state["adaptations"], "train_seconds": round(state["seconds"], 1), "memory": mon.result(),
               "budget": budget.as_dict(), "acceptance_sha256": frozen["sha256"], "gpu_audit": audit_summary, "compile": compile_info,
               "profile": prof_info, "checkpoint": str(ckdir / "best_ema.safetensors"), "wall_seconds": round(time.time() - t_start, 1)}
    (ckdir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (ckdir / "history.json").write_text(json.dumps(state["history"], indent=1, default=str), encoding="utf-8")
    return summary

"""Train the ReflectionPresence classifier on a frozen EyeGuard (v2 stage ``presence``).

Labels per eye crop (after the paired camera simulation, visibility in normalised space):
  1  clearly visible designated reflection: >= 12 px with change > vis and a peak change >= 0.15
  0  reflection-free eye (negative shot), crop without an eye, or an eye whose target equals the
     source everywhere (e.g. closed eyes, reflection elsewhere)
  don't care: weaker reflections (skipping them only means no edit, which is safe)
Validation (fixed-seed validation crops) reports the detection rate of clear reflections at <= 1%
false alarms on reflection-free eyes, the AUC, and fixed-threshold rates; the checkpoint with the
highest detection at <= 1% false alarms is kept.  The guard's masks are untouched, so restorers
trained on them stay valid.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from ..data.sampling import stable_seed
from ..gpu.runtime import require_cuda
from ..models.inputs import crop_exposure, guard_input, normalize
from ..models.presence import ReflectionPresence, presence_input
from .data import CacheStore, TrainSampler
from .trainer import load_guard, steps_ledger, _write_ledger


def presence_labels(b: dict, e: torch.Tensor, vis: float) -> tuple[torch.Tensor, torch.Tensor]:
    diff = (normalize(b["tgt"], e) - normalize(b["src"], e)).abs().amax(1, keepdim=True) * b["valid"]
    eye = (b["labels"][:, 0:1] > 0.2).float()
    d_eye = diff * eye
    n_vis = (d_eye > vis).float().sum((1, 2, 3))
    peak = d_eye.amax((1, 2, 3))
    any_change = (d_eye > 0.015).float().sum((1, 2, 3)) > 0
    neg = b["negative"].bool() | b["force_absent"].bool() | ~any_change
    pos = (n_vis >= 12) & (peak >= 0.15) & ~neg
    y = pos.float()
    w = (pos | neg).float()
    return y, w


def _metrics(pos: list[float], neg: list[float]) -> dict:
    if not pos or not neg:
        return {"auc": None, "tpr_at_fpr_1pct": None}
    negs = sorted(neg)
    thr = negs[min(len(negs) - 1, int(math.ceil(0.99 * len(negs))) - 1)]  # <= 1% of negatives above thr
    tpr = sum(p > thr for p in pos) / len(pos)
    auc = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg) / (len(pos) * len(neg))
    rates = {f"fpr@{t}": sum(q > t for q in neg) / len(neg) for t in (0.5, 0.9, 0.95, 0.98)}
    rates.update({f"tpr@{t}": sum(p > t for p in pos) / len(pos) for t in (0.5, 0.9, 0.95, 0.98)})
    return {"auc": auc, "tpr_at_fpr_1pct": tpr, "threshold_at_fpr_1pct": thr, "n_pos": len(pos), "n_neg": len(neg), **rates}


@torch.no_grad()
def evaluate_presence(model, guard, sampler: TrainSampler, n: int, vis: float) -> dict:
    model.eval()
    pos, neg = [], []
    for _ in range(n):
        b = sampler.batch(1, negative_frac=0.1)
        e = crop_exposure(b["src"])
        gi = guard_input(b["src"], e)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            probs = torch.sigmoid(guard(gi)["masks"].float())
            logit = model(presence_input(gi, probs))
        y, w = presence_labels(b, e, vis)
        if float(w[0]) > 0:
            (pos if float(y[0]) > 0 else neg).append(float(torch.sigmoid(logit.float())[0]))
    model.train()
    return _metrics(pos, neg)


def train_presence(cfg: dict, resume: str = "auto", log=print, ckpt_dir_override: Path | None = None, count_steps: bool = True) -> dict:
    dev = require_cuda(cfg["gpu"]["policy"], cfg["gpu"]["device_index"])
    run_dir = Path(cfg["run_dir"]).resolve()
    st = cfg["training"].get("presence") or {"steps": 4000, "microbatch": 16, "effective_batch": 16}
    total, micro = int(st["steps"]), int(st["microbatch"])
    accum = max(1, int(st["effective_batch"]) // micro)
    vis = float(cfg["model"].get("reflection_visibility", 0.06))
    ckdir = Path(ckpt_dir_override) if ckpt_dir_override else run_dir / "checkpoints" / "presence"
    ckdir.mkdir(parents=True, exist_ok=True)
    guard = load_guard(run_dir, dev)
    model = ReflectionPresence().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    warm = min(300, total // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    sim = cfg["data"].get("camera_sim")
    smp = TrainSampler(CacheStore(run_dir, "train"), dev, stable_seed(cfg["seed"], "train", "presence"), cfg["model"]["crop_size"],
                       float(cfg["inference"]["minimum_iris_diameter_pixels"]), sim=sim)
    vstore = CacheStore(run_dir, "validation")
    state = {"step": 0, "best": None, "history": [], "seconds": 0.0}
    sp = ckdir / "last_state.pt"
    led = steps_ledger(run_dir) if count_steps else {"presence": [0]}
    runs = led.setdefault("presence", [])
    if sp.exists() and resume in ("auto", "require"):
        ck = torch.load(sp, map_location=dev, weights_only=True)
        model.load_state_dict(load_file(str(ckdir / "last.safetensors"), device=str(dev)))
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        smp.load_state_dict(ck["sampler"])
        state = ck["state"]
        log(f"[train:presence] resumed at step {state['step']}")
    elif count_steps:
        runs.append(0)
    cap = cfg["resources"]["max_total_optimizer_steps"]
    if count_steps and sum(sum(v) for v in led.values()) - runs[-1] + total > cap:
        raise RuntimeError(f"step cap {cap} would be exceeded by the presence stage")

    def save_last():
        save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(ckdir / "last.safetensors"))
        torch.save({"opt": opt.state_dict(), "sched": sched.state_dict(), "sampler": smp.state_dict(), "state": state}, sp)
        if count_steps:
            runs[-1] = state["step"]
            _write_ledger(run_dir, led)

    model.train()
    t0 = time.time()
    while state["step"] < total:
        ts = time.time()
        opt.zero_grad(set_to_none=True)
        tot_loss = 0.0
        for _ in range(accum):
            b = smp.batch(micro, negative_frac=0.1)
            e = crop_exposure(b["src"])
            gi = guard_input(b["src"], e)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                probs = torch.sigmoid(guard(gi)["masks"].float())
            y, w = presence_labels(b, e, vis)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit = model(presence_input(gi, probs))
            # class-balanced focal loss (gamma 2): confidently wrong samples - above all near-miss false
            # alarms on reflection-free eyes - dominate; negatives carry twice the class weight
            wpos = w * y / (w * y).sum().clamp(min=1)
            wneg = 2.0 * w * (1 - y) / (w * (1 - y)).sum().clamp(min=1)
            pr = torch.sigmoid(logit.float())
            pt = torch.where(y > 0.5, pr, 1 - pr)
            bce = F.binary_cross_entropy_with_logits(logit.float(), y, reduction="none")
            loss = (bce * (1 - pt).pow(2) * (wpos + wneg)).sum() / 3.0 * 4.0
            (loss / accum).backward()
            tot_loss += float(loss) / accum
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        state["step"] += 1
        state["seconds"] += time.time() - ts
        if state["step"] % 100 == 0 or state["step"] == 1:
            log(f"[train:presence] step={state['step']} loss={tot_loss:.4f} lr={sched.get_last_lr()[0]:.3g} sec_per_step={time.time() - ts:.3f}")
        if state["step"] % 500 == 0 or state["step"] == total:
            vs = TrainSampler(vstore, dev, stable_seed(cfg["seed"], "val", "presence"), cfg["model"]["crop_size"],
                              float(cfg["inference"]["minimum_iris_diameter_pixels"]), sim=sim)
            vm = evaluate_presence(model, guard, vs, 600, vis)
            vm["step"] = state["step"]
            state["history"].append(vm)
            log("[train:presence] validation " + json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in vm.items()}))
            if vm["tpr_at_fpr_1pct"] is not None and (state["best"] is None or vm["tpr_at_fpr_1pct"] > state["best"]["tpr_at_fpr_1pct"]):
                state["best"] = vm
                save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(ckdir / "best.safetensors"))
        if state["step"] % 500 == 0 or state["step"] == total:
            save_last()
    save_last()
    summary = {"stage": "presence", "steps": state["step"], "parameters": sum(p.numel() for p in model.parameters()), "best": state["best"],
               "train_seconds": round(state["seconds"], 1), "wall_seconds": round(time.time() - t0, 1), "checkpoint": str(ckdir / "best.safetensors")}
    (ckdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (ckdir / "history.json").write_text(json.dumps(state["history"], indent=1), encoding="utf-8")
    return summary

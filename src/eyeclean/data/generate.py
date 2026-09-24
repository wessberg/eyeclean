"""`eyeclean generate`: plan, render and cache the paired CG dataset (resumable).

Stages (each resumable and recorded under <run_dir>/manifests):
  1. deterministic plan + split-integrity check (identities/textures/skins/scenes/clips disjoint)
  2. CUDA texture synthesis per identity
  3. pair validation of the target construction (must pass before any dataset render)
  4. Blender/Cycles OptiX renders (parallel headless workers, one identity per job)
  5. GPU conversion to float16 caches with per-frame decomposition checks; raw EXRs removed
     except a diagnostic subset; representative .blend scenes copied to <run_dir>/scenes
"""

from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import torch

from ..render import childproc
from ..render.blender import BlenderError, find_blender, run_script
from ..render.childproc import Cancelled
from . import cache as C
from .sampling import Plan, plan_dataset, plan_digest, split_integrity
from .textures import write_identity_textures


class GenerationError(RuntimeError):
    pass


def dir_size_gib(p: Path) -> float:
    tot = 0
    for root, _, files in os.walk(p):
        for f in files:
            try:
                tot += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return tot / 2**30


def free_gib(p: Path) -> float:
    return shutil.disk_usage(p).free / 2**30


def _write_json(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".part")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)


def check_resources(cfg: dict, run_dir: Path) -> dict:
    used = dir_size_gib(run_dir) if run_dir.exists() else 0.0
    free = free_gib(run_dir if run_dir.exists() else run_dir.parent)
    res = cfg["resources"]
    info = {"workspace_used_gib": round(used, 3), "workspace_cap_gib": res["workspace_cap_gib"],
            "free_disk_gib": round(free, 2), "min_free_disk_gib": res["min_free_disk_gib"]}
    if used > res["workspace_cap_gib"]:
        raise GenerationError(f"workspace {used:.1f} GiB exceeds cap {res['workspace_cap_gib']} GiB; stopping cleanly (resume later)")
    if free < res["min_free_disk_gib"]:
        raise GenerationError(f"free disk {free:.1f} GiB below minimum {res['min_free_disk_gib']} GiB; stopping cleanly")
    return info


def plan_extends(old: dict, new: dict) -> tuple[bool, str]:
    """Is ``new`` a pure superset of the already-rendered plan ``old``?

    True only when the identities are identical and every old shot is present in the new plan
    unchanged (both compared after a JSON round trip, so tuple/list spelling does not count), i.e.
    the new plan adds shots in a fresh index space and resuming renders only those.  Anything else
    - a re-seeded shot, a changed identity, a removed shot - is a different dataset.
    """
    if old.get("identities") != new.get("identities"):
        changed = [k for k in old.get("identities", {}) if new.get("identities", {}).get(k) != old["identities"][k]]
        missing = [k for k in old.get("identities", {}) if k not in new.get("identities", {})]
        return False, f"identities differ (changed {len(changed)}, missing {len(missing)})"
    by_id = {s["shot_id"]: s for s in new.get("shots", [])}
    changed = [s["shot_id"] for s in old.get("shots", []) if by_id.get(s["shot_id"]) != s]
    if changed:
        return False, f"{len(changed)} already-planned shots would change or vanish (e.g. {changed[:3]})"
    added = len(by_id) - len(old.get("shots", []))
    return True, f"pure extension: {added} shots added, {len(old.get('shots', []))} unchanged"


def prepare_plan(cfg: dict, run_dir: Path, resume: str, log=print) -> Plan:
    man = run_dir / "manifests"
    man.mkdir(parents=True, exist_ok=True)
    plan = plan_dataset(cfg)
    digest = plan_digest(plan)
    pp = man / "dataset_plan.json"
    if pp.exists():
        old = json.loads(pp.read_text(encoding="utf-8"))
        if old.get("digest") != digest:
            if resume == "require":
                raise GenerationError("existing plan differs from the configuration; refusing to mix datasets")
            ok, why = plan_extends(old, json.loads(json.dumps(plan.to_json())))
            if not ok:
                raise GenerationError(f"existing dataset plan digest {old.get('digest')} != {digest}; the config changed "
                                      f"({why}). Use a new run_dir instead of mixing datasets.")
            # a superset plan: keep the old one on disk for the record and adopt the new digest
            backup = man / f"dataset_plan.superseded_{str(old.get('digest'))[:12]}.json"
            if not backup.exists():
                _write_json(backup, old)
            _write_json(pp, dict(plan.to_json(), digest=digest, extends=old.get("digest")))
            log(f"[generate] plan extended ({why}); previous plan kept at {backup.name}")
    else:
        if resume == "require":
            raise GenerationError("--resume require but no existing plan")
        _write_json(pp, dict(plan.to_json(), digest=digest))
    integ = split_integrity(plan)
    _write_json(man / "split_integrity.json", integ)
    if not integ["passed"]:
        raise GenerationError(f"split integrity failed: {integ['problems']}")
    return plan


def _hm(s: float) -> str:
    s = int(s)
    return f"{s // 3600}h{s % 3600 // 60:02d}m" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"


def progress_line(done: int, total: int, converted: int, active_s: float, elapsed_s: float) -> str:
    """Status after each finished identity job; rate and ETA count only this run's unpaused render time."""
    rate = 60 * converted / active_s if converted and active_s > 0 else 0.0
    eta = _hm((total - done) / rate * 60) if rate else "n/a"
    return (f"[generate] {done}/{total} shots done ({100 * done / max(1, total):.1f}%) - {converted} converted this run, "
            f"{rate:.1f} shots/min, ETA {eta} ({_hm(elapsed_s)} elapsed)")


def generate(cfg: dict, resume: str = "auto", workers: int = 2, log=print, pause=None) -> dict:
    """``pause`` (a ``cli.pause.KeyboardPause``) holds new work while the user has paused the run.

    Ctrl+C terminates the Blender workers in flight, drops the rest of the queue, writes the
    manifests and re-raises: nothing is left rendering behind the CLI, and the next run resumes.
    """
    hold = pause.wait if pause is not None else (lambda: None)
    paused_s = pause.paused_seconds if pause is not None else (lambda: 0.0)
    childproc.reset()  # an earlier interrupted run in this process must not block this one
    run_dir = Path(cfg["run_dir"]).resolve()  # Blender needs absolute paths
    run_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    t0 = time.time()
    check_resources(cfg, run_dir)
    plan = prepare_plan(cfg, run_dir, resume, log=log)
    data_dir = run_dir / "data"
    tex_root, raw_root, cache_root = data_dir / "textures", data_dir / "raw", data_dir / "cache"
    blender = find_blender()
    log(f"[generate] {len(plan.identities)} identities, {len(plan.shots)} shots; Blender {blender.version} at {blender.path}")
    # 2. textures
    tex_paths = {}
    for iid, ident in plan.identities.items():
        hold()
        tex_paths[iid] = write_identity_textures(ident, tex_root / iid, dev)
    tex_manifest = {iid: {side: {k: {"path": v, "sha256": hashlib.sha256(Path(v).read_bytes()).hexdigest()} for k, v in p.items()}
                          for side, p in paths.items()} for iid, paths in tex_paths.items()}
    _write_json(run_dir / "manifests" / "textures.json", tex_manifest)
    # 3. pair validation gate
    pv_dir = run_dir / "validation" / "pair"
    rep_p = pv_dir / "pair_validation_report.json"
    if rep_p.exists() and json.loads(rep_p.read_text(encoding="utf-8")).get("passed"):
        pv = json.loads(rep_p.read_text(encoding="utf-8"))
    else:
        from ..validation.pair_validation import run as run_pair_validation
        first = sorted(tex_paths)[0]
        pv = run_pair_validation(pv_dir, tex_paths[first]["L"],
                                 clean_iris_render=bool((cfg["data"].get("catchlights") or {}).get("clean_iris_render")))
    if not pv["passed"]:
        raise GenerationError(f"pair validation failed; not rendering the dataset: {pv['checks']}")
    log("[generate] pair validation passed")
    # 4/5. renders + conversion
    pending: dict[str, list[dict]] = {}
    for sh in plan.shots:
        if not C.cache_valid(cache_root / sh["split"], sh["shot_id"]):
            pending.setdefault(sh["identity_id"], []).append(sh)
    diag_budget = {"train": 12, "validation": 4, "test": 8}
    diag_used = {k: len(list((data_dir / "diagnostic").glob(f"{k}_*"))) if (data_dir / "diagnostic").exists() else 0 for k in diag_budget}
    jobs = []
    for iid, shots in sorted(pending.items()):
        ident = plan.identities[iid]
        js = []
        for sh in shots:
            s = dict(sh)
            s["out_dir"] = str(raw_root / sh["shot_id"])
            s["layers"] = ["clean", "reflection", "lab_a", "lab_b"]
            s.setdefault("save_blend", False)
            js.append(s)
        job = {"identities": [dict({k: v for k, v in ident.items() if k not in ("iris", "sclera")}, textures=tex_paths[iid], shots=js,
                                   render_subdiv=2)],
               "overwrite": False, "status_path": str(raw_root / f"_job_{iid}.status.json")}
        jp = raw_root / f"_job_{iid}.json"
        _write_json(jp, job)
        jobs.append((iid, jp, js))
    total = len(plan.shots)
    done_before = total - sum(len(j[2]) for j in jobs)
    log(f"[generate] {total - done_before} shots pending in {len(jobs)} identity jobs; "
        f"{done_before}/{total} already done ({100 * done_before / max(1, total):.1f}%)")
    render_log = []
    converted = 0
    failures = []
    t_render, paused0 = time.time(), paused_s()

    def run_job(item):
        iid, jp, js = item
        hold()  # no new Blender worker starts while paused
        return iid, js, run_script("generate.py", ["--job", str(jp)], raw_root / f"_job_{iid}.log", blender)

    interrupted = None
    with cf.ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = [ex.submit(run_job, j) for j in jobs]
        try:
            for fut in cf.as_completed(futs):
                try:
                    iid, js, rinfo = fut.result()
                except Cancelled:
                    continue  # a queued job the interrupt reached before Blender started
                except BlenderError as e:
                    failures.append(str(e)[:2000])
                    log(f"[generate] Blender job failed: {str(e)[:400]}")
                    continue
                render_log.append({"identity": iid, **{k: v for k, v in rinfo.items() if k != "command"}})
                for sh in js:
                    hold()
                    raw = Path(sh["out_dir"])
                    if (raw / "error.json").exists() or not (raw / "meta.json").exists():
                        failures.append(f"{sh['shot_id']}: render error {(raw / 'error.json').read_text()[:500] if (raw / 'error.json').exists() else 'no meta'}")
                        continue
                    keep = False
                    if diag_used[sh["split"]] < diag_budget[sh["split"]] and sh["kind"] != "demo":
                        keep = True
                    try:
                        info = C.convert_shot(raw, cache_root / sh["split"], sh, dev, keep_raw=keep)
                    except Exception as e:
                        failures.append(f"{sh['shot_id']}: conversion failed: {e}")
                        log(f"[generate] conversion failed for {sh['shot_id']}: {e}")
                        (raw / "meta.json").unlink(missing_ok=True)  # forces a re-render on resume
                        continue
                    if keep:
                        dd = data_dir / "diagnostic" / f"{sh['split']}_{sh['shot_id']}"
                        dd.mkdir(parents=True, exist_ok=True)
                        for fm in info["frames"][:1]:
                            src = raw / fm["exr"]
                            if src.exists():
                                shutil.move(str(src), dd / fm["exr"])
                        shutil.copy(raw / "meta.json", dd / "meta.json")
                        diag_used[sh["split"]] += 1
                        for fm in info["frames"]:
                            (raw / fm["exr"]).unlink(missing_ok=True)
                    if (raw / "scene.blend").exists():
                        sd = run_dir / "scenes"
                        sd.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(raw / "scene.blend"), sd / f"{sh['shot_id']}.blend")
                        _write_json(sd / f"{sh['shot_id']}.json", {"shot": sh, "identity": plan.identities[sh["identity_id"]]})
                    converted += 1
                try:
                    check_resources(cfg, run_dir)
                except GenerationError as e:
                    failures.append(str(e))
                    log(f"[generate] {e}")
                    childproc.cancel_all()  # out of room: stop the renders in flight, not just the queue
                    ex.shutdown(cancel_futures=True)
                    break
                log(progress_line(done_before + converted, total, converted,
                                  time.time() - t_render - (paused_s() - paused0), time.time() - t0))
        except KeyboardInterrupt:
            interrupted = True
            queued = [f for f in futs if not f.done()]
            killed = childproc.cancel_all()
            ex.shutdown(wait=False, cancel_futures=True)
            log(f"[generate] interrupted: {killed} Blender worker(s) terminated, "
                f"{sum(1 for f in queued if f.cancelled())} queued job(s) dropped; "
                f"writing manifests - re-run the same command to resume from here")
    rl_path = run_dir / "manifests" / "render_log.json"
    prev = json.loads(rl_path.read_text(encoding="utf-8")) if rl_path.exists() else []
    _write_json(rl_path, prev + render_log)
    manifest = build_manifest(plan, cache_root, run_dir)
    if interrupted:  # after the manifests, so the run is on disk; callers must not record a pass
        raise KeyboardInterrupt
    return {"converted_now": converted, "failures": failures, "manifest": manifest, "seconds": round(time.time() - t0, 1),
            "pair_validation": {"passed": pv["passed"], "report": str(rep_p)}}


def build_manifest(plan: Plan, cache_root: Path, run_dir: Path) -> dict:
    shots = []
    missing = []
    for sh in plan.shots:
        jp = cache_root / sh["split"] / f"{sh['shot_id']}.json"
        if not jp.exists():
            missing.append(sh["shot_id"])
            continue
        info = json.loads(jp.read_text(encoding="utf-8"))
        ident = plan.identities[sh["identity_id"]]
        shots.append({"shot_id": sh["shot_id"], "split": sh["split"], "kind": sh["kind"], "identity_id": sh["identity_id"],
                      "iris_family": ident["iris_family"], "iris_color": ident["iris_color"], "skin": ident["skin"],
                      "light_families": [L["family"] for L in sh["lights"]], "negative": sh.get("negative", False),
                      "frames": len(info["frames"]), "cache": f"{sh['split']}/{info['cache_file']}", "sha256": info["sha256"],
                      "case": sh.get("case"), "has_blink": sh.get("has_blink", False)})
    man = {"schema": "eyeclean.dataset_manifest/1", "shots": shots, "missing": missing,
           "counts": {s: sum(1 for x in shots if x["split"] == s) for s in ("train", "validation", "test")},
           "negative_fraction_train": (sum(1 for x in shots if x["split"] == "train" and x["negative"]) /
                                       max(1, sum(1 for x in shots if x["split"] == "train")))}
    _write_json(run_dir / "manifests" / "dataset_manifest.json", man)
    return {"path": str(run_dir / "manifests" / "dataset_manifest.json"), "shots": len(shots), "missing": len(missing)}

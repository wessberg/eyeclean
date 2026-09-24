"""Ctrl+C during the render stage of `eyeclean generate`.

The stage is driven by a thread pool, so the interrupt arrives in a thread that is not the one
waiting on Blender.  It must still terminate the workers in flight, drop the queue, leave the
manifests on disk and propagate, rather than silently rendering on with the progress line gone.
"""

import json
import sys
from pathlib import Path

import pytest

from eyeclean.data import generate as G
from eyeclean.data.sampling import Plan
from eyeclean.render import childproc
from eyeclean.render.blender import BlenderInfo

WORKERS, JOBS = 1, 4


def _cfg(run_dir: Path) -> dict:
    return {"run_dir": str(run_dir), "data": {"catchlights": {}},
            "resources": {"workspace_cap_gib": 10_000, "min_free_disk_gib": 0}}


def _plan() -> Plan:
    ids = {f"id{i}": {"iris_family": "f", "iris_color": "c", "skin": "s"} for i in range(JOBS)}
    shots = [{"shot_id": f"sh{i}", "split": "train", "kind": "still", "identity_id": f"id{i}",
              "lights": [], "negative": False, "case": None, "has_blink": False} for i in range(JOBS)]
    return Plan(identities=ids, shots=shots, config_profile="test")


@pytest.fixture
def harness(tmp_path, monkeypatch):
    """Real `generate()` with the GPU/Blender stages replaced by cheap stand-ins."""
    run_dir = tmp_path / "run"
    (run_dir / "validation" / "pair").mkdir(parents=True)
    (run_dir / "validation" / "pair" / "pair_validation_report.json").write_text(
        json.dumps({"passed": True, "checks": {}}), encoding="utf-8")
    tex = tmp_path / "tex.png"
    tex.write_bytes(b"not really a texture")

    monkeypatch.setattr(G, "prepare_plan", lambda *a, **k: _plan())
    monkeypatch.setattr(G, "check_resources", lambda *a, **k: {})
    monkeypatch.setattr(G, "find_blender", lambda *a, **k: BlenderInfo("stub-blender", "5.2.1", None))
    monkeypatch.setattr(G, "write_identity_textures", lambda *a, **k: {"L": {"iris": str(tex)}})
    monkeypatch.setattr(G.C, "cache_valid", lambda *a, **k: False)

    childproc.reset()
    yield run_dir
    childproc.cancel_all()
    childproc.reset()


def test_interrupt_kills_the_workers_drops_the_queue_and_propagates(harness, monkeypatch):
    """KeyboardInterrupt in the main thread, raised where the real one lands: inside the loop.

    (Not via ``_thread.interrupt_main``: on Windows that does not wake a main thread already
    parked in ``as_completed``, so it would test delivery rather than the handling under test.)
    """
    run_dir = harness
    logs, pids = [], []

    def fake_run_script(script, args, log_path, blender=None, timeout=None):
        if childproc.cancelled():
            raise childproc.Cancelled("run cancelled")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        iid = Path(args[1]).stem.removeprefix("_job_")
        ok = {"returncode": 0, "seconds": 0.0, "log": str(log_path), "command": [], "optix_enabled": None}
        if iid == "id0":  # finishes at once, so the main thread reaches the conversion below
            raw = run_dir / "data" / "raw" / "sh0"
            raw.mkdir(parents=True, exist_ok=True)
            (raw / "meta.json").write_text("{}", encoding="utf-8")
            return ok
        with childproc.spawn([sys.executable, "-c", "import time; time.sleep(120)"]) as proc:
            pids.append(proc.pid)
            proc.wait()
        return ok

    def interrupt_here(*a, **k):
        raise KeyboardInterrupt  # what Ctrl+C does to the main thread, at a point it really can

    monkeypatch.setattr(G, "run_script", fake_run_script)
    monkeypatch.setattr(G.C, "convert_shot", interrupt_here)

    with pytest.raises(KeyboardInterrupt):
        G.generate(_cfg(run_dir), workers=WORKERS, log=logs.append)

    assert childproc._LIVE == {}, "a Blender child was left running"
    assert childproc.cancelled(), "the run did not latch cancelled"

    line = next(m for m in logs if m.startswith("[generate] interrupted:"))
    assert f"{len(pids)} Blender worker(s) terminated" in line
    assert "queued job(s) dropped" in line

    assert (run_dir / "manifests" / "render_log.json").exists(), "no render log after the interrupt"
    man = json.loads((run_dir / "manifests" / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert man["missing"] == [f"sh{i}" for i in range(JOBS)], "the interrupted shots must still be pending"


def test_console_ctrl_handler_kills_the_workers_without_the_main_thread(tmp_path):
    """The path that fires first on a real Ctrl+C, on a thread of Windows' choosing."""
    from eyeclean.cli.pause import KeyboardPause

    childproc.reset()
    kp = KeyboardPause(log=lambda *a: None, on_interrupt=childproc.cancel_all)
    with childproc.spawn([sys.executable, "-c", "import time; time.sleep(120)"]) as proc:
        assert kp._on_ctrl(0) is False, "the handler must not swallow the interrupt"
        assert proc.wait(30) != 0, "the console handler did not stop the worker"
    assert childproc.cancelled()


def test_a_failing_on_interrupt_never_swallows_the_interrupt():
    from eyeclean.cli.pause import KeyboardPause

    msgs = []
    kp = KeyboardPause(log=msgs.append, on_interrupt=lambda: 1 / 0)
    assert kp._on_ctrl(0) is False
    assert msgs and "interrupt cleanup failed" in msgs[0]


def test_a_later_run_in_the_same_process_is_not_blocked_by_the_cancel(harness, monkeypatch):
    """`childproc.reset()` on entry: the latch is per run, not per process."""
    run_dir = harness
    childproc.cancel_all()
    started = []

    def fake_run_script(script, args, log_path, blender=None, timeout=None):
        if childproc.cancelled():
            raise childproc.Cancelled("run cancelled")
        started.append(args)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        return {"returncode": 0, "seconds": 0.0, "log": str(log_path), "command": [], "optix_enabled": None}

    monkeypatch.setattr(G, "run_script", fake_run_script)
    monkeypatch.setattr(G.C, "convert_shot", lambda *a, **k: {"frames": []})

    res = G.generate(_cfg(run_dir), workers=WORKERS, log=lambda *a: None)
    assert len(started) == JOBS, "a stale cancel from the previous run suppressed the jobs"
    assert res["converted_now"] == 0  # the stub renders write no meta.json, so nothing converts

"""Ctrl+C must actually stop the headless workers, and they must not outlive the CLI.

Regression: `generate` ran Blender from thread-pool workers via `subprocess.run`, whose
kill-on-interrupt only fires in the thread the KeyboardInterrupt lands in - never a worker.
Interrupting the CLI left the renders running, unattended and unreported.
"""

import os
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from eyeclean.render import childproc
from eyeclean.render.blender import BlenderInfo, run_script

SLEEPER = "import time; time.sleep(120)"


@pytest.fixture(autouse=True)
def _clear_cancel():
    childproc.reset()
    yield
    childproc.cancel_all()
    childproc.reset()


def _alive(pid: int) -> bool:
    if os.name == "nt":
        out = subprocess.run(["tasklist", "/fi", f"pid eq {pid}", "/nh"], capture_output=True, text=True).stdout
        return str(pid) in out
    return Path(f"/proc/{pid}").exists()


def test_cancel_all_kills_a_child_a_worker_thread_is_waiting_on():
    """The exact shape of the bug: the waiter is not the thread that gets the interrupt."""
    started, pid, rc = threading.Event(), [], []

    def worker():
        with childproc.spawn([sys.executable, "-c", SLEEPER]) as proc:
            pid.append(proc.pid)
            started.set()
            rc.append(proc.wait())

    t = threading.Thread(target=worker)
    t.start()
    assert started.wait(20)
    assert childproc.cancel_all() == 1
    t.join(20)
    assert not t.is_alive(), "cancel_all() did not release the waiting worker"
    assert rc and rc[0] != 0
    assert not _alive(pid[0])


def test_cancel_all_refuses_further_children_until_reset():
    assert childproc.cancel_all() == 0  # nothing live, but the run is now cancelled
    with pytest.raises(childproc.Cancelled):
        with childproc.spawn([sys.executable, "-c", "pass"]):
            pytest.fail("spawned a child after cancellation")
    childproc.reset()
    with childproc.spawn([sys.executable, "-c", "pass"]) as proc:
        assert proc.wait(30) == 0


def test_run_script_refuses_to_start_and_keeps_the_previous_log(tmp_path):
    """A dropped job must not truncate the log of the run that did happen."""
    log = tmp_path / "_job_x.log"
    log.write_text("previous run", encoding="utf-8")
    childproc.cancel_all()
    with pytest.raises(childproc.Cancelled):
        run_script("generate.py", ["--job", "x"], log, BlenderInfo("nonexistent-blender", "5.2.1", None))
    assert log.read_text(encoding="utf-8") == "previous run"


def test_spawn_kills_the_child_when_the_block_raises():
    with pytest.raises(RuntimeError):
        with childproc.spawn([sys.executable, "-c", SLEEPER]) as proc:
            pid = proc.pid
            raise RuntimeError("boom")
    assert not _alive(pid)


def test_registry_is_empty_after_a_normal_exit():
    with childproc.spawn([sys.executable, "-c", "pass"]) as proc:
        proc.wait(30)
    assert childproc._LIVE == {}


@pytest.mark.skipif(os.name != "nt", reason="the kill-on-close job object is Windows-only")
def test_child_dies_with_a_hard_killed_parent(tmp_path):
    """Closing the console or killing the CLI must not leave an orphan rendering."""
    pidfile = tmp_path / "grandchild.pid"
    parent = subprocess.Popen([sys.executable, "-c", textwrap.dedent(f"""
        import sys, time
        from eyeclean.render import childproc
        with childproc.spawn([sys.executable, "-c", {SLEEPER!r}]) as p:
            open({str(pidfile)!r}, "w").write(str(p.pid))
            time.sleep(120)
    """)])
    try:
        deadline = time.time() + 60
        while not pidfile.exists() and time.time() < deadline:
            time.sleep(0.1)
        assert pidfile.exists(), "the helper never started its child"
        gc_pid = int(pidfile.read_text())
        assert _alive(gc_pid)
        parent.kill()  # no unwinding, no atexit: only the job object can save us here
        parent.wait(30)
        deadline = time.time() + 30
        while _alive(gc_pid) and time.time() < deadline:
            time.sleep(0.2)
        assert not _alive(gc_pid), f"pid {gc_pid} outlived the process that started it"
    finally:
        if parent.poll() is None:
            parent.kill()

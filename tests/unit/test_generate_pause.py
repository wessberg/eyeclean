import os
import subprocess
import sys
import threading
import time

import pytest

from eyeclean.cli.pause import KeyboardPause
from eyeclean.data.generate import progress_line


def test_progress_line_reports_percentage_rate_and_eta():
    s = progress_line(21450, 39333, 3329, active_s=3600, elapsed_s=3600)
    assert "21450/39333 shots done (54.5%)" in s
    assert "3329 converted this run, 55.5 shots/min, ETA 5h22m (1h00m elapsed)" in s
    assert "ETA n/a" in progress_line(100, 200, 0, active_s=0, elapsed_s=5)


def test_pause_blocks_wait_until_resumed():
    msgs = []
    kp = KeyboardPause(log=msgs.append)  # no console under pytest: the gate still works when toggled directly
    kp.toggle()
    t = threading.Thread(target=kp.wait)
    t.start()
    time.sleep(0.3)
    assert t.is_alive() and kp.paused
    kp.toggle()
    t.join(2)
    assert not t.is_alive() and not kp.paused
    assert kp.paused_seconds() >= 0.25  # Windows' monotonic clock ticks every ~15 ms
    assert msgs[0].startswith("[pause] PAUSED") and msgs[1].startswith("[pause] resumed after")


@pytest.mark.skipif(os.name != "nt", reason="worker suspension is Windows-only")
def test_pause_suspends_and_resumes_child_processes(tmp_path):
    out = tmp_path / "ticks.txt"
    code = ("import sys, time\nf = open(sys.argv[1], 'a')\n"
            "while True:\n    f.write('x'); f.flush(); time.sleep(0.02)\n")
    child = subprocess.Popen([sys.executable, "-c", code, str(out)])
    try:
        deadline = time.time() + 20
        while (not out.exists() or out.stat().st_size < 5) and time.time() < deadline:
            time.sleep(0.05)
        msgs = []
        with KeyboardPause(log=msgs.append) as kp:
            kp.toggle()
            # a uv venv's python.exe is a launcher that starts the real interpreter: 2 descendants
            assert int(msgs[0].split(" - ")[1].split()[0]) >= 1, msgs[0]
            time.sleep(0.2)  # let a write in flight land
            n0 = out.stat().st_size
            time.sleep(0.6)
            assert out.stat().st_size == n0, "a suspended child kept running"
            kp.toggle()
            time.sleep(0.6)
            assert out.stat().st_size > n0, "the child did not resume"
            kp.toggle()  # leave it paused: leaving the context must resume it
        n1 = out.stat().st_size
        time.sleep(0.6)
        assert out.stat().st_size > n1, "leaving the context did not resume the child"
    finally:
        child.kill()
        child.wait()

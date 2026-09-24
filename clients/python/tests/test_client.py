"""eyeclean-client without the engine: the numpy EXR reader against an independent decoder's output, and error mapping."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eyeclean_client import EyeCleanConfigError, EyeCleanInputError, read_exr, run  # noqa: E402

FIX = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "exr_zip"
META = json.loads((FIX / "fixtures.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", sorted(META))
def test_reader_matches_opencv(name):
    m = META[name]
    img = read_exr(FIX / f"{name}.exr")
    exp = np.fromfile(FIX / f"{name}.expected.bin", "<f4").reshape(len(m["channels"]), m["height"], m["width"])
    for i, c in enumerate(m["channels"]):
        assert np.array_equal(img.channels[c].astype(np.float32), exp[i]), (name, c)
    assert img.box == (0, 0, m["width"], m["height"])


def _fake_engine(tmp_path, code, lines):
    script = tmp_path / "fake_engine.py"
    script.write_text("import sys, json\n" + "".join(f"print(json.dumps({l!r}), flush=True)\n" for l in lines)
                      + f"sys.exit({code})\n", encoding="utf-8")
    return [sys.executable, str(script)]


def test_run_streams_events_and_returns_summary(tmp_path):
    lines = [{"schema": "eyeclean.progress/1", "event": "frame", "stage": "analysis", "frame": 0, "done": 1},
             {"schema": "eyeclean.progress/1", "event": "done", "summary": {"frames": 1}}]
    seen = []
    res = run({"input": str(tmp_path / "a.mp4"), "output": str(tmp_path / "o")}, engine=_fake_engine(tmp_path, 0, lines),
              on_event=seen.append, keep_events=True)
    assert res.returncode == 0 and res.summary == {"frames": 1} and [e["event"] for e in seen] == ["frame", "done"]


@pytest.mark.parametrize("code,exc", [(2, EyeCleanConfigError), (4, EyeCleanInputError)])
def test_exit_codes_map_to_exceptions(tmp_path, code, exc):
    lines = [{"schema": "eyeclean.progress/1", "event": "error", "code": code, "kind": "x", "message": "boom"}]
    with pytest.raises(exc, match="boom"):
        run({"input": "a.mp4", "output": str(tmp_path / "o")}, engine=_fake_engine(tmp_path, code, lines))


def test_cancel_by_watched_file(tmp_path):
    import threading
    import time

    script = tmp_path / "slow_engine.py"
    script.write_text(
        "import sys, json, time, os\n"
        "cf = sys.argv[sys.argv.index('--cancel-file') + 1]\n"
        "for _ in range(200):\n"
        "    if os.path.exists(cf):\n"
        "        print(json.dumps({'schema': 'eyeclean.progress/1', 'event': 'error', 'code': 7, 'kind': 'cancelled', "
        "'message': 'stopped'}), flush=True)\n"
        "        sys.exit(7)\n"
        "    time.sleep(0.05)\n"
        "sys.exit(0)\n", encoding="utf-8")
    flag = tmp_path / "release.flag"
    threading.Timer(0.3, lambda: flag.write_text("x")).start()
    from eyeclean_client import EyeCleanCancelled
    with pytest.raises(EyeCleanCancelled, match="stopped"):
        run({"input": "a.mp4", "output": str(tmp_path / "o")}, engine=[sys.executable, str(script)], cancel=str(flag))

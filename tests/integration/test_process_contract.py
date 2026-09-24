"""End-to-end external contract of `eyeclean process` (GPU; needs the default model installed).

A synthetic ZIP-compressed EXR sequence without faces: every run must emit schema-valid progress events, write a
schema-valid RUN_REPORT.json and patches/index.json, never touch the input, and stop cleanly when cancelled.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import jsonschema
import pytest
import torch

from eyeclean.io.exr import ZIP_COMPRESSION, write_exr
from eyeclean.models.registry import load_registry, locate

ROOT = Path(__file__).resolve().parents[2]
SCHEMAS = ROOT / "src" / "eyeclean" / "schemas"
N_FRAMES = 6

pytestmark = pytest.mark.skipif(locate(load_registry()["default"]) is None, reason="default model not installed")


def schema(name):
    return json.loads((SCHEMAS / f"{name}.schema.json").read_text(encoding="utf-8"))


@pytest.fixture()
def sequence(dev, tmp):
    d = tmp / "seq"
    d.mkdir()
    g = torch.Generator(device=dev).manual_seed(52)
    for k in range(N_FRAMES):
        base = torch.full((3, 64, 96), 0.18, device=dev)
        img = (base + 0.01 * torch.randn((3, 64, 96), generator=g, device=dev)).clamp(min=0)
        write_exr(d / f"{k:06d}.exr", {"R": img[0], "G": img[1], "B": img[2]}, pixel_type=1, compression=ZIP_COMPRESSION)
    return d


def test_api_run_is_schema_valid(dev, tmp, sequence):
    from eyeclean.api import process

    events = []
    before = {p.name: p.read_bytes() for p in sequence.iterdir()}
    res = process(str(sequence / "%06d.exr"), str(tmp / "out"), input_color="linear-rec709", working_color="linear-rec709",
                  fps="60000/1001", export=["patches", "report"], exr_compression="zip", on_progress=events.append)
    ev_schema = schema("progress_event")
    for ev in events:
        jsonschema.validate(ev, ev_schema)
    kinds = [e["event"] for e in events]
    assert kinds[0] == "start" and kinds.count("frame") == 2 * N_FRAMES
    assert [e["stage"] for e in events if e["event"] == "stage"] == ["analysis", "render", "finalize"]
    report = json.loads(res.report_path.read_text(encoding="utf-8"))
    jsonschema.validate(report, schema("process_report"))
    assert report["model"]["restore_mode"] == "single" and report["model"]["restore_mode_source"] == "model registry"
    assert report["fps_exact"] == "60000/1001" and report["frames"] == N_FRAMES
    index = json.loads(res.index_path.read_text(encoding="utf-8"))
    jsonschema.validate(index, schema("process_index"))
    assert {p.name: p.read_bytes() for p in sequence.iterdir()} == before


def test_api_cancel_leaves_no_report(dev, tmp, sequence):
    from eyeclean.api import Cancelled, process

    seen = []

    def stop():
        return len(seen) >= 3

    with pytest.raises(Cancelled):
        process(str(sequence), str(tmp / "out"), input_color="linear-rec709", fps=24, export=["report"],
                on_progress=lambda e: seen.append(e) if e["event"] == "frame" else None, cancel=stop)
    assert not (tmp / "out" / "RUN_REPORT.json").exists()


def _cli(args, cwd):
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    return subprocess.run([sys.executable, "-m", "eyeclean.cli.main", *args], cwd=cwd, env=env, capture_output=True, text=True,
                          timeout=600)


def test_cli_job_jsonl_and_exit_codes(dev, tmp, sequence):
    job = {"schema": "eyeclean.job/1", "input": "seq/%06d.exr", "output": "cli_out", "input_color": "linear-rec709", "fps": "24",
           "export": ["report"], "client": {"caller": "test"}}
    (tmp / "job.json").write_text(json.dumps(job), encoding="utf-8")
    r = _cli(["process", "--job", str(tmp / "job.json"), "--progress", "jsonl"], tmp)
    assert r.returncode == 0, r.stderr[-2000:]
    lines = [json.loads(line) for line in r.stdout.splitlines() if line.strip()]
    assert lines[-1]["event"] == "done" and lines[-1]["summary"]["frames"] == N_FRAMES
    rep = json.loads((tmp / "cli_out" / "RUN_REPORT.json").read_text(encoding="utf-8"))
    assert rep["job"]["resolved"]["client"] == {"caller": "test"}
    # unsupported input -> 4, with a final error event on stdout
    r = _cli(["process", str(tmp / "nope.mp4"), "--output", str(tmp / "x"), "--progress", "jsonl"], tmp)
    assert r.returncode == 4
    assert json.loads(r.stdout.splitlines()[-1])["event"] == "error"
    # invalid job -> 2
    (tmp / "bad.json").write_text(json.dumps(dict(job, colour="rec709")), encoding="utf-8")
    assert _cli(["process", "--job", str(tmp / "bad.json")], tmp).returncode == 2
    # cancel file present from the start -> 7, incomplete output
    (tmp / "STOP").write_text("", encoding="utf-8")
    r = _cli(["process", "--job", str(tmp / "job.json"), "--output", str(tmp / "c"), "--cancel-file", str(tmp / "STOP")], tmp)
    assert r.returncode == 7 and not (tmp / "c" / "RUN_REPORT.json").exists()


def test_cli_version_and_models(tmp):
    r = _cli(["--version"], tmp)
    assert r.returncode == 0 and r.stdout.startswith("eyeclean ")
    r = _cli(["models"], tmp)
    assert r.returncode == 0 and json.loads(r.stdout)["default"] in json.loads(r.stdout)["models"]

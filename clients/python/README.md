# eyeclean-client

Drive the [EyeClean](../../README.md) engine from any Python ≥ 3.9: Blender, Nuke, pipeline scripts, farm workers.
- **No torch, no CUDA:** the engine runs in its own environment as a subprocess.
- **numpy** is optional and only needed to read the output EXRs.

```bash
pip install -e clients/python            # or vendor the eyeclean_client/ folder
export EYECLEAN_CMD=/opt/eyeclean/.venv/bin/eyeclean     # or EYECLEAN_PYTHON=..., or `eyeclean` on PATH
```

```python
import threading
from eyeclean_client import run, RunOutput, EyeCleanCancelled, EyeCleanInputError

stop = threading.Event()                                   # set() it from a UI button to cancel cleanly
res = run({"input": "shot/fg/%06d.exr", "output": "shot/eyeclean",
           "input_color": "linear-rec709", "working_color": "linear-rec709", "fps": "24000/1001",
           "export": ["patches", "report"], "exr_compression": "zip"},
          on_event=lambda e: e["event"] == "frame" and print(e["stage"], e["done"], e.get("total")),
          cancel=stop)

out = RunOutput(res.output)
print(out.report["eye_status_counts"])
for fr in out.frames():
    for eye in fr.eyes:
        if eye.status == "repaired":
            corr = out.read(eye.patch)                     # R, G, B numpy (H, W): output - source, working space
            x0, y0, x1, y1 = eye.data_window               # add it to source[y0:y1, x0:x1]
```

- `run(job, engine=None, on_event=None, cancel=None, stderr_path=None, check=True)`:
  - `job` is an `eyeclean.job/1` dict or a path to a job file.
  - It returns a `RunResult` (`returncode`, `output`, `summary`).
  - When `check` is true, it raises `EyeCleanConfigError` (exit 2), `EyeCleanGpuError` (3), `EyeCleanInputError` (4),
    `EyeCleanCancelled` (7) or `EyeCleanError`.
- `RunOutput(dir)`:
  - `.report`, `.index`, `.working_color`, `.size`;
  - `.frames()` gives per-eye `status`, `reasons`, `data_window` and file paths;
  - `.read(relpath)` reads an EXR.
- `read_exr(path)` is a numpy reader for EyeClean's EXRs: scanline, none / ZIP / ZIPS, HALF / FLOAT.
- Contracts: [`docs/INTEGRATION.md`](../../docs/INTEGRATION.md).  Tests: `python -m pytest clients/python/tests`.

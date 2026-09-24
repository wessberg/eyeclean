"""eyeclean.job/1: the job file accepted by `eyeclean process --job` and eyeclean.api.process(job=...).

Precedence: explicit arguments > the job > built-in defaults.  Relative paths in a job FILE resolve against
the file's directory; in a job DICT against the current directory.  Schema: contracts/process_job.schema.json.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..config.schema import ConfigError
from ..version import PROCESS_JOB_SCHEMA

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "schemas" / "process_job.schema.json"
DEFAULTS = {"model": None, "input_color": None, "mode": "conservative", "allow_synthesis": False,
            "export": ["patches", "preview", "report"], "input_range": None, "working_color": None, "fps": None, "timestamps": None,
            "strength": 1.0, "max_frames": None, "max_region": None, "full_frame_master": False, "restore_mode": "auto",
            "exr_compression": "none", "progress_file": None, "cancel_file": None}
PATH_KEYS = ("input", "output", "timestamps", "max_region", "progress_file", "cancel_file")


def validate_job(job: dict, base: Path | None = None, where: str = "job") -> dict:
    import jsonschema

    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    try:
        jsonschema.validate(job, schema)
    except jsonschema.ValidationError as e:
        raise ConfigError(f"{where}: {e.message} at {'/'.join(map(str, e.absolute_path)) or '<root>'}") from None
    job = dict(job)
    if base is not None:
        for k in PATH_KEYS:                       # patterns (%06d, ####, globs) join like plain paths
            if isinstance(job.get(k), str) and not Path(job[k]).is_absolute():
                job[k] = str(base / job[k])
        m = job.get("model")
        if isinstance(m, str) and m.endswith(".safetensors") and not Path(m).is_absolute():
            job["model"] = str(base / m)
    return job


def load_job(path: str | Path) -> dict:
    path = Path(path)
    try:
        job = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise ConfigError(f"job file {path}: {e}") from None
    return validate_job(job, path.resolve().parent, f"job file {path}")


def merge(job: dict | None, overrides: dict) -> dict:
    """Resolved options: overrides (non-None) > job > DEFAULTS; input/output required."""
    job = job or {}
    opts = {k: (overrides.get(k) if overrides.get(k) is not None else job.get(k, d)) for k, d in DEFAULTS.items()}
    opts["input"] = overrides.get("input") or job.get("input")
    opts["output"] = overrides.get("output") or job.get("output")
    if not opts["input"] or not opts["output"]:
        raise ConfigError("process needs an input and an output (as arguments or in the job)")
    if isinstance(opts["export"], str):
        opts["export"] = [e.strip() for e in opts["export"].split(",") if e.strip()]
    opts["client"] = job.get("client")
    return opts


def job_record(opts: dict, job_file: str | None) -> dict:
    """The resolved job as recorded in RUN_REPORT.json (re-runnable with `eyeclean process --job`)."""
    rec = {"schema": PROCESS_JOB_SCHEMA, **{k: v for k, v in opts.items() if v is not None and k != "client"}}
    if opts.get("client") is not None:
        rec["client"] = opts["client"]
    return {"resolved": rec, "job_file": job_file}

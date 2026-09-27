"""Where a checkout keeps its working outputs (renders, training runs, evaluations): the RUNS directory.

By default it is ``runs/`` inside the checkout.  A checkout on a small system drive can keep it elsewhere without a
link: set ``EYECLEAN_RUNS`` to a directory, or put ``{"runs_dir": "D:/somewhere/runs"}`` in ``.eyeclean-local.json``
at the checkout root (not tracked by git).  The environment variable wins over the file.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

CHECKOUT = Path(__file__).resolve().parents[2]
LOCAL_CONFIG = CHECKOUT / ".eyeclean-local.json"


def runs_dir() -> Path:
    env = os.environ.get("EYECLEAN_RUNS")
    if env:
        return Path(env)
    if LOCAL_CONFIG.exists():
        d = json.loads(LOCAL_CONFIG.read_text(encoding="utf-8")).get("runs_dir")
        if d:
            return Path(d)
    return CHECKOUT / "runs"

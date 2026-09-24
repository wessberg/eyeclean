"""Frozen acceptance thresholds: hash contracts/acceptance.yaml before training; never lower them."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
ACCEPTANCE = ROOT / "contracts" / "acceptance.yaml"
SCHEMA = ROOT / "contracts" / "run_report.schema.json"


def acceptance_sha256() -> str:
    return hashlib.sha256(ACCEPTANCE.read_bytes()).hexdigest()


def load_acceptance() -> dict:
    return yaml.safe_load(ACCEPTANCE.read_text(encoding="utf-8"))


def freeze(run_dir: Path) -> dict:
    """Record the acceptance hash on first training use; afterwards verify it never changed."""
    p = run_dir / "manifests" / "acceptance_frozen.json"
    digest = acceptance_sha256()
    if p.exists():
        rec = json.loads(p.read_text(encoding="utf-8"))
        if rec["sha256"] != digest:
            raise RuntimeError(f"contracts/acceptance.yaml changed after it was frozen ({rec['sha256']} -> {digest}); "
                               "thresholds must not be modified after training started")
        return rec
    p.parent.mkdir(parents=True, exist_ok=True)
    rec = {"sha256": digest, "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "path": str(ACCEPTANCE),
           "thresholds": load_acceptance()}
    p.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    return rec

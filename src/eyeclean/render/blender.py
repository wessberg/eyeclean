"""Locate Blender and run project scripts headlessly in the isolated project-local profile.

Detection order: $EYECLEAN_BLENDER, `blender` on PATH, standard Windows/Linux install folders,
then a project-local portable copy under third_party/blender/.  The user's own Blender config
is never modified: BLENDER_USER_RESOURCES points at third_party/blender_user.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from . import childproc
from .childproc import Cancelled

ROOT = Path(__file__).resolve().parents[3]
USER_RES = ROOT / "third_party" / "blender_user"
BLENDER_DIR = ROOT / "blender"


class BlenderError(RuntimeError):
    pass


@dataclass
class BlenderInfo:
    path: str
    version: str
    build_hash: str | None

    def as_dict(self) -> dict:
        return {"path": self.path, "version": self.version, "build_hash": self.build_hash}


def _candidates() -> list[str]:
    c = []
    if os.environ.get("EYECLEAN_BLENDER"):
        c.append(os.environ["EYECLEAN_BLENDER"])
    w = shutil.which("blender")
    if w:
        c.append(w)
    if sys.platform == "win32":
        base = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Blender Foundation"
        if base.exists():
            for d in sorted(base.glob("Blender *"), reverse=True):
                c.append(str(d / "blender.exe"))
    else:
        for d in ("/usr/bin/blender", "/opt/blender/blender", "/snap/bin/blender"):
            c.append(d)
    for d in sorted((ROOT / "third_party" / "blender").glob("*/blender*"), reverse=True):
        c.append(str(d))
    return [p for p in c if Path(p).exists()]


def find_blender(min_version: tuple[int, int] = (4, 2)) -> BlenderInfo:
    errs = []
    for p in _candidates():
        try:
            out = subprocess.run([p, "-b", "--factory-startup", "--version"], capture_output=True, text=True, timeout=120)
            m = re.search(r"Blender (\d+)\.(\d+)\.(\d+)", out.stdout)
            h = re.search(r"build hash: (\w+)", out.stdout)
            if not m:
                errs.append(f"{p}: unparsable version output")
                continue
            ver = tuple(int(x) for x in m.groups())
            if ver[:2] < min_version:
                errs.append(f"{p}: version {ver} < {min_version}")
                continue
            return BlenderInfo(p, ".".join(map(str, ver)), h.group(1) if h else None)
        except Exception as e:  # pragma: no cover - environment dependent
            errs.append(f"{p}: {e}")
    raise BlenderError("no compatible Blender found (set EYECLEAN_BLENDER); tried: " + "; ".join(errs or ["nothing"]))


def env() -> dict:
    e = dict(os.environ)
    e["BLENDER_USER_RESOURCES"] = str(USER_RES)
    e["PYTHONDONTWRITEBYTECODE"] = "1"
    return e


def run_script(script: str, args: list[str], log_path: Path, blender: BlenderInfo | None = None,
               timeout: float | None = None) -> dict:
    """Run blender/<script> headlessly; stdout/stderr go to ``log_path``. Raises on failure.

    The child is tracked by :mod:`.childproc` so an interrupt actually terminates it: callers run
    this from thread-pool workers, which ``KeyboardInterrupt`` never reaches.  Raises ``Cancelled``
    rather than starting (or truncating the previous log of) a job the run has already given up on.
    """
    if childproc.cancelled():
        raise Cancelled(f"run cancelled; not starting Blender for {script}")
    blender = blender or find_blender()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [blender.path, "-b", "--factory-startup", "--python-exit-code", "1",
           "--python", str(BLENDER_DIR / script), "--", *args]
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8", errors="replace") as lf:
        with childproc.spawn(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env()) as proc:
            rc = proc.wait(timeout=timeout)
    dt = time.time() - t0
    text = log_path.read_text(encoding="utf-8", errors="replace")
    optix = re.search(r"OptiX devices .*'enabled': \[([^\]]*)\]", text)
    res = {"returncode": rc, "seconds": round(dt, 2), "log": str(log_path), "command": cmd,
           "optix_enabled": optix.group(1) if optix else None}
    if rc != 0:
        tail = "\n".join(text.splitlines()[-40:])
        raise BlenderError(f"Blender script {script} failed (exit {proc.returncode}); log {log_path}\n{tail}")
    return res

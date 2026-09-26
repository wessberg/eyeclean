"""`eyeclean fix`: repair a photo with no options to get right.

The format, orientation, bit depth and colour come from the file (io/stills.py); the default model and its settings
come from the registry.  The full `process` run happens in a scratch directory, and the repaired photo is written
next to the input as ``<name>.eyeclean.<ext>`` (or where ``--output`` says).  Files inside the checkout's private
footage folders are never written next to: their output goes to ``runs/fixed/``.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from ..io.errors import InputError
from ..io.stills import is_still
from .frames import is_video
from .process import PRIVATE_DIRS, ROOT, ProcessArgs, run_process


def _destination(src: Path, name: str, output: str | None, many: bool) -> Path:
    if output:
        o = Path(output)
        if many or o.is_dir() or o.suffix == "":
            o.mkdir(parents=True, exist_ok=True)
            return o / name
        o.parent.mkdir(parents=True, exist_ok=True)
        return o
    parent = src.resolve().parent
    for private in PRIVATE_DIRS:
        root = (ROOT / private).resolve()
        if os.path.normcase(str(parent)).startswith(os.path.normcase(str(root))):
            d = ROOT / "runs" / "fixed"
            d.mkdir(parents=True, exist_ok=True)
            return d / name
    return parent / name


def fix_file(path: str, model: str | None = None, output: str | None = None, many: bool = False, keep_work: bool = False,
             progress=None) -> dict:
    src = Path(path)
    if not src.is_file():
        raise InputError(f"{path}: not found")
    if is_video(path) or not is_still(path):
        raise InputError(f"{path}: `eyeclean fix` handles photos in this version; video support is in progress. For now use "
                         "`eyeclean process <video> --input-color ...`")
    work = Path(tempfile.mkdtemp(prefix="eyeclean_fix_"))
    try:
        res = run_process(ProcessArgs(input=str(src), checkpoint=model, input_color=None, mode="conservative", allow_synthesis=False,
                                      output=str(work), export=["patches", "report", "photo"], restore_mode="auto", profile=False,
                                      progress=progress))
        s, rep = res["summary"], res["report"]
        photo = work / s["photo"]
        dst = _destination(src, photo.name, output, many)
        shutil.move(str(photo), str(dst))
        still = rep["interpretation"].get("still", {})
        out = {"input": str(src), "output": str(dst), "eyes": s["eye_status_counts"], "model": s["model"],
               "colour": still.get("colour"), "decode": still.get("decode"), "format": still.get("format"),
               "size": [still.get("width"), still.get("height")], "warnings": s["warnings"], "seconds": s["seconds"]}
        if keep_work:
            out["work"] = str(work)
        return out
    finally:
        if not keep_work:
            shutil.rmtree(work, ignore_errors=True)

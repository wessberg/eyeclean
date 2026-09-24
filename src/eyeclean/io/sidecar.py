"""Input colour interpretation: explicit CLI flags > input sidecar > trustworthy stream metadata.

Sidecar: ``<input>.eyeclean.json`` beside the source (for sequences: beside the first frame or
the directory), e.g. {"schema_version": "1.0", "input_color": "panasonic-vlog-vgamut",
"input_range": "limited", "working_color": "linear-acescg", "ycbcr_matrix": "bt709"}.
Ambiguity is an error naming the flag/sidecar key to set; no transfer curve is guessed.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..color.transforms import INPUT_COLORS, WORKING_COLORS
from .errors import InputError

SIDECAR_KEYS = {"schema_version", "input_color", "input_range", "working_color", "ycbcr_matrix", "notes", "fps", "timestamps"}


def find_sidecar(input_path: str, first_frame: str | None = None) -> Path | None:
    cands = [Path(input_path + ".eyeclean.json")]
    if first_frame:
        cands.append(Path(first_frame + ".eyeclean.json"))
        cands.append(Path(first_frame).parent / "sequence.eyeclean.json")
    p = Path(input_path)
    if p.is_dir():
        cands.append(p / "sequence.eyeclean.json")
    for c in cands:
        if c.exists():
            return c
    return None


def load_sidecar(path: Path | None) -> dict:
    if path is None:
        return {}
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise InputError(f"sidecar {path}: invalid JSON ({e})") from e
    unknown = set(d) - SIDECAR_KEYS
    if unknown:
        raise InputError(f"sidecar {path}: unknown keys {sorted(unknown)}; allowed {sorted(SIDECAR_KEYS)}")
    if d.get("schema_version") not in (None, "1.0"):
        raise InputError(f"sidecar {path}: unsupported schema_version {d.get('schema_version')}")
    return d


def resolve_interpretation(*, is_video: bool, cli_color: str | None, cli_range: str | None, cli_working: str | None,
                           sidecar: dict, meta_range: str | None, meta_matrix: str | None, config_working: str) -> dict:
    """Returns the resolved interpretation plus where each value came from."""
    src = {}
    color = cli_color or sidecar.get("input_color")
    src["input_color"] = "cli" if cli_color else ("sidecar" if sidecar.get("input_color") else None)
    if not color:
        raise InputError("input colour interpretation unknown: pass --input-color "
                         f"({'|'.join(INPUT_COLORS)}) or set 'input_color' in the <input>.eyeclean.json sidecar")
    if color not in INPUT_COLORS:
        raise InputError(f"unsupported --input-color {color!r}; supported: {INPUT_COLORS}")
    working = cli_working or sidecar.get("working_color") or config_working
    src["working_color"] = "cli" if cli_working else ("sidecar" if sidecar.get("working_color") else "config")
    if working not in WORKING_COLORS:
        raise InputError(f"unsupported working colour {working!r}; supported: {WORKING_COLORS}")
    res = {"input_color": color, "working_color": working}
    if is_video:
        if color.startswith("linear-"):
            raise InputError(f"--input-color {color} describes floating-point linear data; compressed video is display/log "
                             "encoded (e.g. rec709, srgb, panasonic-vlog-vgamut)")
        rng = cli_range or sidecar.get("input_range") or "auto"
        src["input_range"] = "cli" if cli_range else ("sidecar" if sidecar.get("input_range") else "default_auto")
        if rng == "auto":
            m = (meta_range or "").upper()
            if m == "MPEG":
                rng = "limited"
                src["input_range"] = "stream_metadata"
            elif m == "JPEG":
                rng = "full"
                src["input_range"] = "stream_metadata"
            else:
                raise InputError("--input-range auto needs trustworthy stream metadata, but the stream does not signal its "
                                 f"range ({meta_range}); pass --input-range full|limited or set 'input_range' in the sidecar")
        if rng not in ("full", "limited"):
            raise InputError(f"--input-range must be auto|full|limited, got {rng!r}")
        mat = sidecar.get("ycbcr_matrix")
        if mat:
            src["ycbcr_matrix"] = "sidecar"
        else:
            mm = (meta_matrix or "").upper()
            if mm == "BT_601":
                mat, src["ycbcr_matrix"] = "bt601", "stream_metadata"
            elif mm == "BT_709":
                mat, src["ycbcr_matrix"] = "bt709", "stream_metadata"
            else:
                # the declared colour interpretations are all BT.709-matrix encodings
                mat, src["ycbcr_matrix"] = "bt709", f"implied_by_input_color:{color}"
        res.update(input_range=rng, ycbcr_matrix=mat)
    else:
        if not color.startswith("linear-"):
            if color not in ("rec709", "srgb", "rec709-display", "panasonic-vlog-vgamut"):
                raise InputError(f"unsupported colour {color} for EXR input")
        res.update(input_range="n/a", ycbcr_matrix="n/a")
        src["input_range"] = "n/a"
    res["sources"] = src
    return res

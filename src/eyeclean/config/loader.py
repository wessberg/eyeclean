"""YAML configuration loading: ``extends`` resolved relative to the containing file, mappings
deep-merged, the merged result validated against the single public schema (unknown keys rejected)."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from .schema import Config, ConfigError, validate


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load_raw(path: Path, chain: list[Path]) -> dict:
    path = path.resolve()
    if path in chain:
        raise ConfigError(f"cyclic 'extends': {' -> '.join(str(p) for p in chain + [path])}")
    if not path.exists():
        raise ConfigError(f"configuration file not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as e:
        raise ConfigError(f"{path}: YAML parse error: {e}") from e
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    parent = data.pop("extends", None)
    if parent is None:
        return data
    if not isinstance(parent, str):
        raise ConfigError(f"{path}: 'extends' must be a relative/absolute path string")
    base = _load_raw((path.parent / parent), chain + [path])
    return _deep_merge(base, data)


def load_config(path: str | Path) -> Config:
    p = Path(path)
    raw = _load_raw(p, [])
    validate(raw)
    return Config(raw=raw, source_paths=[str(p.resolve())])


def config_digest(cfg: Config | dict) -> str:
    raw = cfg.raw if isinstance(cfg, Config) else cfg
    return hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest()


def dump(cfg: Config) -> str:
    return yaml.safe_dump(cfg.raw, sort_keys=False)


def get(cfg: Config | dict, dotted: str, default: Any = None) -> Any:
    raw = cfg.raw if isinstance(cfg, Config) else cfg
    cur: Any = raw
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur

"""The external contract's CPU parts: job files, option precedence, fps parsing, the model registry, schemas."""
import json
from fractions import Fraction
from pathlib import Path

import jsonschema
import pytest

from eyeclean.config.schema import ConfigError
from eyeclean.inference.job import DEFAULTS, job_record, load_job, merge, validate_job
from eyeclean.inference.process import parse_fps
from eyeclean.io.errors import InputError
from eyeclean.models.registry import list_models, load_registry, resolve_model

SCHEMAS = Path(__file__).resolve().parents[2] / "src" / "eyeclean" / "schemas"


def test_fps_parsing_is_exact():
    assert parse_fps("60000/1001") == Fraction(60000, 1001)
    assert parse_fps(24) == 24 and parse_fps("25") == 25
    assert parse_fps(29.97) == Fraction("29.97")
    assert parse_fps(None) is None
    for bad in ("0", "-24", "abc", "30/0"):
        with pytest.raises(InputError):
            parse_fps(bad)


def test_job_file_paths_resolve_against_the_file(tmp_path):
    (tmp_path / "shot").mkdir()
    jf = tmp_path / "shot" / "job.json"
    jf.write_text(json.dumps({"schema": "eyeclean.job/1", "input": "fg/%06d.exr", "output": "../out", "fps": "60000/1001",
                              "input_color": "linear-rec709", "client": {"shot": "sh042"}}), encoding="utf-8")
    job = load_job(jf)
    assert Path(job["input"]) == tmp_path / "shot" / "fg" / "%06d.exr"
    assert Path(job["output"]).resolve() == (tmp_path / "out").resolve()
    assert job["client"] == {"shot": "sh042"}


def test_job_is_strict(tmp_path):
    with pytest.raises(ConfigError, match="Additional properties"):
        validate_job({"schema": "eyeclean.job/1", "input": "a.mp4", "output": "o", "inptu_color": "rec709"})
    with pytest.raises(ConfigError):
        validate_job({"schema": "eyeclean.job/2", "input": "a.mp4", "output": "o"})
    with pytest.raises(ConfigError, match="input_color"):
        validate_job({"schema": "eyeclean.job/1", "input": "a.mp4", "output": "o", "input_color": "bt2020"})


def test_precedence_flags_over_job_over_defaults():
    job = {"schema": "eyeclean.job/1", "input": "a.mp4", "output": "o", "strength": 0.5, "export": ["patches"]}
    o = merge(job, {"strength": None, "export": "patches,report", "input": None, "output": "other"})
    assert o["strength"] == 0.5 and o["export"] == ["patches", "report"] and o["output"] == "other"
    assert o["restore_mode"] == DEFAULTS["restore_mode"] == "auto"
    with pytest.raises(ConfigError):
        merge(None, {"input": "a.mp4"})
    rec = job_record(o, None)["resolved"]
    assert rec["schema"] == "eyeclean.job/1"
    validate_job(rec)                                  # a recorded job is itself a valid job


def test_registry_default_and_errors(tmp_path, monkeypatch):
    reg = load_registry()
    assert reg["default"] in reg["models"] and reg["schema"] == "eyeclean.models/1"
    assert reg["models"][reg["default"]]["restore_mode"] == "single"
    with pytest.raises(InputError, match="unknown model"):
        resolve_model("no_such_model")
    with pytest.raises(InputError, match="not found"):
        resolve_model(str(tmp_path / "missing.safetensors"))
    fake = tmp_path / "custom.safetensors"
    fake.write_bytes(b"not really a bundle")
    m = resolve_model(str(fake))
    assert m.name is None and m.restore_mode is None and len(m.sha256) == 64
    monkeypatch.setenv("EYECLEAN_MODEL_DIR", str(tmp_path))
    assert str(tmp_path) in list_models()["search_dirs"]


@pytest.mark.parametrize("name", ["process_job", "progress_event", "process_index", "process_report"])
def test_schemas_are_valid_json_schema(name):
    s = json.loads((SCHEMAS / f"{name}.schema.json").read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(s)


def test_progress_schema_accepts_the_documented_events():
    s = json.loads((SCHEMAS / "progress_event.schema.json").read_text(encoding="utf-8"))
    ok = [{"schema": "eyeclean.progress/1", "event": "stage", "t": 0.1, "stage": "analysis"},
          {"schema": "eyeclean.progress/1", "event": "frame", "t": 1, "stage": "render", "frame": 3, "done": 4, "total": 10,
           "eyes_repaired": 1, "eyes_tracked": 2},
          {"schema": "eyeclean.progress/1", "event": "error", "code": 7, "kind": "cancelled", "message": "x"}]
    for ev in ok:
        jsonschema.validate(ev, s)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"schema": "eyeclean.progress/1", "event": "frame", "stage": "render"}, s)

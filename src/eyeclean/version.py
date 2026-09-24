"""EyeClean version and the versions of its external contracts (docs/INTEGRATION.md)."""

__version__ = "0.3.0"

# Bumped on any incompatible change; additive fields keep the version.
PROCESS_REPORT_SCHEMA = "eyeclean.process/1"   # RUN_REPORT.json, patches/index.json
PROCESS_JOB_SCHEMA = "eyeclean.job/1"          # --job files
PROGRESS_SCHEMA = "eyeclean.progress/1"        # --progress jsonl events
MODEL_REGISTRY_SCHEMA = "eyeclean.models/1"    # models/registry.json

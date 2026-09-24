# Contributing to EyeClean

Thank you for helping.  EyeClean is judged on real footage by people watching it, so the bar is honest evidence, not a
green metric.

## Ground rules

- **Preservation is sacred.**  Nothing may change a pixel outside the final hard support.  The compositor checks every
  frame; don't weaken that check.  Tests under `tests/preservation/` must pass.
- **GPU only.**  Numerical image, video and ML work runs on CUDA.  The strict audit (`src/eyeclean/gpu/audit.py`) fails
  a run on CPU fallbacks and undeclared host↔device transfers.  New transfers need a named boundary in
  `TRANSFER_ALLOWLIST`, with a reason.
- **Contracts are versioned.**
  - `eyeclean.job/1`, `eyeclean.progress/1` and `eyeclean.process/1` (docs/INTEGRATION.md) may only gain fields within
    a version.
  - Exit codes are stable.
  - A change that breaks a consumer needs a new version and a CHANGELOG entry.
- **Never fabricate results.**  Every number in a PR description comes from a run you can point to.

## Changing a model or the gate

Follow the discipline in [docs/APPROACH.md](docs/APPROACH.md):
1. **Pre-register.**  Write the protocol, the arms, a budget-matched control and the pass bars *before* running.  Do
   not lower a bar after seeing results.
2. **Ruler first.**  Show that your metric scores the known answers correctly (the target, the source, a reference
   fix).
3. **Look.**  Include a 12-frame filmstrip at 8-12× of the eyes, before and after, and say what you see.
4. **Report the misses** as prominently as the wins.

A candidate model is promoted only after a human reviews it on real footage.

## Development

```powershell
./bootstrap.ps1
uv run pytest -q tests/unit                     # quick; many tests need CUDA and skip without it
uv run pytest -q tests                          # full (GPU; the end-to-end tests need the default model installed)
python -m pytest clients/python/tests           # the client, in any Python >= 3.9 with numpy
python tools/release_tree.py --tracked            # before every commit: scans what git would commit
```

- **Sign-off:** commits carry a `Signed-off-by:` line (`git commit -s`), certifying the
  [Developer Certificate of Origin](https://developercertificate.org/).
- **Conduct:** [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).  **Security issues:** [SECURITY.md](SECURITY.md), not a public
  issue.

- Style: match the surrounding code.  Type hints, docstrings that say *why*, no dead code.
- Private material never goes in: footage you don't have the rights to publish, local paths, credentials.  The scan
  above catches the common cases.
- New CLI flags need a job-schema key (`src/eyeclean/schemas/process_job.schema.json`), a test and a line in
  docs/INTEGRATION.md.

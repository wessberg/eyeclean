# Notes for coding agents

EyeClean removes studio-light reflections from eyes in video on an NVIDIA GPU, and leaves every pixel outside a
small hard support bit-identical to the source.  Start with `README.md`, then `docs/ARCHITECTURE.md` (where things
are) and `docs/APPROACH.md` (why they are that way).

## Rules that are easy to break

- **Preservation.**  Nothing may change a pixel outside the final hard support.  `inference/compositor.py` checks
  every frame and raises; never weaken that check.  `tests/preservation/` must pass.
- **GPU-only numerics.**  Image, video and ML computation runs on CUDA.  CPU work is limited to orchestration, scene
  construction and byte I/O.  The strict audit (`src/eyeclean/gpu/audit.py`) fails a run on CPU fallbacks and
  undeclared host-device transfers.
- **Contracts.**  `eyeclean.job/1`, `eyeclean.progress/1` and `eyeclean.process/1` (`docs/INTEGRATION.md`) only gain
  fields within a version.  Exit codes are fixed.
- **Outputs.**  A change that is not meant to alter results must leave the default model's outputs byte-identical.
  Compare a full render before and after.
- **Acceptance thresholds** (`contracts/acceptance.yaml`) are never relaxed to make a result pass.
- **Honest reporting.**  Never fabricate weights, measurements or footage results.  State failures as plainly as
  wins.
- **Working outputs go to the RUNS directory**, never to a hard-coded `runs/`.  Find it with
  `eyeclean.paths.runs_dir()`.  The default is `runs/` in the checkout; `EYECLEAN_RUNS` or `"runs_dir"` in
  `.eyeclean-local.json` (untracked) move it.

## Commands

```
uv run pytest -q tests/unit        # quick; many tests need CUDA and skip without it
uv run pytest -q tests             # full (GPU; end-to-end tests need the default model installed)
python tools/release_tree.py --tracked   # before a commit: scans every tracked file for private material
```

## Maintainers

If a `private/` directory exists, it holds the maintainers' build history and round handoff.  Read
`private/README.md` first.  It is never committed.

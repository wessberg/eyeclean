# models/

Model files go here (or in `$EYECLEAN_MODEL_DIR`, or the user model directory `eyeclean models` prints).  They are
release assets, never committed.

```
uv run eyeclean models --fetch      # download the default model and verify its SHA-256
uv run eyeclean models              # what is installed, and where EyeClean looks
```

The registry (`src/eyeclean/models/registry.json`) pins each model's file name and SHA-256, and says how to run it.
See [MODEL_CARD.md](../MODEL_CARD.md) for the default model, `eyeclean-r54`.  The previous default, `eyeclean-r48`, stays in the
registry: select it with `--model eyeclean-r48`.

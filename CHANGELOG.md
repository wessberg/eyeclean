# Changelog

## 0.4.0 (unreleased): photos

### Added

- **`eyeclean fix <photo>`:** repairs a photo with no colour flags and writes `<name>.eyeclean.<ext>` next to it.
- **Photo input to `process`:** JPEG (decoded on the GPU), PNG, WebP, TIFF and BMP.
  - The colour comes from the file: an ICC matrix/TRC profile, a PNG cICP / sRGB chunk, or sRGB when untagged
    (recorded).
  - The EXIF orientation is applied; any size and bit depth is accepted.
  - A new `photo` export writes the result in the input's format; original code values stay outside the edit
    support.
- **A CPU decode fallback** (PyAV) for image formats without a GPU decoder, recorded in the report.
- **The GPU audit** gains the declared boundaries `image_bitstream_decode`, `image_bitstream_download` and
  `cpu_decode_fallback_upload`.

### Changed

- **The default model is `eyeclean-r54`**, 238.5 MB, MIT (see MODEL_CARD.md).
  - It is `eyeclean-r48` with EyeGuard and the presence classifier fine-tuned across iris sizes of 8-160 px; the
    restorer is identical.
  - On close-up photos it detects many more catchlights: on held-out CG, 55% -> 74%; on three real phone photos,
    1 of 4 -> 4 of 4 catchlight eyes.
  - On iris images it removes more: error reduction 26% -> 44%.
  - It makes fewer false edits on bright sclera during blinks.
  - Trade-offs: on CG close-ups it edits more clean eyes (4.9% -> 7.1%), and on the development shot its fill is
    about one 8-bit level cooler (its colour guard fails; promoted by the owners after reviewing stills).
  - `eyeclean-r48` stays in the registry (`--model eyeclean-r48`).
- `--restore-mode auto` on a bundle that is not in the registry and has no five-frame restorer now selects `single`
  instead of failing.

## 0.3.0 (unreleased): the open-source release

### Changed

- **The default model is `eyeclean-r48`**, 238.5 MB, MIT.  It is the promoted model with the same guard, presence and
  restorer tensors, without the unused five-frame CNN slot, and with a scrubbed manifest.  Outputs are byte-identical
  to the development bundle's on the full development clip.
- **Bundles:** the five-frame `restore.` component is optional.  `--restore-mode temporal` on a bundle without it is
  refused with exit 4.  Overriding a model's registered restore mode adds a run warning.
- **`eyeclean doctor`** accepts any GPU architecture the installed PyTorch build supports (it required compute
  capability 10+).
- **`bootstrap`** installs only what processing needs by default; `-Training` / `--training` adds the Blender data
  pipeline.
- **Licence:** MIT now covers all of the project's own files and the released weights; `blender/` stays
  GPL-3.0-or-later.

### Added

- **`eyeclean models --fetch [NAME] [--dir DIR]`** downloads a registry model and verifies its SHA-256.  Models are
  also looked for in the checkout's `models/`.
- `SECURITY.md`, `CODE_OF_CONDUCT.md`, DCO sign-off.  `tools/release_tree.py --tracked` scans what git would commit.

### Removed

- **Retired research code no released model uses:**
  - the LaMa inpainter component;
  - the flip ensemble and refiner wrappers;
  - the temporal adapter / fuse;
  - the geometry guard;
  - the membrane, polar-copy, canonical-copy and pupil-circle fills;
  - temporal fill averaging;
  - the masked-inpainting restorer input.
- A bundle that asks for any of these is refused with a clear error instead of rendering differently.
- **The research drivers and the development record** moved out of the public tree.

## 0.2.0: the external interface

### Added

- **Versioned contracts** (docs/INTEGRATION.md): `eyeclean.job/1` job files, `eyeclean.progress/1` JSON-lines events,
  and the `eyeclean.process/1` output directory, each with a JSON Schema in `src/eyeclean/schemas/`.
- `eyeclean process`:
  - `--job`, `--progress jsonl`, `--progress-file`, `--cancel-file`;
  - `--fps` as an exact ratio (`60000/1001`);
  - `--exr-compression none|zips|zip`;
  - `--model` (a registry name or a path; `--checkpoint` still works), `--restore-mode auto`.
- **Model registry** (`eyeclean models`): the promoted model runs with no hidden flags.
- **Exit code 7** for a cancelled run.  Unsupported EXR and colour errors now exit 4, as documented.
- **`patches/index.json`** lists every tracked eye of every frame, with `status: repaired | untouched` and reasons.
  `RUN_REPORT.json` gains `schema`, `eyeclean_version`, `model`, `job`, `interpretation`, `fps_exact` and
  `eye_status_counts`.
- **EXR:** ZIP and ZIPS input and output (lossless).  The decode is bit-exact against OpenCV and the reference OpenEXR
  library, including real production EXRs.
- **`eyeclean.api.process()`** for callers in the engine's environment.
- **`clients/python`** (`eyeclean-client`): runs the engine as a subprocess from any Python ≥ 3.9.  It gives events,
  typed errors and a numpy EXR reader.
- **`integrations/blender-virtual-production`:** a reference eye pass for a virtual-production pipeline.
- **Docs:** README, INTEGRATION, ARCHITECTURE, APPROACH, MODEL_CARD, RELEASE, CONTRIBUTING, with diagrams.
- **`tools/release_tree.py`:** a private-material scan of the release tree.

### Unchanged

- **The numerical path.**  The promoted model's full-clip output through the new interface is byte-identical to the
  previous release's: 8,936 EXR layers over 1,200 frames, and every index entry.

## 0.1.0: research prototype

- The engine, the synthetic data pipeline and 51 development rounds.
- The promoted model is `best_single_r48`.

# EyeClean in blender-virtual-production-experiment

This is how EyeClean replaces that product's homegrown eye removal (`eye_removal_gpu.py`, `eye_soft_removal.py`,
`eye_gpu.py`, `eye_sequence_atlas.py`, `eye_reflection_repair.py`).  It keeps every downstream stage working unchanged.
Nothing here has been merged into that product; the adapter and this plan are for its maintainers.

## Why out of process

| | blender-virtual-production | EyeClean engine |
|---|---|---|
| Python | 3.13 (uv venvs, no pip) | 3.12 |
| torch / CUDA | 2.8.0 + cu128 (pipeline venv) | 2.14 + cu130, PyNvVideoCodec |
| runs as | standalone CLI scripts, one at a time | a CLI per shot |

The two cannot share an interpreter.  So EyeClean lives in its own environment under `dependencies/`, following the
product's convention, and the product calls it as a subprocess through the dependency-free
[`eyeclean-client`](../../clients/python).

## Install (once)

```powershell
git clone <eyeclean repo> dependencies/eyeclean
cd dependencies/eyeclean; ./bootstrap.ps1; cd ../..
dependencies\eyeclean\.venv\Scripts\eyeclean.exe models --fetch --dir dependencies\eyeclean\models   # or %LOCALAPPDATA%\eyeclean\models
$env:EYECLEAN_CMD = "$PWD\dependencies\eyeclean\.venv\Scripts\eyeclean.exe"
$env:EYECLEAN_MODEL_DIR = "$PWD\dependencies\eyeclean\models"
copy dependencies\eyeclean\integrations\blender-virtual-production\eyeclean_pass.py scripts\
uv pip install --python dependencies/CorridorKey/.venv -e dependencies/eyeclean/clients/python
```

## Run (replaces `eye_removal_gpu.py`)

```powershell
dependencies\CorridorKey\.venv\Scripts\python.exe scripts\eyeclean_pass.py --shot music_videos/<video>/shots/<shot> --name eyeclean_r48
```

`scripts/eyeclean_pass.py`:
1. reads `<shot>/shot.json`, taking the frame rate exactly as the stored ratio (`60000/1001`), so there is no float
   drift;
2. runs EyeClean on `key/fg/%06d.exr`.  These are ZIP-compressed HALF files, which EyeClean reads natively; its decode
   matches OpenCV's bit for bit on this shot's frames.  It uses `input_color = working_color = linear-rec709`, so
   every correction is in the fg's own linear space;
3. streams progress as timestamped log lines; the full JSON-lines stream is in the engine's output;
4. writes `<shot>/composite/<name>/`:
   - `engine/`: EyeClean's float outputs (the exact result);
   - `source/`, `removal/`, `weight/` `%06d.png` and `removal.json`: the product's existing eye-pass layout, so the
     existing consumers keep working: `composite_ratio.py --eye-removal <name>`, `sclera_glint_removal.py --run <name>`,
     `import_overlap_eyes.py`, `validate_eye_gpu.py`, the `audit_r3x_*.py` scripts and `review_*`.
   - `removal.json` keeps the keys those consumers read: `rows[].frame`, `source_frame`, `source_sha256` (the fg EXR's
     hash, as EyeClean recorded it before processing), `changed_pixels`, and `eyes[].side / repaired / reason`.  It
     adds the EyeClean version, model SHA-256 and per-eye status;
5. writes `<shot>/logs/<name>.json` as evidence: command, engine version, model, timings.

**Eye sides:**
- This product names eyes from the subject's view (`limbus_track.json`: "left" is image-right).  EyeClean names them
  from the image's (`image_left` / `image_right`).
- The adapter matches each EyeClean eye to the nearest limbus-track centre of the frame.  Without a track it maps
  `image_right` → `left`.

## Do better than 8 bits: apply the float corrections

The PNG layout costs 8-bit sRGB precision inside the eye, as the old pass did.  The corrections in `engine/` are exact.
Add them in float, in `composite_ratio.py`, next to `--eye-removal`:

```python
from eyeclean_client import RunOutput

if args.eye_patches:                                   # new flag: path to <run>/engine
    eyeclean_run = RunOutput(args.eye_patches)
    assert eyeclean_run.working_color == "linear-rec709"
    eyeclean_frames = {fr.frame: fr for fr in eyeclean_run.frames()}
...
fr = eyeclean_frames.get(fi) if args.eye_patches else None
for eye in (fr.eyes if fr else []):
    if eye.status == "repaired":
        x0, y0, x1, y1 = eye.data_window
        c = eyeclean_run.read(eye.patch).channels
        fg[y0:y1, x0:x1] += np.stack([c["R"], c["G"], c["B"]], -1).astype(np.float32)   # exactly 0 outside the support
```

**Gate the new reflection on the removal.**  `docs/iris-prior-art.md` intends it: a failed eye keeps its original
reflection and gets no added one.  Today `composite_ratio.py` adds the corneal pass wherever a limbus fit exists.  With
EyeClean, add it only where the frame's eye has `status == "repaired"`.

## Keep

- `face_track.py`, `eye_geometry.py`, `limbus_fit.py` / `limbus_trust.py` and `eyelid_fit.py`.  They still anchor and
  matte the new scene reflection (`stress_proxy.py --corneal`, `composite_ratio.py --corneal-matte`).
- EyeClean does its own tracking for the removal.  A later version may accept these tracks as input (see
  `docs/INTEGRATION.md` §10).

## Accepting the switch

This follows the product's own method; the artist's verdict is the bar:
1. Run `eyeclean_pass.py` on the full development shot.
2. Run `audit_r35_defects.py`, `audit_r37_flicker.py` (p95, max) and `review_r35_eyes.py` against the accepted runs
   `r40_floorB` and `r42_wet_removal`.
3. Run `review_video.py --compare`.
4. Get the artist's verdict.
5. Check `sclera_glint_removal.py --run <name>` on top: EyeClean does not target scleral or tear-lake glints, so keep
   that stage until it is shown redundant.
6. Never run EyeClean at the same time as Blender or another GPU stage.

## The artist's own footage

The artist's own footage is 5776×3056 ProRes 422 HQ, V-Log/V-Gamut, 24p, 1.6× anamorphic:
- **ProRes** is not decodable on NVDEC.  The product's ingest already produces linear EXR (`key/fg`), which EyeClean
  reads, so this is not a blocker.
- **V-Log** input is supported (`panasonic-vlog-vgamut`) if the product ever hands EyeClean the log plate directly.
- **Iris size:** irises will be several times larger than the development shot's (~12 px).  EyeClean scales its crop by the
  measured iris diameter, so it is resolution-independent by design.  It is untested at that size.
- **Anamorphic squeeze:** in stored (squeezed) pixels the iris is elliptical.  EyeClean has not been validated on
  squeezed eyes.  Process the de-squeezed plate, or test a shot first.

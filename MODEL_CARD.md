# Model card: `eyeclean-r48`

| | |
|---|---|
| file | `eyeclean-r48.safetensors`, 238.5 MB, FP32 |
| SHA-256 | `1505f69df6b57bb51a9031294dbe410d51a178984f22868e2010588558cced1f` |
| licence | MIT (see Licences below) |
| promoted | 2026-09-24, on a human review of a full real clip (development round 48) |
| get it | `eyeclean models --fetch` (verifies the SHA-256), or download it from the release page into a model folder |
| run with | `eyeclean process` (the registry sets `restore_mode single`); gate calibration is inside the bundle |
| components | EyeGuard 1.78M · presence classifier 0.34M · EyeRestoreTX 57.5M |

The file carries the same tensors as the development bundle it was promoted as (`best_single_r48`, SHA-256
`d745caf3…`).  That bundle's unused five-frame CNN slot and its development notes are left out.  On the full
development clip its outputs are byte-identical.

## What it does

- **The task:** finds eyes in video and removes the *designated* specular reflection of a studio light (softbox, ring,
  window, panel) from the cornea.  It reconstructs the iris and pupil underneath, inside a hard support that a
  conservative gate derives.
- **Preservation:** pixels outside that support are bit-identical to the decoded source, verified on every frame.
- **Abstention:** eyes it is unsure about are left untouched and reported with a reason.

## Intended use

- **For:** post-production and virtual production on footage you have the rights to.  The typical case: removing the
  on-set lights' reflections before relighting a performer into a new environment, or before adding the new scene's
  reflection.
- **Delivery:** the outputs are float EXR layers for a compositor to review and apply.  They are not an automatic final.
- **Not intended for:** altering footage to deceive (evidence, news, identity), or for use without the consent of the
  people filmed.

## Lineage

1. **EyeGuard and the presence classifier:** trained on the synthetic CG bank; they have been unchanged in the bundle
   since round 19.
2. **EyeRestoreTX:** trained for 10,000 updates on a 50/50 mix of CG pairs and generated iris images (round 22).
   - The targets remove only the designated catchlight.
   - Two invariance fine-tunes followed, on two-frame CG pseudo-clips (round 45: 3,000 updates; round 46: 2,000
     updates at consistency weight 10).
   - The loss penalises how much the restorer's *error* changes between frames that differ only in sampling phase,
     noise, codec and tracker jitter.
3. **Gate:** inherited from `best_single_r43`.
   - The restorer input is softened by 0.5 native px.
   - The reflection mask is a temporal union over ±2 frames, and presence hysteresis is on (round 24).

## Training data

- **Synthetic only.**
  - Blender 5.2 Cycles renders of MPFB 2 / MakeHuman heads, with CC0 skins, eyelids, blinks, gaze and procedural
    lights.
  - Each render comes in exact pairs, with and without the designated reflection.
  - A camera simulation adds sensor noise, codec and sub-pixel sampling.
- **Iris textures and the iris image bank** come from `fatdove/Iris_Database`, whose images are diffusion-generated
  (see Licences).
- **No real footage** was used for training.

## Evaluation

**One real shot: the development shot.**
- The footage: 1,200 frames, 1080p 59.94, Rec.709, one presenter in front of a softbox, irises about 12 px across.
- **The same shot selected the checkpoint and tuned the gate.**  These are development results, not a held-out test.
- Preservation: 0 bits changed outside the support on all 1,200 frames.  Eyes were edited on 1,149 frames.
- Temporal stability: measured against a hand-made reference fix of the same shot, it removes 78.4% / 87.2% (two
  scorers) of the excess frame-to-frame change of its predecessor.  Every guard (brightness, colour, texture, damage)
  passes.
- Human review of the full clip: "like 95% perfect".

**Held-out synthetic test sets** (CG pairs and iris images, compared with the round-22 restorer):
- CG fill error −13%; pupil-edge error −17%;
- iris fill error −25%;
- false edits on clean eyes, 99th percentile: −63% on CG, −20% on iris images;
- frame-to-frame error change on held-out pseudo-clips: −28%.
- **The CG fill is softer:** gradient-energy ratio 0.91, against 1.05 for the round-22 restorer.

**Speed:** about 2.3 frames per second at 1080p on an RTX 5080, end to end (two decode passes); about 3.3 GB of GPU
memory.

## Limitations

**It has been validated on one real shot.**  Test on your own footage before relying on it.

**Seen on the development shot:**
- **Softness:** the rebuilt pupil and iris are softer than the real ones around them.
- **Micro-variation:** zoomed in, the fill shows slight frame-to-frame changes in colour and structure.  They are slow,
  low-frequency variations; at normal size they are hard to see.
- **Blinks:** as the lid closes, the fill region can collapse and drop out for a couple of frames (a "blink-edge
  flash").
- **Head turns:** on strong turns the guard can mistake bright sclera along the upper lid for the catchlight, and fill
  it.

**Untested:**
- large irises (4K and above), anamorphic (squeezed) plates;
- dark or light irises unlike the development shot's brown;
- glasses and contact lenses;
- multiple people;
- HDR transfer functions.

**Not targeted:**
- reflections on the sclera or tear line;
- more than 4 faces per frame;
- eyes smaller than about 8 px iris diameter (these abstain).

**Environment:** NVIDIA GPUs only; tested on Windows 11 with an RTX 5080.

## Licences

**Code:** EyeClean's code is MIT.  See `LICENSE` for the parts under GPL-3.0 and BSD-3.

**Weights: MIT**, with attribution of the training data:
- **The iris data.**  The restorer, and EyeGuard's CG iris textures, were trained on images from
  [`fatdove/Iris_Database`](https://huggingface.co/datasets/fatdove/Iris_Database) (revision `1c80e7db…`).
  - Its dataset card declares `license: mit` in its metadata.
  - Its text also says "available for research purposes only", and the pinned revision has no LICENSE file.
  - The images are diffusion-generated, from a model trained on photographs of volunteers who consented to that
    research.
  - The owners release these weights under MIT on the strength of the declared licence.  If your use depends on
    certainty about the training data's terms, contact the dataset's authors.
- **Rendering assets.**  MPFB 2 is GPL-3.0; it was used only to build scenes, and none of its code or data is in
  the weights.  The MakeHuman assets and skins are CC0.
- **The face detector.**  The S3FD and 2DFAN4 weights that `eyeclean` downloads at install carry no explicit
  licence.  They are not part of this model and are not redistributed; each user fetches them from their upstream
  source.


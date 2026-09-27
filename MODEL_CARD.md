# Model card: `eyeclean-r56b`

| | |
|---|---|
| file | `eyeclean-r56b.safetensors`, 238.5 MB, FP32 |
| SHA-256 | `b725366ef2ea3778631892bcbc67270dc09e99721f2ff1069a8a7978f27fa79d` |
| licence | MIT (see Licences below) |
| promoted | 2026-09-27, on a human review of lossless A/B stills (iris images, CG, phone photos and the development shot; development round 56), over two failed development checks (see Evaluation) |
| get it | `eyeclean models --fetch` (verifies the SHA-256), or download it from the release page into a model folder |
| run with | `eyeclean process` (the registry sets `restore_mode single`); gate calibration is inside the bundle |
| components | EyeGuard 1.78M · presence classifier 0.34M · EyeRestoreTX 57.5M |

It is `eyeclean-r56` (SHA-256 `b1a3a7fc…`, still in the registry) with EyeGuard's reflection channel also trained on
the iris image bank, whose images are generated photographs with a known highlight mask.  That is closer to real
photos than the CG renders.  Presence classifier, restorer and gate are r56's, byte for byte.  It removes large, soft
glare on close-ups and photos much better than r56; it also tends to remove small wet-eye sparkles on the sclera next
to a lid.

`eyeclean-r56` was `eyeclean-r54` (SHA-256 `bfdca4a2…`) with EyeGuard fine-tuned to mark more of the catchlight.  The fine-tune is distilled to r54's guard everywhere else, so its other outputs stay
close to r54's.  The gate lets the reflection hole reach a lid edge.  The presence classifier, its threshold and the
restorer are r54's, byte for byte.  The change targets close-ups and photos, where catchlights were often only
partly removed.  `eyeclean-r54` itself is `eyeclean-r48` (SHA-256 `1505f69d…`) with the guard and presence classifier
fine-tuned across iris sizes; both remain in the registry.

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

1. **EyeGuard and the presence classifier:** trained on the synthetic CG bank (round 19), then fine-tuned from those
   weights with eye crops resampled to iris diameters of 8-160 px (round 54).
   - EyeGuard (round 54): 6,000 steps at batch 16; the final EMA weights are used, with no checkpoint selection.
   - Presence classifier (round 54): 3,000 steps at batch 16; the checkpoint with the best validation detection at
     <= 1% false alarms is kept.  Unchanged since.
   - EyeGuard again (round 56, `eyeclean-r56`): 4,000 steps at batch 16 on the same data.  The loss adds a recall
     term on the reflection channel (Tversky, weighting misses 0.6 against false marks 0.4, at 0.3) and distillation
     to the round-54 guard everywhere except the catchlight (weight 1.0).  Final EMA weights.
   - EyeGuard again (round 56, `eyeclean-r56b`): 4,000 steps, each on 8 CG crops (the same loss) and 8 iris-bank
     images.  On the iris images the reflection channel is supervised by the stored highlight mask (BCE + the same
     recall term), and the other channels are distilled to r56's guard, because the bank has no eye labels.  Final
     EMA weights.
2. **EyeRestoreTX:** trained for 10,000 updates on a 50/50 mix of CG pairs and generated iris images (round 22).
   - The targets remove only the designated catchlight.
   - Two invariance fine-tunes followed, on two-frame CG pseudo-clips (round 45: 3,000 updates; round 46: 2,000
     updates at consistency weight 10).
   - The loss penalises how much the restorer's *error* changes between frames that differ only in sampling phase,
     noise, codec and tracker jitter.
3. **Gate:** inherited from `best_single_r43` through `eyeclean-r48`, with two changes:
   - the presence threshold is the round-54 classifier's validation 1%-false-alarm threshold (0.624);
   - `hole_skip_margin: 2` (round 56): inside the guard's own eye surface, the reflection hole is exempt from the
     lid margin, so a catchlight lying against a lid is filled to its edge.
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
- **The same shot selected `eyeclean-r48`'s restorer checkpoint and tuned its gate.**  The round-54 and round-56
  guard fine-tunes were trained and selected on CG only.  These are still development results, not a held-out test.
- Preservation: 0 bits changed outside the support on all 1,200 frames.  Eyes were edited on 1,148 frames
  (2,247 eye-frames repaired; `eyeclean-r56`: 1,138 frames, 2,232 eye-frames; `eyeclean-r54`: 1,134 / 2,229).
- Temporal stability: measured against a hand-made reference fix of the same shot, it removes 76.6% / 88.2% (two
  scorers) of the excess frame-to-frame change of its predecessor (`eyeclean-r56`: 77.7% / 89.6%; `eyeclean-r48`:
  78.4% / 87.2%).
- The brightness and damage guards pass.  **Texture misses its development limit by a hair:** 0.896 against 0.90
  on one of two frame ranges (`eyeclean-r56`: 0.906).  **The colour guard does not pass, as for `eyeclean-r54` and
  `eyeclean-r56`:**
  - inside the catchlight, the fill's colour distance to the hand-made reference is 1.28x / 1.43x the baseline's,
    against a limit of 1.25x (`eyeclean-r56`: 1.32x / 1.51x; `eyeclean-r54`: 1.34x / 1.53x; `eyeclean-r48`: 1.0x);
  - `eyeclean-r54`'s fill measured about one 8-bit level darker and slightly cooler than `eyeclean-r48`'s.  The owners
    promoted r54 over this check after reviewing lossless stills.
- On blinks, `eyeclean-r54` stopped greying out the bright sclera under the lid in the frames where `eyeclean-r48` did.
- At normal size its output on the development shot is practically indistinguishable from `eyeclean-r56`'s.  At 10x it
  also removes a small glint on the sclera beside the iris that the hand-made reference keeps.
- Human review: `eyeclean-r56` over `eyeclean-r54`: "I think g4b is better, so let's promote it"; `eyeclean-r56b` over
  `eyeclean-r56`: "G5 looks better on the irises, stills, and photos", promoted after a look at the development shot.

**Held-out synthetic tests, `eyeclean-r56` -> `eyeclean-r56b`** (identical samples; same measures as below):

| set | catchlights detected | removal | still visible | clean eyes edited |
|---|---|---|---|---|
| CG, close-up scale | 84.4% -> 85.6% | 0.522 -> 0.521 | 52.5% -> 52.4% | 19.6% -> 19.2% |
| CG, development-shot scale | 74.7% -> 74.7% | 0.341 -> 0.346 | 67.5% -> 66.8% | 6.0% -> 6.0% |
| iris images | 82.2% -> **85.6%** | 0.572 -> **0.636** | 49.0% -> **43.4%** | 5.9% -> 4.9% |

- A large, soft glare on a pale iris (the stress case): removal 0.70 -> 0.77; catchlight pixels still visible 53% ->
  27%.
- **Two development checks failed, and the owners promoted it over them:** damage outside the iris on CG close-ups
  rises (sclera 0.099 -> 0.128, lids 0.105 -> 0.153).  That comes from a few eyes: the median eye is unchanged and
  3 of 250 carry 86% of the increase.  And the development-shot texture limit above.

**Held-out synthetic tests, `eyeclean-r54` -> `eyeclean-r56`** (identical samples).  Removal is the share of the
catchlight's error removed, counting missed eyes as 0.  "Still visible" is the share of catchlight pixels left more
than 0.03 brighter (display luma) than the truth:

| set | catchlights detected | removal | still visible | clean eyes edited |
|---|---|---|---|---|
| CG, close-up scale (iris 60-150 px; 250 catchlight / 250 clean eyes) | 75.6% -> 84.4% | 0.461 -> 0.522 | 57.9% -> 52.5% | 10% -> **19.6%** |
| CG, development-shot scale (iris 12-36 px; 162 / 250) | 58.0% -> 74.7% | 0.305 -> 0.341 | 70.7% -> 67.5% | 1.2% -> **6.0%** |
| iris images (298 / 102) | 76.2% -> 82.2% | 0.465 -> 0.572 | 57.9% -> 49.0% | 1.0% -> **5.9%** |

- **Two development checks failed, and the owners promoted it over them:**
  - more clean eyes get some edit (above).  In a review of such eyes, most edits were invisible; one CG eye showed
    a darker smudge over the pupil at 6x;
  - more error on the eye next to the catchlight: on CG close-ups, the whole-eye error reduction fell from 0.190 to
    0.097.

**Earlier step, `eyeclean-r48` -> `eyeclean-r54`** (round 54 samples; the error reduction is over the whole eye):

| set | catchlights detected | error reduction | clean eyes edited |
|---|---|---|---|
| CG, close-up scale (iris 60-150 px; 489 catchlight / 465 clean eyes) | 54.8% -> 74.4% | 12.0% -> 15.3% | 4.9% -> **7.1%** |
| CG, development-shot scale (iris 12-36 px; 162 / 315) | 56.2% -> 58.0% | 1.9% -> 14.2% | 2.9% -> 1.3% |
| iris images (299 / 101) | 55.2% -> 74.6% | 26.3% -> 43.6% | 0% -> 0% |

- Three real close-up phone photos: catchlight eyes repaired 1 of 4 -> 4 of 4.

**The restorer, unchanged from `eyeclean-r48`, on held-out sets** (CG pairs and iris images, compared with the round-22
restorer):
- CG fill error −13%; pupil-edge error −17%;
- iris fill error −25%;
- false edits on clean eyes, 99th percentile: −63% on CG, −20% on iris images;
- frame-to-frame error change on held-out pseudo-clips: −28%.
- **The CG fill is softer:** gradient-energy ratio 0.91, against 1.05 for the round-22 restorer.

**Speed:** about 2.3 frames per second at 1080p on an RTX 5080, end to end (two decode passes); about 3.3 GB of GPU
memory.

## Limitations

**It has been validated on one real shot.**  Test on your own footage before relying on it.

**Close-up photos are still a weak spot.**  At the scale of a phone close-up (irises of roughly 60–150 px):
- on held-out CG it detects about 84% of catchlights and removes about half of each catchlight on average (counting
  misses as 0); only about a quarter of eyes get 80% or more removed;
- large, soft glare over a pale iris is now mostly removed; a thin band can remain against a lid, and the rebuilt
  area is smoother than real iris texture;
- on three real close-up phone photos it repaired all four catchlight eyes.

**Small wet-eye sparkles on the sclera next to a lid are often removed with the catchlight.**  A hand fix would keep
them; this model treats them as part of the reflection.

**More false edits than `eyeclean-r54`:** on held-out sets it edits some pixels of 6-20% of eyes that have no
catchlight (r54: 1-10%).  Most such edits were invisible in review; one showed a darker smudge over the pupil.
Improving removal without them is ongoing work.

**Seen on the development shot:**
- **Softness:** the rebuilt pupil and iris are softer than the real ones around them.
- **Micro-variation:** zoomed in, the fill shows slight frame-to-frame changes in colour and structure.  They are slow,
  low-frequency variations; at normal size they are hard to see.
- **Blinks:** as the lid closes, the fill region can collapse and drop out for a couple of frames (a "blink-edge
  flash").
- **Head turns and blinks:** the guard can mistake bright sclera along a lid for the catchlight, and fill it.  It
  does so less often than `eyeclean-r48` did on the development shot.
- **Colour:** the fill is slightly cooler than `eyeclean-r48`'s (see Evaluation).

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


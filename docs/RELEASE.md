# Releasing EyeClean

This is the maintainers' checklist for publishing the repository and a model.

## 1. Decisions

| # | decision | status |
|---|---|---|
| D1 | **Licence of the model weights** | **MIT**, the owners' decision (2026-09-24).  The iris dataset the restorer was trained on (`fatdove/Iris_Database`) declares `license: mit` in its card metadata.  Its card text also says "research purposes only", and the pinned revision has no LICENSE file.  The model card attributes the dataset and states this. |
| D2 | **Face detector / landmark weights** (S3FD, 2DFAN4) | Fetched at install from upstream, never redistributed; documented in THIRD_PARTY_NOTICES.md.  They have no explicit licence.  Later options: accept caller-supplied eye tracks (§4), or switch to a permissively licensed detector. |
| D3 | **Licence scope** | MIT for all of the project's own files (code, docs, tools, client, integrations), GPL-3.0-or-later for `blender/` (`LICENSE`).  The owner can revisit this before the first push, e.g. Apache-2.0 for an explicit patent grant. |
| D4 | **Illustrations** | Schematics only.  The only real footage EyeClean was validated on is private.  A CG demo clip with CC0 textures is the way to show a before / after. |
| D5 | **Name** | Checked 2026-09-24: `eyeclean` and `eyeclean-client` are free on PyPI.  Two small unrelated GitHub repositories use the name (EEG and eye-tracking data cleaning).  A UK workforce-management app is called "Eye-Clean", in a different field.  No trademark registration was found in a web search; that is not a legal clearance. |
| D6 | **Where the weights live** | A GitHub Release asset on the public repository.  After uploading, set `url` in `src/eyeclean/models/registry.json` so that `eyeclean models --fetch` works. |

## 2. What is public

The repository is what git tracks.  `.gitignore` is an **allowlist**: a new top-level file or folder stays private
until it is added there.  These are never tracked:

| local only | what it is |
|---|---|
| `private/` | the maintainers' build history: round documents, build log, measurement scripts, the original specification, retired code |
| `runs/` | renders, checkpoints, caches and reviews (hundreds of GB, much of it derived from private footage) |
| `own-footage/`, `inputs/` | footage |
| `third_party/`, `.venv/` | fetched by the bootstrap |
| `tools/*` except `release_tree.py` and `README.md` | the maintainers' workstation GPU console |

`python tools/release_tree.py --tracked` scans every file git would commit for private material:
- the development footage;
- local absolute paths;
- personal names and e-mail addresses;
- credentials;
- files over 2 MB.

It must print **CLEAN** before every commit.  Reviewed exceptions are listed, with reasons, in the script.

## 3. Publishing checklist

- [x] D1-D6 decided (§1).
- [x] The private build history moved out of the public tree; retired research code removed; a snapshot of the
      pre-cleanup tree kept privately.
- [x] Release bundle `eyeclean-r48.safetensors` (238.5 MB).
  - It carries the same guard / presence / restore_single tensors as the development bundle, without its unused
    five-frame CNN slot.
  - Its manifest is scrubbed: no local paths, no review notes.
  - Its own registry entry, which is the default.
  - A byte-exact ruler: full development clip, identical outputs.
- [x] `doctor` accepts any GPU architecture in the installed torch build.  The build-demo acceptance contract still
      names its own target hardware.
- [x] `tools/release_tree.py --tracked` CLEAN; a local git repository with one commit, no remote.
- [x] `SECURITY.md`, `CODE_OF_CONDUCT.md` (Contributor Covenant 2.1), DCO sign-off in `CONTRIBUTING.md`.
- [ ] Create the GitHub repository and push.  Then, in its settings:
  - enable **private vulnerability reporting** (SECURITY.md and the Code of Conduct rely on it);
  - protect the default branch.
- [x] Release bundle `eyeclean-r54.safetensors` (238.5 MB), the default since 2026-09-26: `eyeclean-r48` with the
      guard and presence classifier fine-tuned across iris sizes; the restorer tensors are identical.  Its own registry
      entry; a byte-exact ruler against the promoted development file on the full development clip.
- [x] Release bundle `eyeclean-r56.safetensors` (238.5 MB), the default since 2026-09-26 (later the same day as
      r54): r54 with the guard fine-tuned for recall and the hole exempt from the lid margin; presence and restorer
      identical.  Its own registry entry; the full development clip re-rendered with it as the reference.
- [ ] Upload `eyeclean-r56.safetensors` (and, optionally, `eyeclean-r54` / `eyeclean-r48`) as release assets on the
      first public tag.  Set each `url` in the registry, commit, and check `eyeclean models --fetch` on a clean machine.
- [ ] A fresh clone on a clean machine: `bootstrap`, `eyeclean models --fetch`, `pytest`, one `eyeclean process`.
- [ ] Linux: bootstrap and tests (untested so far).
- [ ] README renders on GitHub: SVGs and relative links.
- [ ] Optionally, a secret scanner (e.g. `gitleaks`) over the history before the first push.

## 4. Worth doing next (in order of value)

1. **A curated training recipe** in `tools/`.  It covers the CG bank render, the iris bank, the 10,000-update
   EyeRestoreTX run, the two invariance fine-tunes, and bundle assembly and calibration.
   - Paths come from config, and it ships with a small CC0 demo dataset.
   - It is the one part of the released model's lineage that is not yet public.
2. **External eye tracks:** consistency with a host pipeline, and no detector licence (D2).
3. **Frame ranges** (start/end) and per-frame `max_region` sequences.
4. **EXR:** PIZ and DWAA/B input; a `chromaticities` attribute on outputs; OCIO config awareness.
5. **Host plug-ins:** a Nuke gizmo / Python panel and a Blender add-on on top of `eyeclean-client`; a Linux container
   image for farms.
6. **Runs:** resumable runs, and device selection (`--device`).

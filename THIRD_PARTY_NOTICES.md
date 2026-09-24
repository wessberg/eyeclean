# Third-party components

All downloads are public and ungated; hashes are pinned in `third_party/assets.lock.json`. Nothing is uploaded or
redistributed by this repository.

| Component | Version / hash | License | Use |
|---|---|---|---|
| PyTorch, torchvision | 2.14.0+cu130, 0.29.0+cu130 (uv.lock) | BSD-3-Clause | CUDA tensors, training, inference, NMS |
| NVIDIA PyNvVideoCodec | 2.2.2 | MIT (package metadata) | NVDEC/NVENC, FFmpeg demux/mux |
| nvidia-cuda-runtime-cu12 | 12.9.79 | NVIDIA proprietary redistributable (CUDA EULA) | `cudart64_12.dll` required by PyNvVideoCodec |
| Blender | 5.2.1 LTS (existing install, hash 9e2066aef7ef) | GPL-2.0-or-later | Cycles/OptiX rendering (not modified) |
| MPFB (MakeHuman plugin for Blender) | 2.0.17, sha256 4f0a879d…a87 (published) | GPL-3.0-or-later | parametric head/lids/rig, project-local profile |
| MakeHuman system assets | asset pack, sha256 b542127a…107 | CC0-1.0 | base mesh, eyebrows, eyelashes, system skins |
| MakeHuman skins01 / skins02 | asset packs | CC0-1.0 | natural skin textures |
| face-alignment code | master archive (reference only) | BSD-3-Clause (c) 2017 Adrian Bulat | S3FD/FAN layer definitions adapted in `src/eyeclean/tracking/networks.py` |
| S3FD weights `s3fd-619a316812.pth` | sha256 619a3168…543 | no explicit license published | face detection, local research use only; converted once to safetensors with `weights_only=True` |
| 2DFAN4 weights `2DFAN4-11f355bf06.pth.tar` | sha256 11f355bf…4aa | no explicit license published | 68-point landmarks, local research use only; converted once to safetensors |
| PyAV (av) | uv.lock | BSD-3-Clause (FFmpeg LGPL) | only to fabricate a ProRes rejection test fixture |
| colour-science, OpenEXR (dev) | uv.lock | BSD-3-Clause | offline reference tests only (never in the processing path) |
| fatdove/Iris_Database | revision `1c80e7dbdab7cb7df09260615dc8a49c3723c1de`, acquired 2026-09-13 | card metadata: MIT; card text: "research purposes only"; no LICENSE file in the pinned tree | training data of the released weights (iris images, and CG iris textures derived from them); not redistributed; see MODEL_CARD.md |
| safetensors, jsonschema, PyYAML, NumPy | uv.lock | Apache-2.0, MIT, MIT, BSD-3-Clause | model files, job/contract validation, configs, byte buffers at the I/O boundary |
| OpenCV (opencv-python) | 4.13 (external venv) | Apache-2.0 | only to generate the ZIP EXR test fixtures in `tests/fixtures/exr_zip/` once; not a dependency |

Evaluated during development and not used: UnReflectAnything weights (availability conflict), LaMa, DINOv3, FlowSeek, SEA-RAFT,
public HDRIs, Triton (unavailable on this Windows host; `torch.compile` disabled with the recorded reason, CUDA eager used).

face-alignment BSD 3-Clause notice: Copyright (c) 2017, Adrian Bulat. All rights reserved. Redistribution and use in
source and binary forms, with or without modification, are permitted provided that the conditions of the BSD 3-Clause
License are met (retain this notice; do not use the author's name to endorse derived products; provided "AS IS" without
warranty).

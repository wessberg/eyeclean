#!/usr/bin/env bash
# EyeClean project-local bootstrap (Linux adapter; Windows is the tested platform).
# Installs nothing system-wide.  Requires an NVIDIA driver with CUDA 13 support and uv.  --training also needs
# Blender >= 5.2 (on PATH, in /opt/blender, or EYECLEAN_BLENDER=/path/to/blender).
#
#   ./bootstrap.sh              # to process footage: environment, face-detector weights, quick doctor
#   ./bootstrap.sh --training   # also the CG data pipeline: MPFB + CC0 asset packs, the iris bank, full doctor
set -euo pipefail
cd "$(dirname "$0")"
training=0
[ "${1:-}" = "--training" ] && training=1
command -v uv >/dev/null || { echo "uv is required: https://docs.astral.sh/uv/" >&2; exit 2; }
uv python install 3.12
uv sync --frozen
if [ "$training" = 1 ]; then
  uv run python -m eyeclean.assets.fetch
  uv run python -m eyeclean.assets.blender_profile
else
  uv run python -m eyeclean.assets.fetch s3fd_weights fan2d_weights face_alignment_source
fi
uv run python -m eyeclean.tracking.weights
mkdir -p runs
quick="--quick"
[ "$training" = 1 ] && quick=""
uv run eyeclean doctor --gpu-policy strict $quick --output runs/doctor.json \
  || { echo "doctor reported a failed check; see runs/doctor.json" >&2; exit 3; }
echo "Bootstrap complete.  Next: put the model in models/ (see models/README.md), then:  uv run eyeclean models"

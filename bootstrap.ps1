# EyeClean project-local bootstrap (Windows).
# Installs nothing system-wide: no driver, CUDA toolkit, system Python or user Blender changes.
# Requires: an NVIDIA driver with CUDA 13 support and uv.  -Training also needs Blender >= 5.2 (or set EYECLEAN_BLENDER).
#
#   ./bootstrap.ps1             # to process footage: environment, face-detector weights, quick doctor
#   ./bootstrap.ps1 -Training   # also the CG data pipeline: MPFB + CC0 asset packs, the iris bank, full doctor
param([switch]$Training)
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Error "uv is required (https://docs.astral.sh/uv/). Example: winget install astral-sh.uv"
}
uv python install 3.12
uv sync --frozen                                   # exact versions from uv.lock (torch 2.14.0+cu130, PyNvVideoCodec 2.2.2, ...)
if (-not $?) { exit 1 }
if ($Training) {
    uv run python -m eyeclean.assets.fetch         # every public, ungated asset, with pinned SHA-256
    if (-not $?) { exit 1 }
    uv run python -m eyeclean.assets.blender_profile   # MPFB 2.0.17 + CC0 asset packs into third_party/blender_user
    if (-not $?) { exit 1 }
} else {
    uv run python -m eyeclean.assets.fetch s3fd_weights fan2d_weights face_alignment_source   # the face detector
    if (-not $?) { exit 1 }
}
uv run python -m eyeclean.tracking.weights         # one-time isolated conversion of S3FD/FAN pickles to safetensors
if (-not $?) { exit 1 }
New-Item -ItemType Directory -Force runs | Out-Null
if ($Training) { uv run eyeclean doctor --gpu-policy strict --output runs/doctor.json }
else { uv run eyeclean doctor --gpu-policy strict --quick --output runs/doctor.json }
if (-not $?) { Write-Warning "doctor reported a failed check; see runs/doctor.json" ; exit 3 }
Write-Host "Bootstrap complete.  Next: put the model in models/ (see models/README.md), then:  uv run eyeclean models"

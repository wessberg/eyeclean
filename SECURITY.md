# Security policy

## Reporting a vulnerability

Please report security issues **privately**, through GitHub's private vulnerability reporting for this repository:
the **Security** tab, then **Report a vulnerability**.  Do not open a public issue or pull request for them.  The same
channel reaches only the maintainers, and it is also the place for Code of Conduct reports (title them "Conduct:").

We aim to acknowledge a report within a week and to agree on a disclosure date with you.

## Scope

EyeClean processes untrusted media files on a GPU.  Of most interest:
- **Parsers:** the EXR reader (`src/eyeclean/io/exr.py`), video demuxing and decoding, job files and sidecars.
- **Model files:** bundles load through `safetensors`.  Pickles are never loaded, and a registry model's SHA-256 is
  verified before use.
- **The preservation guarantee:** any way to make EyeClean change pixels outside its reported hard support.
- **Paths:** writing outside the requested output directory, or into the input's own folder.

The face-detector weights are PyTorch pickles from their upstream source.  They are converted once, at install, with
`torch.load(weights_only=True)` in an isolated step (`python -m eyeclean.tracking.weights`).

## Supported versions

Only the latest release receives fixes.

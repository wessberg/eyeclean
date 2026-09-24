# blender-virtual-production-experiment adapter

- `eyeclean_pass.py` is a reference eye pass for blender-virtual-production-experiment.
  - Copy it into that product's `scripts/`.
  - It runs EyeClean out of process, via `eyeclean-client`.
  - It writes both EyeClean's float outputs and the product's existing
    `composite/<run>/{source,removal,weight}` + `removal.json` layout.
- The full plan (install, float compositing, gating the corneal pass, acceptance) is in
  [`docs/integrations/blender-virtual-production.md`](../../docs/integrations/blender-virtual-production.md).

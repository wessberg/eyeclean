"""Isolated-scene validation of the paired-target construction (runs inside Blender).

Renders one optical eye (no head) in front of a diffuse backdrop with a softbox, a small bulb and
a procedural environment, using the *same* material/switch code as production:

  beauty, clean, reflection, lab_a, lab_b, unit_transmission   (main.exr, high spp, no adaptivity)
  lab_b with corneal IOR 1.0                                    (ior1_labb.exr)

The Python side (eyeclean.validation.pair_validation) checks statistically that
beauty == clean + reflection in expectation, clean == (1-F) * unit_transmission on the cornea,
refraction is active, off-eye pixels are unchanged and the reflection layer is exactly zero
outside the camera-visible ocular surface.

Usage: blender -b --factory-startup --python blender/pair_validation.py -- --job JOB.json
"""

from __future__ import annotations

import json
import math
import os
import sys

import bpy
import mathutils
from bpy_extras.object_utils import world_to_camera_view

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import anatomy  # noqa: E402
import lighting as LT  # noqa: E402
import materials as M  # noqa: E402
from common import VIEW_LAYERS, clear_scene, log, look_at_matrix, setup_optix, write_json  # noqa: E402


def main() -> int:
    argv = sys.argv[sys.argv.index("--") + 1:]
    job = json.load(open(argv[argv.index("--job") + 1], encoding="utf-8"))
    clear_scene()
    scene = bpy.context.scene
    dev = setup_optix(scene)
    log("OptiX devices", dev)
    ep = dict(job["eye"])
    ep["radius"] = 0.012
    mats = {
        "cornea": M.cornea_material("val_cornea", ep),
        "sclera": M.sclera_material("val_sclera", ep, job["textures"]),
        "iris": M.iris_material("val_iris", ep, job["textures"]),
        "interior": M.interior_material("val_interior"),
    }
    eye = anatomy.build_eye("val", ep, mats)
    problems = anatomy.validate_eye(eye)
    # backdrop: diffuse card behind and around the eye (receives corneal reflection caustics)
    me = bpy.data.meshes.new("backdrop")
    s = 0.06
    me.from_pydata([(-s, -s, 0), (s, -s, 0), (s, s, 0), (-s, s, 0)], [], [(0, 1, 2, 3)])
    bd = bpy.data.objects.new("backdrop", me)
    bd.data.materials.append(M.plain_material("backdrop_m", color=(0.35, 0.3, 0.28), roughness=0.9))
    scene.collection.objects.link(bd)
    bd.location = (0, 0, -0.0125)  # just behind the 12 mm eyeball, never intersecting it
    # camera straight on the optical axis
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    scene.collection.objects.link(cam)
    scene.camera = cam
    cam.data.lens = 100.0
    cam.data.sensor_width = 36.0
    dist = float(job.get("camera_distance_m", 0.14))
    cam.matrix_world = look_at_matrix(mathutils.Vector((0.0, 0.0, dist)), mathutils.Vector((0, 0, 0.004)))
    # lights
    hf = {"fwd": mathutils.Vector((0, 0, 1)), "right": mathutils.Vector((1, 0, 0)), "up": mathutils.Vector((0, 1, 0))}
    def sph(az, el):
        a, e = math.radians(az), math.radians(el)
        return (hf["fwd"] * math.cos(e) * math.cos(a) + hf["right"] * math.cos(e) * math.sin(a) + hf["up"] * math.sin(e)).normalized()
    for k, ls in enumerate(job["lights"]):
        spec = dict(ls)
        spec["position"] = sph(ls["azimuth_deg"], ls["elevation_deg"]) * ls["distance_m"]
        spec["target"] = (0, 0, 0)
        spec["name"] = f"VL{k}"
        LT.create_light(spec)
    LT.build_world(dict(job["world"], name="val_world"))
    # render configuration
    r = scene.render
    r.resolution_x = int(job["width"])
    r.resolution_y = int(job["height"])
    r.resolution_percentage = 100
    r.use_single_layer = False
    c = scene.cycles
    c.samples = int(job["samples"])
    c.use_adaptive_sampling = False
    c.use_denoising = False
    c.seed = int(job.get("seed", 1))
    c.max_bounces = 16
    c.diffuse_bounces = 4
    c.glossy_bounces = 6
    c.transmission_bounces = 16
    c.transparent_max_bounces = 48
    c.caustics_reflective = True
    c.caustics_refractive = True
    c.sample_clamp_direct = 0.0
    c.sample_clamp_indirect = 0.0
    if job.get("clean_iris_render"):  # v2 render settings (same as blender/generate.py)
        c.sample_clamp_indirect = 10.0
    c.blur_glossy = 0.0
    im = r.image_settings
    im.media_type = "MULTI_LAYER_IMAGE"
    im.file_format = "OPEN_EXR_MULTILAYER"
    im.color_depth = "32"
    im.exr_codec = "NONE"
    layers = ["beauty", "clean", "reflection", "lab_a", "lab_b", "unit_transmission"]
    vls = scene.view_layers
    vls[0].name = layers[0]
    for name in layers[1:]:
        vls.new(name)
    for vl in vls:
        vl["ec_mode"] = float(VIEW_LAYERS[vl.name])
        vl.samples = 64 if vl.name.startswith("lab_") else 0
    out = job["out_dir"]
    os.makedirs(out, exist_ok=True)
    r.filepath = os.path.join(out, "main")
    bpy.ops.render.render(write_still=True)
    log("main render done")
    # refraction check: same label layer with corneal IOR 1.0 (no bending)
    for vl in vls:
        vl.use = vl.name == "lab_b"
    for n in mats["cornea"].node_tree.nodes:
        if n.bl_idname == "ShaderNodeBsdfRefraction":
            n.inputs["IOR"].default_value = 1.0
    r.filepath = os.path.join(out, "ior1_labb")
    bpy.ops.render.render(write_still=True)
    log("ior1 render done")
    W, H = r.resolution_x, r.resolution_y
    g = eye["geom"]
    def proj(v):
        p = world_to_camera_view(scene, cam, mathutils.Vector(v))
        return [p.x * W, (1 - p.y) * H]
    info = {"device": dev, "blender": bpy.app.version_string, "problems": problems, "geom": g,
            "center_px": proj((0, 0, 0)), "apex_px": proj((0, 0, g["z_apex"])),
            "limbus_px": [proj((g["rl"] * math.cos(t), g["rl"] * math.sin(t), g["zl"])) for t in [2 * math.pi * k / 32 for k in range(32)]],
            "ior_surface": ep["ior_surface"], "n_eff": ep["n_eff"], "samples": c.samples, "width": W, "height": H,
            "files": {"main": "main.exr", "ior1_labb": "ior1_labb.exr"}}
    write_json(os.path.join(out, "validation_scene.json"), info)
    return 0


if __name__ == "__main__":
    sys.exit(main())

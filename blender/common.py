"""Shared Blender-side helpers (runs inside Blender's bundled Python).

CPU work here is limited to scene/mesh construction and metadata; all sampling/shading runs in
Cycles on an explicitly selected OptiX device.
"""

from __future__ import annotations

import importlib
import json
import math
import os
import sys
import time

import bpy
import mathutils

MODE_BEAUTY, MODE_REFLECTION, MODE_LABEL_A, MODE_CLEAN, MODE_LABEL_B = 0, 1, 2, 3, 4
VIEW_LAYERS = {
    "clean": MODE_CLEAN,
    "reflection": MODE_REFLECTION,
    "lab_a": MODE_LABEL_A,
    "lab_b": MODE_LABEL_B,
    "beauty": MODE_BEAUTY,  # ordinary render, only for pair-validation scenes
    "unit_transmission": 5,  # validation only: cornea camera rays = weight-1 refraction
}

_T0 = time.time()


def log(*a) -> None:
    print(f"[ec {time.time() - _T0:8.2f}s]", *a, flush=True)


def enable_mpfb() -> None:
    mod = "bl_ext.user_default.mpfb"
    if mod not in bpy.context.preferences.addons:
        bpy.ops.preferences.addon_enable(module=mod)


def mpfb(name: str, key: str):
    """MPFB's documented dynamic-import quirk (script_samples/01_create_a_basic_human.py)."""
    for amod in list(sys.modules):
        if amod.endswith(name):
            m = importlib.import_module(amod)
            if not hasattr(m, key):
                raise AttributeError(f"{amod} lacks {key}")
            return getattr(m, key)
    raise ValueError(f"module ending in {name} not loaded (is MPFB enabled?)")


def setup_optix(scene: bpy.types.Scene) -> dict:
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "OPTIX"
    prefs.refresh_devices()
    devs = []
    for d in prefs.devices:
        d.use = d.type == "OPTIX"
        devs.append({"name": d.name, "type": d.type, "use": d.use})
    enabled = [d for d in devs if d["use"]]
    if not enabled:
        raise RuntimeError(f"no OptiX device available to Cycles; devices={devs}")
    scene.render.engine = "CYCLES"
    scene.cycles.device = "GPU"
    return {"compute_device_type": prefs.compute_device_type, "devices": devs,
            "enabled": [d["name"] for d in enabled], "scene_device": scene.cycles.device}


def clear_scene() -> None:
    for coll in (bpy.data.objects, bpy.data.meshes, bpy.data.materials, bpy.data.lights, bpy.data.cameras,
                 bpy.data.images, bpy.data.node_groups, bpy.data.curves, bpy.data.armatures, bpy.data.worlds):
        for item in list(coll):
            try:
                coll.remove(item)
            except Exception:
                pass


def look_at_matrix(eye: mathutils.Vector, target: mathutils.Vector, roll: float = 0.0,
                   up: mathutils.Vector = mathutils.Vector((0, 0, 1))) -> mathutils.Matrix:
    fwd = (target - eye).normalized()
    q = fwd.to_track_quat("-Z", "Y")
    m = q.to_matrix().to_4x4()
    if roll:
        m = m @ mathutils.Matrix.Rotation(roll, 4, "Z")
    m.translation = eye
    return m


def link(obj: bpy.types.Object, coll: bpy.types.Collection | None = None) -> bpy.types.Object:
    (coll or bpy.context.scene.collection).objects.link(obj)
    return obj


def set_visibility(obj: bpy.types.Object, camera: bool = True, diffuse: bool = True, glossy: bool = True,
                   transmission: bool = True, shadow: bool = True, volume: bool = True) -> None:
    obj.visible_camera = camera
    obj.visible_diffuse = diffuse
    obj.visible_glossy = glossy
    obj.visible_transmission = transmission
    obj.visible_shadow = shadow
    obj.visible_volume_scatter = volume


def write_json(path: str, obj) -> None:
    tmp = path + ".part"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def vec(v) -> list:
    return [round(float(x), 6) for x in v]


def rad(d: float) -> float:
    return math.radians(d)

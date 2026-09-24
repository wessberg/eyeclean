"""Headless paired-data generator.

Usage:
  blender -b --factory-startup --python-exit-code 1 --python blender/generate.py -- --job JOB.json

The job (written by ``eyeclean generate``) lists identities and shots.  Every frame is rendered
once with four view layers into one uncompressed FLOAT multipart EXR:
  clean (mode 3), reflection (mode 1), lab_a (mode 2), lab_b (mode 4)  [+ beauty (mode 0) optionally]
Metadata (camera, projected eye anatomy, gaze, openness, lights) goes to meta.json.  These are
generator outputs for training/evaluation; the prediction process never receives them.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
import traceback

import bpy
import mathutils
from bpy_extras.object_utils import world_to_camera_view

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import head_adapter as HA  # noqa: E402
import lighting as LT  # noqa: E402
import materials as M  # noqa: E402
from common import VIEW_LAYERS, clear_scene, log, look_at_matrix, setup_optix, vec, write_json  # noqa: E402


def configure_render(scene, shot: dict, device_info: dict) -> None:
    r = scene.render
    r.resolution_x = int(shot["width"])
    r.resolution_y = int(shot["height"])
    r.resolution_percentage = 100
    r.use_persistent_data = True
    r.film_transparent = False
    r.use_single_layer = False
    c = scene.cycles
    c.samples = int(shot["samples"])
    c.use_adaptive_sampling = bool(shot.get("adaptive", True))
    c.adaptive_threshold = float(shot.get("adaptive_threshold", 0.01))
    c.adaptive_min_samples = int(shot.get("adaptive_min_samples", 0))
    c.use_denoising = False
    c.seed = int(shot["seed"]) & 0x7FFFFFFF
    c.max_bounces = 16
    c.diffuse_bounces = 4
    c.glossy_bounces = 6
    c.transmission_bounces = 16
    c.transparent_max_bounces = 48
    c.volume_bounces = 0
    c.caustics_reflective = True
    c.caustics_refractive = True
    c.sample_clamp_direct = 0.0
    c.sample_clamp_indirect = 0.0
    c.blur_glossy = 0.0
    if shot.get("clean_iris_render"):
        # v2: suppress firefly "sparkle" on irises under the cornea by clamping indirect samples.
        # (Disabling refractive caustics instead failed pair validation E6: it changes how the iris is
        # lit through the cornea.)  Source and target share this clean layer, so the pair stays exact.
        c.sample_clamp_indirect = 10.0
    s = r.image_settings
    s.media_type = "MULTI_LAYER_IMAGE"
    s.file_format = "OPEN_EXR_MULTILAYER"
    s.color_depth = "32"
    s.exr_codec = "NONE"
    scene.view_settings.view_transform = "Standard"
    layers = shot.get("layers", ["clean", "reflection", "lab_a", "lab_b"])
    vls = scene.view_layers
    first = vls[0]
    first.name = layers[0]
    existing = {vl.name for vl in vls}
    for name in layers[1:]:
        if name not in existing:
            vls.new(name)
    for vl in list(vls):
        if vl.name not in layers:
            vls.remove(vl)
    for vl in vls:
        vl["ec_mode"] = float(VIEW_LAYERS[vl.name])
        vl.use_pass_combined = True
        vl.use_pass_z = False
        vl.samples = int(shot.get("label_samples", 16)) if vl.name.startswith("lab_") else 0
        try:
            vl.cycles.denoising_store_passes = False
        except Exception:
            pass


# ------------------------------------------------------------------ head-relative placement
def head_frame(I) -> dict:
    cL, fL, uL, _ = I.frames["L"]
    cR, fR, uR, _ = I.frames["R"]
    mid = (cL + cR) / 2
    fwd = (fL + fR).normalized()
    up = mathutils.Vector((0, 0, 1))
    right = fwd.cross(up).normalized()
    up = right.cross(fwd).normalized()
    return {"mid": mid, "fwd": fwd, "up": up, "right": right, "L": cL, "R": cR, "ipd": (cL - cR).length}


def posed_eye_center(I, side):
    return I.eyes[side]["root"].matrix_world.translation.copy()


def sph(hf: dict, az_deg: float, el_deg: float) -> mathutils.Vector:
    az, el = math.radians(az_deg), math.radians(el_deg)
    return (hf["fwd"] * math.cos(el) * math.cos(az) + hf["right"] * math.cos(el) * math.sin(az) + hf["up"] * math.sin(el)).normalized()


def resolve_target(I, hf, name: str) -> mathutils.Vector:
    if name == "eye_L":
        return posed_eye_center(I, "L")
    if name == "eye_R":
        return posed_eye_center(I, "R")
    if name == "eyes_mid":
        return (posed_eye_center(I, "L") + posed_eye_center(I, "R")) / 2
    if name == "face":
        return hf["mid"] + hf["up"] * (-0.035) + hf["fwd"] * (-0.02)
    raise ValueError(name)


def place_camera(scene, cam_obj, I, hf, cam: dict, shot: dict) -> dict:
    cd = cam_obj.data
    cd.lens = float(cam["lens_mm"])
    cd.sensor_fit = "HORIZONTAL"
    cd.sensor_width = float(cam.get("sensor_mm", 36.0))
    cd.clip_start = 0.01
    cd.clip_end = 200.0
    tgt = resolve_target(I, hf, cam.get("target", "eyes_mid"))
    tgt = tgt + hf["right"] * cam.get("offset_right_m", 0.0) + hf["up"] * cam.get("offset_up_m", 0.0)
    fx = cd.lens / cd.sensor_width * shot["width"]
    if "iris_px" in cam:
        rl = I.eyes["L"]["geom"]["rl"]
        dist = 2 * rl * fx / float(cam["iris_px"])
    elif "face_frac" in cam:
        dist = 0.24 * fx / (float(cam["face_frac"]) * shot["height"])  # ~24 cm face height
    else:
        dist = float(cam.get("distance_m", 0.5))
    d = sph(hf, cam.get("azimuth_deg", 0.0), cam.get("elevation_deg", 0.0))
    pos = tgt + d * dist
    cam_obj.matrix_world = look_at_matrix(pos, tgt, math.radians(cam.get("roll_deg", 0.0)))
    return {"position": vec(pos), "target": vec(tgt), "distance_m": dist, "fx_px": fx}


def build_lights(I, hf, shot: dict):
    objs = []
    face = resolve_target(I, hf, "face")
    for k, ls in enumerate(shot.get("lights", [])):
        spec = dict(ls)
        d = sph(hf, ls["azimuth_deg"], ls["elevation_deg"])
        spec["position"] = face + d * ls["distance_m"]
        tgt = face + hf["right"] * ls.get("aim_right_m", 0.0) + hf["up"] * ls.get("aim_up_m", 0.0)
        spec["target"] = tgt
        spec["name"] = f"L{k}_{ls['family']}"
        o = LT.create_light(spec)
        o["ec_spec"] = json.dumps({k2: v for k2, v in ls.items()})
        objs.append((o, ls, spec))
    for k, ss in enumerate(shot.get("silhouettes", [])):
        spec = dict(ss)
        d = sph(hf, ss["azimuth_deg"], ss["elevation_deg"])
        spec["position"] = face + d * ss["distance_m"]
        spec["target"] = face
        spec["name"] = f"S{k}"
        objs.append((LT.create_silhouette(spec), ss, spec))
    return objs


def move_lights(objs, hf, I, frame: dict) -> None:
    face = resolve_target(I, hf, "face")
    for k, (o, ls, spec) in enumerate(objs):
        mv = frame.get("light_motion", {}).get(str(k))
        if mv is not None:
            d = sph(hf, ls["azimuth_deg"] + mv.get("d_az", 0.0), ls["elevation_deg"] + mv.get("d_el", 0.0))
            pos = face + d * (ls["distance_m"] + mv.get("d_dist", 0.0))
            mw = look_at_matrix(pos, face, spec.get("roll", 0.0))
            if o.type == "MESH":
                mw = mw @ mathutils.Matrix.Rotation(math.pi, 4, "X")
            o.matrix_world = mw
            if "power_scale" in mv:
                if o.type == "LIGHT":
                    o.data.energy = float(ls["power"]) * mv["power_scale"]
        if ls.get("family") == "flame" and "flame_phase" in frame:
            LT.update_flame_phase(o, ls.get("seed", 0), frame["flame_phase"])


def update_world_phase(world, env: dict, phase: float) -> None:
    seed = float(env.get("seed", 0))
    for n in world.node_tree.nodes:
        if n.bl_idname == "ShaderNodeTexVoronoi" and n.voronoi_dimensions == "4D" and abs(n.inputs["Scale"].default_value - env.get("panel_scale", 3.0)) < 1e-6:
            n.inputs["W"].default_value = seed * 1.7 + phase
        if n.bl_idname == "ShaderNodeTexNoise":
            n.inputs["W"].default_value = seed * 0.61 + phase


# ------------------------------------------------------------------ metadata
def project(scene, cam, co) -> list:
    v = world_to_camera_view(scene, cam, co)
    W, H = scene.render.resolution_x, scene.render.resolution_y
    return [v.x * W, (1.0 - v.y) * H, v.z]


def eye_metadata(scene, cam, I, side: str, frame: dict) -> dict:
    eye = I.eyes[side]
    g = eye["geom"]
    mw = eye["root"].matrix_world
    c = mw.translation
    fwd = (mw.to_3x3() @ mathutils.Vector((0, 0, 1))).normalized()
    cam_pos = cam.matrix_world.translation
    facing = fwd.dot((cam_pos - c).normalized())
    limbus = [project(scene, cam, mw @ mathutils.Vector((g["rl"] * math.cos(t), g["rl"] * math.sin(t), g["zl"])))
              for t in [2 * math.pi * k / 32 for k in range(32)]]
    frac = frame.get("pupil", {}).get(side, 0.5)
    rp = g["p_min"] + frac * (g["p_max"] - g["p_min"])
    pupil = [project(scene, cam, mw @ mathutils.Vector((rp * math.cos(t), rp * math.sin(t), g["zi"] + g["bulge"])))
             for t in [2 * math.pi * k / 16 for k in range(16)]]
    ctr = project(scene, cam, c)
    apex = project(scene, cam, mw @ mathutils.Vector((0, 0, g["z_apex"])))
    lx = [p[0] for p in limbus]
    ly = [p[1] for p in limbus]
    W, H = scene.render.resolution_x, scene.render.resolution_y
    diam = max(max(lx) - min(lx), max(ly) - min(ly))
    in_frame = (min(lx) > 0 and max(lx) < W and min(ly) > 0 and max(ly) < H and ctr[2] > 0)
    return {
        "center_px": [round(ctr[0], 3), round(ctr[1], 3)], "depth_m": round(ctr[2], 5),
        "apex_px": [round(apex[0], 3), round(apex[1], 3)],
        "limbus_px": [[round(p[0], 3), round(p[1], 3)] for p in limbus],
        "pupil_px": [[round(p[0], 3), round(p[1], 3)] for p in pupil],
        "iris_diameter_px": round(diam, 3),
        "gaze_deg": frame.get("gaze", {}).get(side, [0.0, 0.0]),
        "openness": frame.get("openness", {}).get(side, 1.0),
        "pupil_frac": frac,
        "pupil_radius_m": rp, "limbus_radius_m": g["rl"], "sclera_radius_m": g["Rs"],
        "facing_cos": round(facing, 4), "in_frame": bool(in_frame),
    }


# ------------------------------------------------------------------ main
def render_shot(scene, I, shot: dict, dev_info: dict) -> dict:
    configure_render(scene, shot, dev_info)
    out_dir = shot["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    cam_obj = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj
    HA.set_head(I, 0.0, 0.0, 0.0)
    bpy.context.view_layer.update()
    hf = head_frame(I)
    world = LT.build_world(dict(shot["world"], name=shot["shot_id"] + "_world"))
    lights = build_lights(I, hf, shot)
    frames_meta = []
    cam_info = None
    for fi, fr in enumerate(shot["frames"]):
        t0 = time.time()
        hy, hp, hr = fr.get("head", [0.0, 0.0, 0.0])
        HA.set_head(I, hy, hp, hr)
        for side in ("L", "R"):
            gy, gp = fr.get("gaze", {}).get(side, [0.0, 0.0])
            HA.set_gaze(I, side, gy, gp)
            HA.set_lids(I, side, fr.get("openness", {}).get(side, 1.0), gp)
            from anatomy import set_pupil
            set_pupil(I.eyes[side]["iris"], fr.get("pupil", {}).get(side, 0.5))
        bpy.context.view_layer.update()
        if fi == 0 or fr.get("camera") is not None:
            # frame 0 places the camera; later frames move it only when they carry a camera delta
            cam_info = place_camera(scene, cam_obj, I, hf, dict(shot["camera"], **(fr.get("camera") or {})), shot)
        move_lights(lights, hf, I, fr)
        if "env_phase" in fr:
            update_world_phase(world, shot["world"], fr["env_phase"])
        scene.frame_set(int(fr.get("frame", fi + 1)))
        bpy.context.view_layer.update()
        path = os.path.join(out_dir, f"frame_{fi:04d}")
        scene.render.filepath = path
        bpy.ops.render.render(write_still=True)
        dt = time.time() - t0
        meta = {
            "index": fi, "frame": int(fr.get("frame", fi + 1)), "time_s": fr.get("time_s", fi / 24.0),
            "exr": os.path.basename(path) + ".exr",
            "camera": dict(cam_info, lens_mm=cam_obj.data.lens, sensor_mm=cam_obj.data.sensor_width,
                           matrix_world=[vec(r) for r in cam_obj.matrix_world]),
            "head_deg": [hy, hp, hr],
            "eyes": {side: eye_metadata(scene, cam_obj, I, side, fr) for side in ("L", "R")},
            "render_seconds": round(dt, 3),
        }
        frames_meta.append(meta)
        log(f"{shot['shot_id']} frame {fi + 1}/{len(shot['frames'])} {dt:.2f}s")
    if shot.get("save_blend"):
        bpy.ops.wm.save_as_mainfile(filepath=os.path.join(out_dir, "scene.blend"), copy=True, compress=True)
    # tear down shot-level objects
    for o, _, _ in lights:
        data = o.data
        bpy.data.objects.remove(o)
        if data is not None and getattr(data, "users", 1) == 0:
            try:
                (bpy.data.lights if isinstance(data, bpy.types.Light) else bpy.data.meshes).remove(data)
            except Exception:
                pass
    bpy.data.objects.remove(cam_obj)
    bpy.data.worlds.remove(world)
    return {"shot_id": shot["shot_id"], "frames": frames_meta}


def main() -> int:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    job_path = argv[argv.index("--job") + 1]
    with open(job_path, "r", encoding="utf-8") as f:
        job = json.load(f)
    scene = bpy.context.scene
    dev = setup_optix(scene)
    log("OptiX devices", dev)
    status = {"job": job_path, "device": dev, "blender": bpy.app.version_string, "build_hash": bpy.app.build_hash.decode() if isinstance(bpy.app.build_hash, bytes) else str(bpy.app.build_hash), "identities": []}
    for ident in job["identities"]:
        clear_scene()
        scene = bpy.context.scene
        dev = setup_optix(scene)
        t0 = time.time()
        I = HA.build_identity(ident, ident["textures"], render_subdiv=int(ident.get("render_subdiv", 2)))
        log(f"identity {ident['identity_id']} built in {time.time() - t0:.1f}s lid_close={I.lid_close}")
        ident_meta = {"identity_id": ident["identity_id"], "build_seconds": round(time.time() - t0, 2),
                      "eyes": {s: {"geom": I.eyes[s]["geom"], "params": I.eyes[s]["params"]} for s in ("L", "R")},
                      "meta": I.meta, "shots": []}
        for shot in ident["shots"]:
            done_flag = os.path.join(shot["out_dir"], "meta.json")
            if os.path.exists(done_flag) and not job.get("overwrite", False):
                log(f"skip {shot['shot_id']} (done)")
                continue
            try:
                res = render_shot(scene, I, shot, dev)
                res.update({"identity_id": ident["identity_id"], "shot": {k: v for k, v in shot.items() if k != "frames"},
                            "device": dev, "blender": status["blender"], "eye_geometry": ident_meta["eyes"],
                            "identity_meta": I.meta})
                write_json(done_flag, res)
                ident_meta["shots"].append(shot["shot_id"])
            except Exception as e:  # record and continue with the next shot
                traceback.print_exc()
                write_json(os.path.join(shot["out_dir"], "error.json"), {"error": repr(e), "trace": traceback.format_exc()})
        status["identities"].append(ident_meta)
    if job.get("status_path"):
        write_json(job["status_path"], status)
    return 0


if __name__ == "__main__":
    try:
        code = main()
    except Exception:
        traceback.print_exc()
        code = 1
    sys.exit(code)

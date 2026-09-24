"""Light rigs and procedural environments (all shading evaluated by Cycles on the GPU).

Families: rect_softbox, octa_softbox, strip, window, grid_panel, bulb, colored_led, sun, flame,
emissive_card, silhouette (dark occluder cards visible only to glossy rays), and ring_light,
which is the held-out light family (test split only).  Environments are procedural world
shaders (rooms, patterns, coloured/dark structure) with an animatable phase.
"""

from __future__ import annotations

import math

import bpy
import mathutils

import materials as M
from common import link, look_at_matrix, set_visibility

LIGHT_FAMILIES = ("rect_softbox", "octa_softbox", "strip", "window", "grid_panel", "bulb", "colored_led",
                  "sun", "flame", "emissive_card", "ring_light")
HELDOUT_LIGHT = "ring_light"


def _poly_mesh(name, pts2d):
    verts = [(x, y, 0.0) for x, y in pts2d]
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], [list(range(len(verts)))])
    uv = me.uv_layers.new(name="UVMap")
    xs = [p[0] for p in pts2d]
    ys = [p[1] for p in pts2d]
    for poly in me.polygons:
        for li in poly.loop_indices:
            v = verts[me.loops[li].vertex_index]
            uv.data[li].uv = ((v[0] - min(xs)) / (max(xs) - min(xs) + 1e-9), (v[1] - min(ys)) / (max(ys) - min(ys) + 1e-9))
    me.update()
    return me


def _rect_mesh(name, w, h):
    return _poly_mesh(name, [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)])


def _ring_mesh(name, r_out, r_in, n=64):
    verts, faces = [], []
    for k in range(n):
        a = 2 * math.pi * k / n
        verts += [(r_out * math.cos(a), r_out * math.sin(a), 0.0), (r_in * math.cos(a), r_in * math.sin(a), 0.0)]
    for k in range(n):
        k2 = (k + 1) % n
        faces.append((2 * k, 2 * k2, 2 * k2 + 1, 2 * k + 1))
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    me.update()
    return me


def _uv_socket(nt):
    tc = nt.nodes.new("ShaderNodeTexCoord")
    return tc.outputs["UV"]


def _window_pattern(cols, rows, mullion):
    def build(nt):
        uv = _uv_socket(nt)
        sep = nt.nodes.new("ShaderNodeSeparateXYZ")
        nt.links.new(uv, sep.inputs[0])
        out = None
        for axis, n in ((0, cols), (1, rows)):
            mul = nt.nodes.new("ShaderNodeMath"); mul.operation = "MULTIPLY"; mul.inputs[1].default_value = n
            nt.links.new(sep.outputs[axis], mul.inputs[0])
            fr = nt.nodes.new("ShaderNodeMath"); fr.operation = "FRACT"
            nt.links.new(mul.outputs[0], fr.inputs[0])
            # 1 inside panes, 0 on mullions
            d = nt.nodes.new("ShaderNodeMath"); d.operation = "PINGPONG"; d.inputs[1].default_value = 0.5
            nt.links.new(fr.outputs[0], d.inputs[0])
            st = nt.nodes.new("ShaderNodeMath"); st.operation = "GREATER_THAN"; st.inputs[1].default_value = mullion
            nt.links.new(d.outputs[0], st.inputs[0])
            if out is None:
                out = st.outputs[0]
            else:
                m = nt.nodes.new("ShaderNodeMath"); m.operation = "MULTIPLY"
                nt.links.new(out, m.inputs[0]); nt.links.new(st.outputs[0], m.inputs[1])
                out = m.outputs[0]
        return out
    return build


def _grid_dots(n):
    def build(nt):
        uv = _uv_socket(nt)
        vor = nt.nodes.new("ShaderNodeTexVoronoi")
        vor.inputs["Scale"].default_value = n
        vor.inputs["Randomness"].default_value = 0.0
        nt.links.new(uv, vor.inputs["Vector"])
        lt = nt.nodes.new("ShaderNodeMath"); lt.operation = "LESS_THAN"; lt.inputs[1].default_value = 0.28
        nt.links.new(vor.outputs["Distance"], lt.inputs[0])
        return lt.outputs[0]
    return build


def _flame(seed, phase):
    """Teardrop mask x animated turbulence -> warm flame colour (animated through phase)."""
    def build(nt):
        uv = _uv_socket(nt)
        sep = nt.nodes.new("ShaderNodeSeparateXYZ")
        nt.links.new(uv, sep.inputs[0])
        noise = nt.nodes.new("ShaderNodeTexNoise")
        noise.noise_dimensions = "4D"
        noise.inputs["Scale"].default_value = 3.0
        noise.inputs["Detail"].default_value = 6.0
        noise.inputs["W"].default_value = seed * 0.37 + phase
        nt.links.new(uv, noise.inputs["Vector"])
        # distort x by noise, taper by height
        sub = nt.nodes.new("ShaderNodeMath"); sub.operation = "SUBTRACT"; sub.inputs[1].default_value = 0.5
        nt.links.new(sep.outputs[0], sub.inputs[0])
        nz = nt.nodes.new("ShaderNodeMath"); nz.operation = "SUBTRACT"; nz.inputs[1].default_value = 0.5
        nt.links.new(noise.outputs["Fac"], nz.inputs[0])
        nzs = nt.nodes.new("ShaderNodeMath"); nzs.operation = "MULTIPLY"; nzs.inputs[1].default_value = 0.35
        nt.links.new(nz.outputs[0], nzs.inputs[0])
        dx = nt.nodes.new("ShaderNodeMath"); dx.operation = "ADD"
        nt.links.new(sub.outputs[0], dx.inputs[0]); nt.links.new(nzs.outputs[0], dx.inputs[1])
        ax = nt.nodes.new("ShaderNodeMath"); ax.operation = "ABSOLUTE"
        nt.links.new(dx.outputs[0], ax.inputs[0])
        width = nt.nodes.new("ShaderNodeMath"); width.operation = "MULTIPLY_ADD"  # w(y) = 0.45*(1-y)^0.7
        oneminus = nt.nodes.new("ShaderNodeMath"); oneminus.operation = "SUBTRACT"; oneminus.inputs[0].default_value = 1.0
        nt.links.new(sep.outputs[1], oneminus.inputs[1])
        pw = nt.nodes.new("ShaderNodeMath"); pw.operation = "POWER"; pw.inputs[1].default_value = 0.7
        nt.links.new(oneminus.outputs[0], pw.inputs[0])
        wmul = nt.nodes.new("ShaderNodeMath"); wmul.operation = "MULTIPLY"; wmul.inputs[1].default_value = 0.45
        nt.links.new(pw.outputs[0], wmul.inputs[0])
        inside = nt.nodes.new("ShaderNodeMath"); inside.operation = "LESS_THAN"
        nt.links.new(ax.outputs[0], inside.inputs[0]); nt.links.new(wmul.outputs[0], inside.inputs[1])
        ramp = nt.nodes.new("ShaderNodeValToRGB")
        ramp.color_ramp.elements[0].color = (1.0, 0.25, 0.02, 1)
        ramp.color_ramp.elements[1].color = (1.0, 0.85, 0.45, 1)
        nt.links.new(oneminus.outputs[0], ramp.inputs[0])
        mixc = nt.nodes.new("ShaderNodeMix"); mixc.data_type = "RGBA"
        mixc.inputs["A"].default_value = (0, 0, 0, 1)
        nt.links.new(inside.outputs[0], mixc.inputs["Factor"])
        nt.links.new(ramp.outputs["Color"], mixc.inputs["B"])
        return mixc.outputs["Result"]
    return build


def _card_pattern(seed):
    def build(nt):
        uv = _uv_socket(nt)
        wave = nt.nodes.new("ShaderNodeTexWave")
        wave.inputs["Scale"].default_value = 2.0 + (seed % 7)
        wave.inputs["Distortion"].default_value = float(seed % 5)
        nt.links.new(uv, wave.inputs["Vector"])
        chk = nt.nodes.new("ShaderNodeTexChecker")
        chk.inputs["Scale"].default_value = 3.0 + (seed % 4)
        nt.links.new(uv, chk.inputs["Vector"])
        mx = nt.nodes.new("ShaderNodeMix"); mx.data_type = "RGBA"; mx.blend_type = "MULTIPLY"
        mx.inputs["Factor"].default_value = 1.0
        nt.links.new(wave.outputs["Color"], mx.inputs["A"])
        nt.links.new(chk.outputs["Color"], mx.inputs["B"])
        return mx.outputs["Result"]
    return build


def create_light(spec: dict, coll=None) -> bpy.types.Object:
    """spec: family, position, target, size [w,h], power, color, roll, seed, phase."""
    fam = spec["family"]
    pos = mathutils.Vector(spec["position"])
    tgt = mathutils.Vector(spec["target"])
    col = tuple(spec.get("color", (1, 1, 1)))
    name = spec.get("name", fam)
    w, h = spec.get("size", (0.5, 0.5))
    power = float(spec.get("power", 100.0))
    obj = None
    if fam in ("rect_softbox", "strip"):
        L = bpy.data.lights.new(name, "AREA")
        L.shape = "RECTANGLE"
        L.size, L.size_y = w, h
        L.energy = power
        L.color = col
        L.spread = math.radians(spec.get("spread_deg", 120.0))
        obj = bpy.data.objects.new(name, L)
    elif fam in ("bulb", "colored_led"):
        L = bpy.data.lights.new(name, "POINT")
        L.shadow_soft_size = w
        L.energy = power
        L.color = col
        obj = bpy.data.objects.new(name, L)
    elif fam == "sun":
        L = bpy.data.lights.new(name, "SUN")
        L.angle = math.radians(spec.get("angle_deg", 1.0))
        L.energy = power  # irradiance W/m^2
        L.color = col
        obj = bpy.data.objects.new(name, L)
    else:
        if fam == "octa_softbox":
            r = w / 2
            me = _poly_mesh(name, [(r * math.cos(2 * math.pi * k / 8 + math.pi / 8), r * math.sin(2 * math.pi * k / 8 + math.pi / 8)) for k in range(8)])
            mat = M.emitter_material(name + "_m", color=col, strength=power)
        elif fam == "ring_light":
            me = _ring_mesh(name, w / 2, w / 2 * spec.get("inner_ratio", 0.72))
            mat = M.emitter_material(name + "_m", color=col, strength=power)
        elif fam == "window":
            me = _rect_mesh(name, w, h)
            pat = _window_pattern(spec.get("cols", 3), spec.get("rows", 2), spec.get("mullion", 0.06))
            mat = M.emitter_material(name + "_m", color_socket_builder=lambda nt: _tint(nt, pat(nt), col), strength=power)
        elif fam == "grid_panel":
            me = _rect_mesh(name, w, h)
            pat = _grid_dots(spec.get("dots", 12))
            mat = M.emitter_material(name + "_m", color_socket_builder=lambda nt: _tint(nt, pat(nt), col), strength=power)
        elif fam == "flame":
            me = _rect_mesh(name, w, h)
            mat = M.emitter_material(name + "_m", color_socket_builder=_flame(spec.get("seed", 0), spec.get("phase", 0.0)), strength=power)
        elif fam == "emissive_card":
            me = _rect_mesh(name, w, h)
            pat = _card_pattern(spec.get("seed", 0))
            mat = M.emitter_material(name + "_m", color_socket_builder=lambda nt: _tint(nt, pat(nt), col), strength=power)
        else:
            raise ValueError(f"unknown light family {fam}")
        me.materials.append(mat)
        obj = bpy.data.objects.new(name, me)
        # mesh emitters face the target (+Z normal toward target)
    link(obj, coll)
    if fam == "sun":
        obj.matrix_world = look_at_matrix(pos, tgt, spec.get("roll", 0.0))
    else:
        mw = look_at_matrix(pos, tgt, spec.get("roll", 0.0))
        obj.matrix_world = mw
        if obj.type == "MESH":
            # look_at points -Z at the target; flip so the emitting +Z face points at it
            obj.matrix_world = mw @ mathutils.Matrix.Rotation(math.pi, 4, "X")
            set_visibility(obj, camera=spec.get("camera_visible", False), shadow=False)
    obj["ec_family"] = fam
    return obj


def _tint(nt, fac_socket, col):
    mx = nt.nodes.new("ShaderNodeMix")
    mx.data_type = "RGBA"
    mx.inputs["A"].default_value = (0.0, 0.0, 0.0, 1.0)
    mx.inputs["B"].default_value = (*col, 1.0)
    nt.links.new(fac_socket, mx.inputs["Factor"])
    return mx.outputs["Result"]


def update_flame_phase(obj, seed: int, phase: float) -> None:
    """Animate the flame texture (same formula as _flame: W = 0.37*seed + phase)."""
    mat = obj.data.materials[0] if obj.type == "MESH" and obj.data.materials else None
    if mat is None:
        return
    for n in mat.node_tree.nodes:
        if n.bl_idname == "ShaderNodeTexNoise" and "W" in n.inputs:
            n.inputs["W"].default_value = seed * 0.37 + phase


def create_silhouette(spec: dict, coll=None) -> bpy.types.Object:
    """Dark camera/person silhouette card: invisible to camera rays, visible in reflections."""
    w, h = spec.get("size", (0.4, 0.9))
    me = _rect_mesh(spec.get("name", "silhouette"), w, h)
    me.materials.append(M.plain_material(spec.get("name", "sil") + "_m", color=tuple(spec.get("color", (0.01, 0.01, 0.012))), roughness=0.8))
    obj = bpy.data.objects.new(spec.get("name", "silhouette"), me)
    link(obj, coll)
    obj.matrix_world = look_at_matrix(mathutils.Vector(spec["position"]), mathutils.Vector(spec["target"])) @ mathutils.Matrix.Rotation(math.pi, 4, "X")
    set_visibility(obj, camera=False, diffuse=True, glossy=True, transmission=True, shadow=False)
    return obj


# ----------------------------------------------------------------------------- environment
def build_world(env: dict) -> bpy.types.World:
    """Procedural room-like environment; env keys: seed, strength, tint, floor, ceiling,
    panel_density, panel_gain, dark_blobs, phase, colorful."""
    world = bpy.data.worlds.new(env.get("name", "ec_world"))
    bpy.context.scene.world = world
    seed = float(env.get("seed", 0))

    def build(nt):
        tc = nt.nodes.new("ShaderNodeTexCoord")
        d = tc.outputs["Generated"]
        sep = nt.nodes.new("ShaderNodeSeparateXYZ")
        nt.links.new(d, sep.inputs[0])
        # floor/wall/ceiling gradient by elevation (z of the direction)
        ramp = nt.nodes.new("ShaderNodeValToRGB")
        ramp.color_ramp.elements[0].position = 0.35
        ramp.color_ramp.elements[0].color = (*env.get("floor", (0.05, 0.045, 0.04)), 1)
        ramp.color_ramp.elements[1].position = 0.75
        ramp.color_ramp.elements[1].color = (*env.get("ceiling", (0.25, 0.25, 0.27)), 1)
        mapz = nt.nodes.new("ShaderNodeMapRange")
        mapz.inputs["From Min"].default_value = -1.0
        mapz.inputs["From Max"].default_value = 1.0
        nt.links.new(sep.outputs[2], mapz.inputs["Value"])
        nt.links.new(mapz.outputs[0], ramp.inputs[0])
        # structured panels (monitors, windows, set equipment) via Voronoi cells on direction
        vor = nt.nodes.new("ShaderNodeTexVoronoi")
        vor.voronoi_dimensions = "4D"
        vor.inputs["Scale"].default_value = env.get("panel_scale", 3.0)
        vor.inputs["W"].default_value = seed * 1.7 + env.get("phase", 0.0)
        nt.links.new(d, vor.inputs["Vector"])
        thr = nt.nodes.new("ShaderNodeMath"); thr.operation = "GREATER_THAN"
        thr.inputs[1].default_value = 1.0 - env.get("panel_density", 0.15)
        _bw(nt, vor.outputs["Color"], thr)
        cell_col = nt.nodes.new("ShaderNodeMix"); cell_col.data_type = "RGBA"
        cell_col.inputs["Factor"].default_value = env.get("colorful", 0.3)
        cell_col.inputs["A"].default_value = (1.0, 1.0, 1.0, 1.0)
        nt.links.new(vor.outputs["Color"], cell_col.inputs["B"])
        gain = nt.nodes.new("ShaderNodeMath"); gain.operation = "MULTIPLY"; gain.inputs[1].default_value = env.get("panel_gain", 4.0)
        nt.links.new(thr.outputs[0], gain.inputs[0])
        panels = nt.nodes.new("ShaderNodeMix"); panels.data_type = "RGBA"; panels.blend_type = "MULTIPLY"
        panels.inputs["Factor"].default_value = 1.0
        nt.links.new(cell_col.outputs["Result"], panels.inputs["A"])
        nt.links.new(gain.outputs[0], panels.inputs["B"])
        # large-scale noise modulation (moving textures through phase)
        noi = nt.nodes.new("ShaderNodeTexNoise"); noi.noise_dimensions = "4D"
        noi.inputs["Scale"].default_value = env.get("noise_scale", 2.5)
        noi.inputs["W"].default_value = seed * 0.61 + env.get("phase", 0.0)
        nt.links.new(d, noi.inputs["Vector"])
        base = nt.nodes.new("ShaderNodeMix"); base.data_type = "RGBA"; base.blend_type = "MULTIPLY"
        base.inputs["Factor"].default_value = 0.7
        nt.links.new(ramp.outputs["Color"], base.inputs["A"])
        nt.links.new(noi.outputs["Color"], base.inputs["B"])
        add = nt.nodes.new("ShaderNodeMix"); add.data_type = "RGBA"; add.blend_type = "ADD"
        add.inputs["Factor"].default_value = 1.0
        nt.links.new(base.outputs["Result"], add.inputs["A"])
        nt.links.new(panels.outputs["Result"], add.inputs["B"])
        # dark blobs (silhouettes of people/equipment) multiply the result
        vb = nt.nodes.new("ShaderNodeTexVoronoi"); vb.voronoi_dimensions = "4D"
        vb.inputs["Scale"].default_value = env.get("blob_scale", 1.6)
        vb.inputs["W"].default_value = seed * 2.3
        nt.links.new(d, vb.inputs["Vector"])
        bl = nt.nodes.new("ShaderNodeMath"); bl.operation = "GREATER_THAN"; bl.inputs[1].default_value = env.get("dark_blobs", 0.15)
        nt.links.new(vb.outputs["Distance"], bl.inputs[0])
        # blend: factor 1 keeps the room, 0 gives a dark silhouette blob
        final = nt.nodes.new("ShaderNodeMix"); final.data_type = "RGBA"
        nt.links.new(bl.outputs[0], final.inputs["Factor"])
        final.inputs["A"].default_value = (*env.get("blob_color", (0.01, 0.01, 0.012)), 1)
        nt.links.new(add.outputs["Result"], final.inputs["B"])
        tint = nt.nodes.new("ShaderNodeMix"); tint.data_type = "RGBA"; tint.blend_type = "MULTIPLY"
        tint.inputs["Factor"].default_value = 1.0
        nt.links.new(final.outputs["Result"], tint.inputs["A"])
        tint.inputs["B"].default_value = (*env.get("tint", (1.0, 1.0, 1.0)), 1)
        return tint.outputs["Result"], float(env.get("strength", 0.3))

    M.world_switch(world, build)
    return world


def _bw(nt, color_socket, target_node):
    rgb2bw = nt.nodes.new("ShaderNodeRGBToBW")
    nt.links.new(color_socket, rgb2bw.inputs[0])
    nt.links.new(rgb2bw.outputs[0], target_node.inputs[0])
    return rgb2bw.inputs[0]

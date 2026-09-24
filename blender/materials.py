"""Shader construction with the per-view-layer paired-target switch.

Each material routes five shaders through the ``EC_Switch`` node group, selected by the view
layer custom property ``ec_mode`` and the Light Path "Is Camera Ray" output:

  mode 0 beauty      ordinary render (validation only)
  mode 1 reflection  camera rays: ONLY the designated ocular reflection closure F*glossy on
                     cornea/sclera; every other camera-visible surface and the background black
  mode 2 lab_a       camera rays: first-hit labels R=designated ocular surface, G=skin/lids,
                     B=lashes/brows (alpha-aware, camera ray continues through card alpha)
  mode 3 clean       camera rays: designated closure replaced by a zero-radiance branch,
                     (1-F)*under - transmission weight and refraction preserved, energy absorbed
  mode 4 lab_b       labels through the refractive cornea: R=iris, G=pupil/eye interior,
                     B=tear meniscus; the cornea becomes a pure refractor with the same IOR

Non-camera rays always use the beauty shader in modes 1 and 3, so illumination transport
outside the camera-visible ocular closure is unchanged.  ``clean + reflection`` is therefore an
unbiased estimator of the beauty image (validated in pair_validation.py).
"""

from __future__ import annotations

import bpy

SWITCH = "EC_Switch"
LABEL_OCULAR = (1.0, 0.0, 0.0)
LABEL_SKIN = (0.0, 1.0, 0.0)
LABEL_LASH = (0.0, 0.0, 1.0)
LABEL_IRIS = (1.0, 0.0, 0.0)
LABEL_PUPIL = (0.0, 1.0, 0.0)
LABEL_TEAR = (0.0, 0.0, 1.0)


def _nt(mat: bpy.types.Material) -> bpy.types.NodeTree:
    try:
        mat.use_nodes = True
    except Exception:
        pass
    return mat.node_tree


def switch_group() -> bpy.types.NodeTree:
    if SWITCH in bpy.data.node_groups:
        return bpy.data.node_groups[SWITCH]
    ng = bpy.data.node_groups.new(SWITCH, "ShaderNodeTree")
    for nm in ("Beauty", "Reflection", "LabelA", "Clean", "LabelB"):
        ng.interface.new_socket(nm, in_out="INPUT", socket_type="NodeSocketShader")
    s = ng.interface.new_socket("LabelB_AllRays", in_out="INPUT", socket_type="NodeSocketFloat")
    s.default_value = 0.0
    ng.interface.new_socket("Shader", in_out="OUTPUT", socket_type="NodeSocketShader")
    N, L = ng.nodes, ng.links
    gi, go = N.new("NodeGroupInput"), N.new("NodeGroupOutput")
    attr = N.new("ShaderNodeAttribute")
    attr.attribute_type = "VIEW_LAYER"
    attr.attribute_name = "ec_mode"
    lp = N.new("ShaderNodeLightPath")

    def cmp(v):
        n = N.new("ShaderNodeMath")
        n.operation = "COMPARE"
        n.inputs[1].default_value = float(v)
        n.inputs[2].default_value = 0.25
        L.new(attr.outputs["Fac"], n.inputs[0])
        return n.outputs[0]

    def op(kind, a, b):
        n = N.new("ShaderNodeMath")
        n.operation = kind
        L.new(a, n.inputs[0])
        L.new(b, n.inputs[1])
        return n.outputs[0]

    cam = lp.outputs["Is Camera Ray"]
    w_ref = op("MULTIPLY", cmp(1), cam)
    w_la = op("MULTIPLY", cmp(2), cam)
    w_cl = op("MULTIPLY", cmp(3), cam)
    w_lb = op("MULTIPLY", cmp(4), op("MAXIMUM", cam, gi.outputs["LabelB_AllRays"]))
    s = gi.outputs["Beauty"]
    for w, name in ((w_cl, "Clean"), (w_ref, "Reflection"), (w_la, "LabelA"), (w_lb, "LabelB")):
        m = N.new("ShaderNodeMixShader")
        L.new(w, m.inputs[0])
        L.new(s, m.inputs[1])
        L.new(gi.outputs[name], m.inputs[2])  # an unlinked group shader input = no closure (black)
        s = m.outputs[0]
    L.new(s, go.inputs["Shader"])
    return ng


def _switch_node(nt, beauty=None, reflection=None, label_a=None, clean=None, label_b=None, lb_all=0.0):
    sw = nt.nodes.new("ShaderNodeGroup")
    sw.node_tree = switch_group()
    for name, sock in (("Beauty", beauty), ("Reflection", reflection), ("LabelA", label_a), ("Clean", clean), ("LabelB", label_b)):
        if sock is not None:
            nt.links.new(sock, sw.inputs[name])
    sw.inputs["LabelB_AllRays"].default_value = float(lb_all)
    return sw


def _emission(nt, color, strength=1.0):
    e = nt.nodes.new("ShaderNodeEmission")
    e.inputs["Color"].default_value = (*color, 1.0)
    e.inputs["Strength"].default_value = strength
    return e.outputs[0]


def _mix(nt, fac, a, b):
    m = nt.nodes.new("ShaderNodeMixShader")
    if isinstance(fac, (int, float)):
        m.inputs[0].default_value = fac
    else:
        nt.links.new(fac, m.inputs[0])
    if a is not None:
        nt.links.new(a, m.inputs[1])
    if b is not None:
        nt.links.new(b, m.inputs[2])
    return m.outputs[0]


def _image(nt, path: str, colorspace: str, uv_map: str = "UVMap", extension: str = "EXTEND"):
    img = bpy.data.images.load(path, check_existing=True)
    img.colorspace_settings.name = colorspace
    t = nt.nodes.new("ShaderNodeTexImage")
    t.image = img
    t.interpolation = "Cubic"
    t.extension = extension
    uv = nt.nodes.new("ShaderNodeUVMap")
    uv.uv_map = uv_map
    nt.links.new(uv.outputs["UV"], t.inputs["Vector"])
    return t


def _linear_cs() -> str:
    names = [c.name for c in bpy.types.ColorManagedInputColorspaceSettings.bl_rna.properties["name"].enum_items]
    for cand in ("Linear Rec.709", "Linear", "Linear sRGB"):
        if cand in names:
            return cand
    return names[0]


def _fresnel(nt, ior):
    f = nt.nodes.new("ShaderNodeFresnel")
    f.inputs["IOR"].default_value = ior
    return f.outputs[0]


def cornea_material(name: str, p: dict) -> bpy.types.Material:
    """Refractive corneal cap: beauty = mix(refraction(n_eff), glossy, F(n_tear))."""
    mat = bpy.data.materials.new(name)
    nt = _nt(mat)
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.target = "CYCLES"
    F = _fresnel(nt, p["ior_surface"])
    refr = nt.nodes.new("ShaderNodeBsdfRefraction")
    refr.distribution = "GGX"
    refr.inputs["IOR"].default_value = p["n_eff"]
    refr.inputs["Roughness"].default_value = 0.0
    gl = nt.nodes.new("ShaderNodeBsdfGlossy")
    gl.distribution = "GGX"
    gl.inputs["Roughness"].default_value = p["cornea_roughness"]
    gl.inputs["Color"].default_value = (1, 1, 1, 1)
    core = _mix(nt, F, refr.outputs[0], gl.outputs[0])
    # shadow rays pass the thin cornea with (1-F) attenuation (standard eye-shading approximation;
    # identical in every layer, so the decomposition stays exact)
    one_minus = nt.nodes.new("ShaderNodeMath")
    one_minus.operation = "SUBTRACT"
    one_minus.inputs[0].default_value = 1.0
    nt.links.new(F, one_minus.inputs[1])
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    nt.links.new(one_minus.outputs[0], tr.inputs["Color"])
    lp = nt.nodes.new("ShaderNodeLightPath")
    beauty = _mix(nt, lp.outputs["Is Shadow Ray"], core, tr.outputs[0])
    refl = _mix(nt, F, None, gl.outputs[0])  # F * glossy only
    clean = _mix(nt, F, refr.outputs[0], None)  # (1-F) * refraction, reflection energy absorbed
    refr_lb = nt.nodes.new("ShaderNodeBsdfRefraction")
    refr_lb.distribution = "GGX"
    refr_lb.inputs["IOR"].default_value = p["n_eff"]
    refr_lb.inputs["Roughness"].default_value = 0.0
    sw = _switch_node(nt, beauty, refl, _emission(nt, LABEL_OCULAR), clean, refr_lb.outputs[0], lb_all=1.0)
    # validation-only mode 5 ("unit_transmission"): camera rays see a weight-1 refraction so the
    # clean branch's (1-F) transmission weighting can be measured; other materials render beauty.
    attr = nt.nodes.new("ShaderNodeAttribute")
    attr.attribute_type = "VIEW_LAYER"
    attr.attribute_name = "ec_mode"
    is5 = nt.nodes.new("ShaderNodeMath")
    is5.operation = "COMPARE"
    is5.inputs[1].default_value = 5.0
    is5.inputs[2].default_value = 0.25
    nt.links.new(attr.outputs["Fac"], is5.inputs[0])
    lp2 = nt.nodes.new("ShaderNodeLightPath")
    w5 = nt.nodes.new("ShaderNodeMath")
    w5.operation = "MULTIPLY"
    nt.links.new(is5.outputs[0], w5.inputs[0])
    nt.links.new(lp2.outputs["Is Camera Ray"], w5.inputs[1])
    final = _mix(nt, w5.outputs[0], sw.outputs[0], refr.outputs[0])
    nt.links.new(final, out.inputs["Surface"])
    mat["ec_kind"] = "cornea"
    return mat


def sclera_material(name: str, p: dict, tex: dict) -> bpy.types.Material:
    """Scleral shell: beauty = mix(SSS base (no built-in specular), tear-film glossy, F)."""
    mat = bpy.data.materials.new(name)
    nt = _nt(mat)
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.target = "CYCLES"
    lin = _linear_cs()
    alb = _image(nt, tex["sclera_albedo"], lin, "UVMap", "REPEAT")
    rough = _image(nt, tex["sclera_roughness"], "Non-Color", "UVMap", "REPEAT")
    base = nt.nodes.new("ShaderNodeBsdfPrincipled")
    nt.links.new(alb.outputs["Color"], base.inputs["Base Color"])
    base.inputs["Roughness"].default_value = 0.5
    base.inputs["Specular IOR Level"].default_value = 0.0  # the only specular is the designated tear film
    base.inputs["Subsurface Weight"].default_value = p["sclera_sss"]
    base.inputs["Subsurface Radius"].default_value = (1.0, 0.45, 0.3)
    base.inputs["Subsurface Scale"].default_value = p["sclera_sss_scale"]
    F = _fresnel(nt, p["ior_tear"])
    gl = nt.nodes.new("ShaderNodeBsdfGlossy")
    gl.distribution = "GGX"
    nt.links.new(rough.outputs["Color"], gl.inputs["Roughness"])
    beauty = _mix(nt, F, base.outputs[0], gl.outputs[0])
    refl = _mix(nt, F, None, gl.outputs[0])
    clean = _mix(nt, F, base.outputs[0], None)
    sw = _switch_node(nt, beauty, refl, _emission(nt, LABEL_OCULAR), clean, None, lb_all=1.0)
    nt.links.new(sw.outputs[0], out.inputs["Surface"])
    mat["ec_kind"] = "sclera"
    return mat


def iris_material(name: str, p: dict, tex: dict) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    nt = _nt(mat)
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.target = "CYCLES"
    alb = _image(nt, tex["iris_albedo"], _linear_cs(), "UVMap", "EXTEND")
    hgt = _image(nt, tex["iris_height"], "Non-Color", "UVMap", "EXTEND")
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = p["iris_bump"]
    bump.inputs["Distance"].default_value = 0.0002
    nt.links.new(hgt.outputs["Color"], bump.inputs["Height"])
    bs = nt.nodes.new("ShaderNodeBsdfPrincipled")
    nt.links.new(alb.outputs["Color"], bs.inputs["Base Color"])
    bs.inputs["Roughness"].default_value = 0.65
    bs.inputs["Specular IOR Level"].default_value = 0.15
    nt.links.new(bump.outputs["Normal"], bs.inputs["Normal"])
    sw = _switch_node(nt, bs.outputs[0], None, None, bs.outputs[0], _emission(nt, LABEL_IRIS), lb_all=1.0)
    nt.links.new(sw.outputs[0], out.inputs["Surface"])
    mat["ec_kind"] = "iris"
    return mat


def interior_material(name: str) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    nt = _nt(mat)
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.target = "CYCLES"
    d = nt.nodes.new("ShaderNodeBsdfDiffuse")
    d.inputs["Color"].default_value = (0.004, 0.0025, 0.002, 1)  # near-black posterior eye
    sw = _switch_node(nt, d.outputs[0], None, None, d.outputs[0], _emission(nt, LABEL_PUPIL), lb_all=1.0)
    nt.links.new(sw.outputs[0], out.inputs["Surface"])
    mat["ec_kind"] = "interior"
    return mat


def tear_material(name: str, roughness: float = 0.03) -> bpy.types.Material:
    """Protected tear meniscus: its own reflection is kept in the clean target."""
    mat = bpy.data.materials.new(name)
    nt = _nt(mat)
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.target = "CYCLES"
    g = nt.nodes.new("ShaderNodeBsdfGlass")
    g.distribution = "GGX"
    g.inputs["IOR"].default_value = 1.336
    g.inputs["Roughness"].default_value = roughness
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    lp = nt.nodes.new("ShaderNodeLightPath")
    beauty = _mix(nt, lp.outputs["Is Shadow Ray"], g.outputs[0], tr.outputs[0])
    sw = _switch_node(nt, beauty, None, None, beauty, _emission(nt, LABEL_TEAR), lb_all=1.0)
    nt.links.new(sw.outputs[0], out.inputs["Surface"])
    mat["ec_kind"] = "tear"
    return mat


def emitter_material(name: str, color_socket_builder=None, color=(1, 1, 1), strength: float = 1.0) -> bpy.types.Material:
    """Emissive light geometry; black for camera rays in reflection/label layers."""
    mat = bpy.data.materials.new(name)
    nt = _nt(mat)
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.target = "CYCLES"
    e = nt.nodes.new("ShaderNodeEmission")
    e.inputs["Color"].default_value = (*color, 1.0)
    e.inputs["Strength"].default_value = strength
    if color_socket_builder is not None:
        sock = color_socket_builder(nt)
        nt.links.new(sock, e.inputs["Color"])
    sw = _switch_node(nt, e.outputs[0], None, None, e.outputs[0], None, lb_all=1.0)
    nt.links.new(sw.outputs[0], out.inputs["Surface"])
    mat["ec_kind"] = "emitter"
    return mat


def plain_material(name: str, color=(0.02, 0.02, 0.02), roughness=0.6, label=None) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    nt = _nt(mat)
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.target = "CYCLES"
    b = nt.nodes.new("ShaderNodeBsdfPrincipled")
    b.inputs["Base Color"].default_value = (*color, 1)
    b.inputs["Roughness"].default_value = roughness
    la = _emission(nt, label) if label else None
    sw = _switch_node(nt, b.outputs[0], None, la, b.outputs[0], None, lb_all=1.0)
    nt.links.new(sw.outputs[0], out.inputs["Surface"])
    return mat


def wrap_existing(mat: bpy.types.Material, category: str) -> None:
    """Insert the switch into an MPFB material. category: 'skin' | 'lash' (alpha-carded hair)."""
    if mat is None or mat.get("ec_wrapped"):
        return
    nt = _nt(mat)
    outs = [n for n in nt.nodes if n.bl_idname == "ShaderNodeOutputMaterial"]
    if not outs:
        return
    out = outs[0]
    if not out.inputs["Surface"].is_linked:
        return
    beauty = out.inputs["Surface"].links[0].from_socket
    alpha = None
    for n in nt.nodes:
        if n.bl_idname == "ShaderNodeBsdfPrincipled" and "Alpha" in n.inputs:
            if n.inputs["Alpha"].is_linked:
                alpha = n.inputs["Alpha"].links[0].from_socket
            elif n.inputs["Alpha"].default_value < 0.999:
                alpha = float(n.inputs["Alpha"].default_value)
            break
    label = LABEL_SKIN if category == "skin" else LABEL_LASH
    if alpha is None:
        la = _emission(nt, label)
        refl = None
        lb = None
    else:
        tr1 = nt.nodes.new("ShaderNodeBsdfTransparent")
        tr2 = nt.nodes.new("ShaderNodeBsdfTransparent")
        tr3 = nt.nodes.new("ShaderNodeBsdfTransparent")
        la = _mix(nt, alpha, tr1.outputs[0], _emission(nt, label))
        refl = _mix(nt, alpha, tr2.outputs[0], None)
        lb = _mix(nt, alpha, tr3.outputs[0], None)
    sw = _switch_node(nt, beauty, refl, la, beauty, lb, lb_all=1.0)
    final = sw.outputs[0]
    if category == "skin":
        # Head faces that intrude into the optical eye volume (MakeHuman lids straddle the stock
        # eyeball) are transparent to every non-camera ray.  Camera rays see the lid exactly as
        # before; rays refracted through the cornea never meet skin inside the eye.
        at = nt.nodes.new("ShaderNodeAttribute")
        at.attribute_type = "GEOMETRY"
        at.attribute_name = "ec_inside_eye"
        lp = nt.nodes.new("ShaderNodeLightPath")
        inv = nt.nodes.new("ShaderNodeMath")
        inv.operation = "SUBTRACT"
        inv.inputs[0].default_value = 1.0
        nt.links.new(lp.outputs["Is Camera Ray"], inv.inputs[1])
        w = nt.nodes.new("ShaderNodeMath")
        w.operation = "MULTIPLY"
        nt.links.new(at.outputs["Fac"], w.inputs[0])
        nt.links.new(inv.outputs[0], w.inputs[1])
        tr = nt.nodes.new("ShaderNodeBsdfTransparent")
        final = _mix(nt, w.outputs[0], final, tr.outputs[0])
    nt.links.new(final, out.inputs["Surface"])
    mat["ec_wrapped"] = True
    mat["ec_kind"] = category


def apply_skin_tone(mat: bpy.types.Material, tone: float) -> int:
    """v2: scale the skin albedo (value, with a slight saturation rise for darker tones) right after
    every diffuse/albedo image texture, including inside node groups.  Returns the number of
    textures adjusted (0 = no albedo texture recognised; the material is left unchanged)."""
    if mat is None or mat.node_tree is None or abs(tone - 1.0) < 1e-3:
        return 0
    done = 0
    seen = set()

    def visit(tree):
        nonlocal done
        if tree.name in seen:
            return
        seen.add(tree.name)
        for n in list(tree.nodes):
            if n.bl_idname == "ShaderNodeTexImage" and n.image is not None:
                key = (n.image.name + " " + (n.image.filepath or "") + " " + n.label + " " + n.name).lower()
                if not any(k in key for k in ("diffuse", "albedo", "basecolor", "base_color")):
                    continue
                links = [l for l in tree.links if l.from_node == n and l.from_socket.name == "Color"]
                if not links:
                    continue
                hs = tree.nodes.new("ShaderNodeHueSaturation")
                hs.inputs["Value"].default_value = float(tone)
                hs.inputs["Saturation"].default_value = 1.0 + (1.0 - float(tone)) * 0.35
                tree.links.new(n.outputs["Color"], hs.inputs["Color"])
                for l in links:
                    tree.links.new(hs.outputs["Color"], l.to_socket)
                done += 1
            elif n.bl_idname == "ShaderNodeGroup" and n.node_tree is not None:
                visit(n.node_tree)

    visit(mat.node_tree)
    mat["ec_skin_tone"] = float(tone)
    return done


def world_switch(world: bpy.types.World, env_socket_builder) -> None:
    """World: environment for beauty/clean (and all non-camera rays); black for camera rays in
    reflection/label layers and for all rays in lab_b."""
    try:
        world.use_nodes = True
    except Exception:
        pass
    nt = world.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputWorld")
    bg = nt.nodes.new("ShaderNodeBackground")
    col, strength = env_socket_builder(nt)
    nt.links.new(col, bg.inputs["Color"])
    if isinstance(strength, float):
        bg.inputs["Strength"].default_value = strength
    else:
        nt.links.new(strength, bg.inputs["Strength"])
    sw = _switch_node(nt, bg.outputs[0], None, None, bg.outputs[0], None, lb_all=1.0)
    nt.links.new(sw.outputs[0], out.inputs["Surface"])

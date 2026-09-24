"""MPFB2 (MakeHuman) head integration and pose controls.

Uses only APIs demonstrated in MPFB v2.0.17 script_samples (HumanService.create_human,
TargetService.load_target/reapply_macro_details, HumanService.add_builtin_rig,
HumanService.set_character_skin, HumanService.add_mhclo_asset, AssetService.find_asset_absolute_path,
LocationService.get_mpfb_data).  The stock MakeHuman eyes are used only to measure the socket and
are then replaced by the project's optical eyes.
"""

from __future__ import annotations

import math
import os

import bpy
import mathutils
from mathutils.bvhtree import BVHTree

import anatomy
import materials as M
from common import enable_mpfb, link, log, mpfb

EYE_BONES = {"L": "eye.L", "R": "eye.R"}
LID_BONES = {"L": ("orbicularis03.L", "orbicularis04.L"), "R": ("orbicularis03.R", "orbicularis04.R")}


class Identity:
    def __init__(self):
        self.body = None
        self.rig = None
        self.proxies = []
        self.eyes = {}
        self.tears = {}
        self.lid_close = {}
        self.frames = {}  # side -> (centre, forward, up, right) of the fitted eyeball at rest
        self.meta = {}


def _eye_frame_world(rig, bone):
    return anatomy.eye_frame_from_bone(rig, bone)


def build_identity(ident: dict, tex: dict, render_subdiv: int = 2) -> Identity:
    enable_mpfb()
    HS = mpfb("mpfb.services.humanservice", "HumanService")
    AS = mpfb("mpfb.services.assetservice", "AssetService")
    TS = mpfb("mpfb.services.targetservice", "TargetService")
    LS = mpfb("mpfb.services.locationservice", "LocationService")
    I = Identity()
    macro = TS.get_default_macro_info_dict()
    for k, v in ident["macro"].items():
        if k == "race":
            for rk, rv in v.items():
                macro["race"][rk] = rv
        else:
            macro[k] = v
    h = HS.create_human(macro_detail_dict=macro)
    troot = LS.get_mpfb_data("targets")
    for rel, w in ident.get("targets", []):
        TS.load_target(h, os.path.join(troot, rel + ".target.gz"), weight=float(w))
    rig = HS.add_builtin_rig(h, "default")
    skin = AS.find_asset_absolute_path(ident["skin"] + ".mhmat", asset_subdir="skins")
    if skin is None:
        raise RuntimeError(f"skin {ident['skin']} not installed")
    HS.set_character_skin(skin, h, skin_type="ENHANCED_SSS")
    # The MakeHuman "enhanced" skins already have eyebrows PAINTED INTO the albedo, and adding the
    # strand eyebrow mesh on top of that renders two brows that do not align - visible as a dark
    # painted arc with a separate hair arc offset below it.  `skip_brow_mesh` drops the mesh so only
    # the painted brow remains, which is aligned with the face by construction.
    assets = [("eyelashes", ident["eyelashes"] + ".mhclo", "Eyelashes"),
              ("eyes", "high-poly.mhclo", "Eyes")]
    if not ident.get("skip_brow_mesh"):
        assets.insert(0, ("eyebrows", ident["eyebrows"] + ".mhclo", "Eyebrows"))
    for sub, fn, at in assets:
        path = AS.find_asset_absolute_path(fn, asset_subdir=sub)
        if path is None:
            raise RuntimeError(f"asset {sub}/{fn} not installed")
        HS.add_mhclo_asset(path, h, asset_type=at)
    I.body, I.rig = h, rig
    sub = h.modifiers.new("ec_subdiv", "SUBSURF")
    sub.levels = 0
    sub.render_levels = render_subdiv
    objs = [o for o in bpy.data.objects if o.parent in (h, rig) and o.type == "MESH" and o is not h]
    proxy_eyes = [o for o in objs if o.name.endswith("high-poly")]
    I.proxies = [o for o in objs if o not in proxy_eyes]
    # --- measure the stock eye socket (robust sphere fit; its corneal bulge is rejected as
    #     outliers), then remove the stock eyes.  The stock eyeball sphere is centred ahead of
    #     the eye bone head, so the fitted centre, not the joint, defines our eye.
    dg = bpy.context.evaluated_depsgraph_get()
    socket = {}
    for side, bone in EYE_BONES.items():
        cb, f, up, right = _eye_frame_world(rig, bone)
        pe = proxy_eyes[0]
        ev = pe.evaluated_get(dg)
        me = ev.to_mesh()
        sgn = 1 if side == "L" else -1
        pts = [pe.matrix_world @ v.co for v in me.vertices if sgn * (pe.matrix_world @ v.co).x > 0]
        ev.to_mesh_clear()
        # the stock eyeball is a partial sphere centred on the joint plus a flatter corneal cap:
        # radius = median joint distance of the scleral part (axial < 60% of the front extent)
        front = max((p - cb).dot(f) for p in pts)
        sc = sorted((p - cb).length for p in pts if (p - cb).dot(f) < 0.6 * front)
        r = sc[len(sc) // 2]
        socket[side] = {"center": [round(x, 6) for x in cb], "radius": r, "front_extent": front,
                        "n_points": len(sc)}
        I.frames[side] = (cb, f, up, right)
    for pe in proxy_eyes:
        bpy.data.objects.remove(pe)
    # --- materials of MPFB assets get the paired-target switch
    tone = float(ident.get("skin_tone", 1.0))
    for slot in h.material_slots:
        M.wrap_existing(slot.material, "skin")
        if abs(tone - 1.0) > 1e-3:
            n = M.apply_skin_tone(slot.material, tone)
            log(f"skin tone {tone:.3f}: adjusted {n} albedo texture(s) in {slot.material.name if slot.material else None}")
    for o in I.proxies:
        for slot in o.material_slots:
            M.wrap_existing(slot.material, "lash")
    # --- build and fit optical eyes
    tear_mat = M.tear_material(ident["identity_id"] + "_tear", ident.get("tear_roughness", 0.03))
    inside_faces: set[int] = set()
    for side, bone in EYE_BONES.items():
        ep = dict(ident["eyes"][side])
        ep["radius"] = socket[side]["radius"] * ep.get("radius_fit", 0.985)
        t = tex[side]
        mats = {
            "cornea": M.cornea_material(f"{ident['identity_id']}_{side}_cornea", ep),
            "sclera": M.sclera_material(f"{ident['identity_id']}_{side}_sclera", ep, t),
            "iris": M.iris_material(f"{ident['identity_id']}_{side}_iris", ep, t),
            "interior": M.interior_material(f"{ident['identity_id']}_{side}_interior"),
        }
        c, f, up, right = I.frames[side]
        x = up.cross(f).normalized()
        frame_m = mathutils.Matrix((x, up, f)).transposed().to_4x4()
        frame_m.translation = c
        pivot = bpy.data.objects.new(f"{ident['identity_id']}_{side}_pivot", None)
        link(pivot)
        pivot.parent = rig
        pivot.parent_type = "BONE"
        pivot.parent_bone = "head"
        pivot.matrix_world = frame_m
        eye = None
        tried = []
        for attempt in range(8):
            eye = anatomy.build_eye(f"{ident['identity_id']}_{side}", ep, mats)
            root = eye["root"]
            root.parent = pivot
            root.matrix_parent_inverse = mathutils.Matrix.Identity(4)
            root.rotation_mode = "QUATERNION"
            root.location = (0, 0, 0)
            root.rotation_quaternion = (1, 0, 0, 0)
            bpy.context.view_layer.update()
            problems = anatomy.validate_eye(eye)
            rest = anatomy.body_positions(h, posed=False)
            touching = anatomy.faces_touching_eye(h, eye, rest)
            bvh = anatomy.body_bvh_excluding(h, set(touching))
            trapped = anatomy.containment_violations(eye, h, bvh=bvh)
            ov = len(anatomy.world_bvh(eye["interior"]).overlap(bvh))
            tried.append({"radius": ep["radius"], "problems": problems, "interior_overlaps": ov,
                          "trapped": trapped, "inside_eye_faces": len(touching)})
            if not problems and ov == 0 and trapped == 0:
                inside_faces |= set(touching)
                eye["validation"] = {"attempts": attempt + 1, "problems": [], "interior_overlaps": 0,
                                     "trapped_rays": 0, "inside_eye_faces": len(touching), "history": tried}
                break
            log(f"eye {side} attempt {attempt}: problems={problems} overlaps={ov} trapped={trapped}; shrinking")
            for key in ("cornea", "sclera", "iris", "interior", "root"):
                ob = eye[key]
                me = ob.data if key != "root" else None
                bpy.data.objects.remove(ob)
                if me is not None and me.users == 0:
                    bpy.data.meshes.remove(me)
            eye = None
            ep["radius"] *= 0.985
        if eye is None:
            raise RuntimeError(f"could not fit a valid {side} eye: {tried}")
        eye["pivot"] = pivot
        eye["params"] = ep
        I.eyes[side] = eye
        I.meta.setdefault("eye_validation", {})[side] = eye.get("validation")
        tear, prof = anatomy.build_tear_meniscus(f"{ident['identity_id']}_{side}_tear", h, rig, I.frames[side],
                                                 ep["radius"], tear_mat, skip_faces=inside_faces)
        I.tears[side] = tear
        I.meta.setdefault("opening_profile", {})[side] = [[round(p, 4), round(a, 4)] for p, a in prof]
    I.meta["socket"] = socket
    I.meta["inside_eye_faces"] = anatomy.mark_inside_eye(h, inside_faces)
    I.inside_faces = inside_faces
    calibrate_lids(I)
    return I


# ----------------------------------------------------------------------------- pose controls
def _pose(rig, name):
    pb = rig.pose.bones[name]
    pb.rotation_mode = "QUATERNION"
    return pb


def set_gaze(I: Identity, side: str, yaw_deg: float, pitch_deg: float) -> None:
    """Gaze relative to the head, rotating the eye about its own centre (the pivot frame):
    +yaw turns the optical axis toward the pivot's +X, +pitch looks up (+Y)."""
    R = (mathutils.Matrix.Rotation(math.radians(yaw_deg), 3, "Y")
         @ mathutils.Matrix.Rotation(math.radians(-pitch_deg), 3, "X"))
    I.eyes[side]["root"].rotation_quaternion = R.to_quaternion()


def _lid_rot(rig, bone, angle_deg):
    pb = rig.pose.bones[bone]
    pb.rotation_mode = "XYZ"
    pb.rotation_euler = (math.radians(angle_deg), 0.0, 0.0)


def _closed_fraction(I: Identity, side: str) -> float:
    """Fraction of central opening rays blocked by the lids (1 = closed)."""
    bvh = anatomy.body_bvh_excluding(I.body, getattr(I, "inside_faces", set()), posed=True)
    c, f, up, right = I.frames[side]
    Rs = I.eyes[side]["params"]["radius"]
    blocked = 0
    n = 0
    for a in (0.0, 4.0, 8.0):
        for phi in range(0, 360, 45) if a > 0 else (0,):
            ar, pr = math.radians(a), math.radians(phi)
            d = (f * math.cos(ar) + (right * math.cos(pr) + up * math.sin(pr)) * math.sin(ar)).normalized()
            hit = bvh.ray_cast(c + d * (Rs * 0.97), d, 0.012)
            blocked += hit[0] is not None
            n += 1
    return blocked / n


def calibrate_lids(I: Identity) -> None:
    """Find per-eye upper-lid bone rotation (sign and magnitude) that closes the eye."""
    for side in ("L", "R"):
        up_b, lo_b = LID_BONES[side]
        best = None
        for sign in (1.0, -1.0):
            for ang in range(4, 64, 3):
                _lid_rot(I.rig, up_b, sign * ang)
                _lid_rot(I.rig, lo_b, -sign * ang * 0.18)
                bpy.context.view_layer.update()
                if _closed_fraction(I, side) >= 0.99:
                    best = (sign, ang + 2.0)
                    break
            _lid_rot(I.rig, up_b, 0.0)
            _lid_rot(I.rig, lo_b, 0.0)
            if best:
                break
        if best is None:
            log(f"WARNING: could not calibrate lid closure for {side}; using default")
            best = (1.0, 40.0)
        I.lid_close[side] = {"sign": best[0], "angle": best[1]}
    bpy.context.view_layer.update()
    I.meta["lid_close"] = I.lid_close


def set_lids(I: Identity, side: str, openness: float, pitch_deg: float = 0.0) -> None:
    """openness 1 = rest open, 0 = closed. The upper lid partly follows downward gaze."""
    up_b, lo_b = LID_BONES[side]
    cal = I.lid_close[side]
    o = min(1.0, max(0.0, openness))
    follow = max(0.0, -pitch_deg) * 0.45 - max(0.0, pitch_deg) * 0.15
    ang = (1.0 - o) * cal["angle"] + follow * (o > 0.0)
    ang = min(cal["angle"], ang)
    _lid_rot(I.rig, up_b, cal["sign"] * ang)
    _lid_rot(I.rig, lo_b, -cal["sign"] * (1.0 - o) * cal["angle"] * 0.18)


def set_head(I: Identity, yaw: float, pitch: float, roll: float) -> None:
    rig = I.rig
    parts = (("head", 0.6), ("neck03", 0.2), ("neck02", 0.1), ("neck01", 0.1))
    for name, w in parts:
        if name not in rig.pose.bones:
            continue
        b = rig.data.bones[name]
        rest = b.matrix_local.to_3x3()
        R = (mathutils.Matrix.Rotation(math.radians(yaw * w), 3, "Z")
             @ mathutils.Matrix.Rotation(math.radians(pitch * w), 3, "X")
             @ mathutils.Matrix.Rotation(math.radians(roll * w), 3, "Y"))
        _pose(rig, name).rotation_quaternion = (rest.inverted() @ R @ rest).to_quaternion()


def eye_world_frame(I: Identity, side: str):
    """Current (posed) eye root world matrix."""
    return I.eyes[side]["root"].matrix_world.copy()

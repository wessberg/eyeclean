"""Procedural optical eye construction and validation (synthetic modeling assumptions).

Local eye frame: origin = eye rotation centre, +Z = optical axis (forward), +Y = up, +X = right.
All lengths are metres.  Default ratios follow common schematic-eye values scaled to the socket
of the head (cornea radius ~0.65 R_s, Q ~ -0.26, limbus radius ~0.49 R_s, iris recessed behind
the limbus plane, effective refractive index 1.3375, tear-film surface IOR 1.376/1.336).  These
are plausible priors for CG supervision, not clinically validated anatomy.
"""

from __future__ import annotations

import math

import bmesh
import bpy
import mathutils
from mathutils.bvhtree import BVHTree
from mathutils.kdtree import KDTree

from common import link, log


def cornea_sag(h: float, Rc: float, Q: float) -> float:
    return h * h / (Rc * (1.0 + math.sqrt(max(1e-9, 1.0 - (1.0 + Q) * h * h / (Rc * Rc)))))


def derive_geometry(p: dict) -> dict:
    Rs = p["radius"]
    rl = p["limbus_ratio"] * Rs
    Rc = p["cornea_R_ratio"] * Rs
    Q = p["asphericity"]
    zl = math.sqrt(Rs * Rs - rl * rl)
    z_apex = zl + cornea_sag(rl, Rc, Q)
    zi = zl - p["iris_recess_ratio"] * Rs
    rio = math.sqrt(Rs * Rs - zi * zi) * 0.992
    g = dict(Rs=Rs, rl=rl, Rc=Rc, Q=Q, zl=zl, z_apex=z_apex, zi=zi, rio=rio,
             bulge=p["iris_bulge_ratio"] * Rs,
             p_min=p["pupil_min_ratio"] * rl, p_max=p["pupil_max_ratio"] * rl)
    return g


def _mesh(name, verts, faces, uvs=None):
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    if uvs is not None:
        uv = me.uv_layers.new(name="UVMap")
        k = 0
        for poly in me.polygons:
            for li in poly.loop_indices:
                uv.data[li].uv = uvs[k]
                k += 1
    me.polygons.foreach_set("use_smooth", [True] * len(me.polygons))
    me.update()
    return me


def build_cornea(name, g, n_r=40, n_t=128):
    verts, faces, uvs = [], [], []
    verts.append((0.0, 0.0, g["z_apex"]))
    for i in range(1, n_r + 1):
        h = g["rl"] * i / n_r
        z = g["z_apex"] - cornea_sag(h, g["Rc"], g["Q"])
        if i == n_r:
            z = g["zl"]  # exact joint with the scleral opening
        for j in range(n_t):
            t = 2 * math.pi * j / n_t
            verts.append((h * math.cos(t), h * math.sin(t), z))
    for j in range(n_t):
        faces.append((0, 1 + j, 1 + (j + 1) % n_t))
    for i in range(n_r - 1):
        a, b = 1 + i * n_t, 1 + (i + 1) * n_t
        for j in range(n_t):
            j2 = (j + 1) % n_t
            faces.append((a + j, b + j, b + j2, a + j2))
    return _mesh(name, verts, faces)


def build_sclera(name, g, n_r=64, n_t=128):
    Rs, zl = g["Rs"], g["zl"]
    th_l = math.acos(zl / Rs)
    verts, faces, uvs = [], [], []
    rows = n_r + 1
    for i in range(rows):
        th = th_l + (math.pi - th_l) * i / n_r
        for j in range(n_t + 1):  # duplicated seam column for UVs
            t = 2 * math.pi * j / n_t
            verts.append((Rs * math.sin(th) * math.cos(t), Rs * math.sin(th) * math.sin(t), Rs * math.cos(th)))
    vcoord = lambda i: (th_l + (math.pi - th_l) * i / n_r - th_l) / (math.pi / 2 - th_l)  # 0 limbus -> 1 equator
    for i in range(n_r):
        for j in range(n_t):
            a = i * (n_t + 1) + j
            b = (i + 1) * (n_t + 1) + j
            faces.append((a, b, b + 1, a + 1))  # e_theta x e_phi = +e_r: outward normals
            u0, u1 = j / n_t, (j + 1) / n_t
            uvs += [(u0, vcoord(i)), (u0, vcoord(i + 1)), (u1, vcoord(i + 1)), (u1, vcoord(i))]
    # seam duplicates share positions but keep separate UVs for texture continuity
    return _mesh(name, verts, faces, uvs)


def build_iris(name, g, n_r=48, n_t=128):
    """Annulus from the pupil margin (rho=0) to the iris root (rho=1), UV=(theta/2pi, rho).
    Basis = mid pupil; shape key 'pupil' maps 0->p_min, 1->p_max (texture stretches radially)."""
    def ring_pos(rp):
        pts = []
        for i in range(n_r + 1):
            rho = i / n_r
            r = rp + rho * (g["rio"] - rp)
            z = g["zi"] + g["bulge"] * (1.0 - rho) ** 1.5
            for j in range(n_t + 1):
                t = 2 * math.pi * j / n_t
                pts.append((r * math.cos(t), r * math.sin(t), z))
        return pts
    base = ring_pos(0.5 * (g["p_min"] + g["p_max"]))
    faces, uvs = [], []
    for i in range(n_r):
        for j in range(n_t):
            a = i * (n_t + 1) + j
            b = (i + 1) * (n_t + 1) + j
            faces.append((a, b, b + 1, a + 1))  # +Z facing
            uvs += [(j / n_t, i / n_r), (j / n_t, (i + 1) / n_r), ((j + 1) / n_t, (i + 1) / n_r), ((j + 1) / n_t, i / n_r)]
    me = _mesh(name, base, faces, uvs)
    obj = bpy.data.objects.new(name, me)
    obj.shape_key_add(name="Basis", from_mix=False)
    kmin = obj.shape_key_add(name="pupil_min", from_mix=False)
    kmax = obj.shape_key_add(name="pupil_max", from_mix=False)
    pmin, pmax = ring_pos(g["p_min"]), ring_pos(g["p_max"])
    for k, v in enumerate(pmin):
        kmin.data[k].co = v
    for k, v in enumerate(pmax):
        kmax.data[k].co = v
    return obj


def set_pupil(iris_obj, frac: float) -> None:
    """frac in [0,1]: 0 = smallest pupil, 1 = largest."""
    kb = iris_obj.data.shape_keys.key_blocks
    frac = min(1.0, max(0.0, frac))
    # basis is the mid pupil: blend min/max keys relative to basis
    kb["pupil_min"].value = max(0.0, 1.0 - 2.0 * frac)
    kb["pupil_max"].value = max(0.0, 2.0 * frac - 1.0)


def interior_profile(g, depth_ratio: float = 0.30):
    """(r, z) profile of the dark posterior capsule: open front ring sealed to the iris root just
    behind the iris plane, straight wall back to z_b, then a hemispherical cap.  Every point stays
    within ~0.87 R_s of the eye centre, i.e. inside the head's socket cavity, so rays through the
    pupil always meet this capsule before any head geometry."""
    ri = g["rio"] * 0.995
    z0 = g["zi"] - 0.004 * g["Rs"]
    zb = -depth_ratio * g["Rs"]
    prof = []
    n_wall = 12
    for i in range(n_wall + 1):
        prof.append((ri, z0 + (zb - z0) * i / n_wall))
    n_cap = 16
    for i in range(1, n_cap + 1):
        a = (math.pi / 2) * i / n_cap
        prof.append((ri * math.cos(a), zb - ri * math.sin(a)))
    return prof


def build_interior(name, g, n_t=64, depth_ratio: float = 0.30):
    prof = interior_profile(g, depth_ratio)
    verts, faces = [], []
    for r, z in prof:
        for j in range(n_t):
            t = 2 * math.pi * j / n_t
            verts.append((r * math.cos(t), r * math.sin(t), z))
    for i in range(len(prof) - 1):
        a, b = i * n_t, (i + 1) * n_t
        for j in range(n_t):
            j2 = (j + 1) % n_t
            faces.append((a + j, b + j, b + j2, a + j2))  # faces the capsule interior
    return _mesh(name, verts, faces)


def inside_eye_local(p, g, margin: float = 0.999) -> bool:
    """Point (eye-local, rest gaze) inside the closed optical eye volume: scleral sphere or corneal cap."""
    if p.length < g["Rs"] * margin:
        return True
    h = math.hypot(p.x, p.y)
    if p.z > g["zl"] and h < g["rl"]:
        return p.z < g["z_apex"] - cornea_sag(h, g["Rc"], g["Q"]) - 1e-6
    return False


INSIDE_EYE_ATTR = "ec_inside_eye"


def faces_touching_eye(body, eye: dict, rest_pos: list, margin: float = 1.03) -> list[int]:
    """Original body faces with any vertex inside the (slightly enlarged) optical eye volume.
    MakeHuman lids straddle the stock eyeball to hide the seam; these faces must not be seen by
    rays that travel inside our refractive eye."""
    inv = eye["root"].matrix_world.inverted()
    g = dict(eye["geom"])
    for k in ("Rs", "rl", "z_apex", "zl", "Rc"):
        g[k] = g[k] * margin
    inside = [inside_eye_local(inv @ p, g) for p in rest_pos]
    return [f.index for f in body.data.polygons if any(inside[v] for v in f.vertices)]


def mark_inside_eye(body, face_indices: set[int]) -> int:
    """FLOAT face attribute read by the skin shader (transparent to non-camera rays)."""
    me = body.data
    attr = me.attributes.get(INSIDE_EYE_ATTR) or me.attributes.new(INSIDE_EYE_ATTR, "FLOAT", "FACE")
    vals = [0.0] * len(me.polygons)
    for i in face_indices:
        vals[i] = 1.0
    attr.data.foreach_set("value", vals)
    me.update()
    return len(face_indices)


def body_positions(body, posed: bool) -> list:
    """World positions of original body vertices (shape keys; armature optional; no mask/subdiv)."""
    saved = [(m, m.show_viewport) for m in body.modifiers]
    for m, _ in saved:
        m.show_viewport = posed and m.type == "ARMATURE"
    try:
        dg = bpy.context.evaluated_depsgraph_get()
        ev = body.evaluated_get(dg)
        me = ev.to_mesh()
        pos = [body.matrix_world @ v.co for v in me.vertices]
        ev.to_mesh_clear()
    finally:
        for m, vis in saved:
            m.show_viewport = vis
    return pos


def body_bvh_excluding(body, skip: set[int], posed: bool = False) -> BVHTree:
    """World-space BVH of the rendered body surface (helpers excluded) minus faces in ``skip``."""
    pos = body_positions(body, posed)
    names = {g.index: g.name for g in body.vertex_groups}
    body_idx = [k for k, n in names.items() if n == "body"]
    in_body = None
    if body_idx:
        bi = body_idx[0]
        in_body = [any(ge.group == bi and ge.weight > 0.5 for ge in v.groups) for v in body.data.vertices]
    polys = []
    for f in body.data.polygons:
        if f.index in skip:
            continue
        if in_body is not None and not all(in_body[v] for v in f.vertices):
            continue  # masked helper geometry is not rendered
        polys.append(tuple(f.vertices))
    return BVHTree.FromPolygons(pos, polys)


def containment_violations(eye: dict, body, samples: int = 160, bvh: BVHTree | None = None) -> int:
    """Count rays from the eye centre to interior-capsule, iris and cornea sample points that hit
    head geometry first (head surfaces trapped inside the eye would be seen through the cornea)."""
    if bvh is None:
        bvh = world_bvh(body)
    root = eye["root"].matrix_world
    c = root.translation
    pts = [v.co for v in eye["interior"].data.vertices][:: max(1, len(eye["interior"].data.vertices) // samples)]
    pts += [v.co for v in eye["iris"].data.vertices][:: max(1, len(eye["iris"].data.vertices) // (samples // 2))]
    pts += [v.co for v in eye["cornea"].data.vertices][:: max(1, len(eye["cornea"].data.vertices) // (samples // 2))]
    bad = 0
    for p in pts:
        w = root @ p
        d = w - c
        L = d.length
        if L < 1e-9:
            continue
        hit = bvh.ray_cast(c, d / L, L * 0.999)
        if hit[0] is not None:
            bad += 1
    return bad


def build_eye(prefix: str, p: dict, mats: dict, coll=None) -> dict:
    g = derive_geometry(p)
    root = bpy.data.objects.new(prefix + "_root", None)
    link(root, coll)
    objs = {"root": root, "geom": g}
    cornea = bpy.data.objects.new(prefix + "_cornea", build_cornea(prefix + "_cornea", g))
    sclera = bpy.data.objects.new(prefix + "_sclera", build_sclera(prefix + "_sclera", g))
    iris = build_iris(prefix + "_iris", g)
    interior = bpy.data.objects.new(prefix + "_interior", build_interior(prefix + "_interior", g))
    for key, o in (("cornea", cornea), ("sclera", sclera), ("iris", iris), ("interior", interior)):
        o.data.materials.append(mats[key])
        link(o, coll)
        o.parent = root
        objs[key] = o
    set_pupil(iris, 0.5)
    for k, v in g.items():
        root["ec_" + k] = float(v)
    return objs


def validate_eye(objs: dict) -> list[str]:
    """Automatic physical sanity checks; returns a list of problems (empty = valid)."""
    g = objs["geom"]
    problems = []
    if not (0.3 * g["Rs"] < g["rl"] < 0.6 * g["Rs"]):
        problems.append("limbus radius outside plausible range")
    if g["z_apex"] <= g["zl"]:
        problems.append("cornea does not protrude")
    if g["zi"] >= g["zl"]:
        problems.append("iris not recessed behind limbus plane")
    # corneal/iris spacing along the axis and at the limbus
    for h_frac in (0.0, 0.5, 0.9):
        h = g["rl"] * h_frac
        zc = g["z_apex"] - cornea_sag(h, g["Rc"], g["Q"])
        rho = max(0.0, (h - g["p_max"]) / (g["rio"] - g["p_max"]))
        zi = g["zi"] + g["bulge"] * (1.0 - rho) ** 1.5
        if zc - zi < 0.08 * g["Rs"] * (1.0 - h_frac) + 0.02 * g["Rs"] * h_frac:
            problems.append(f"insufficient corneal/iris spacing at h={h_frac:.1f} ({(zc - zi) * 1000:.3f} mm)")
    if g["p_max"] >= 0.8 * g["rl"] or g["p_min"] <= 0.1 * g["rl"]:
        problems.append("pupil range implausible")
    # normals face outward (cornea/sclera)
    for key in ("cornea", "sclera"):
        me = objs[key].data
        bad = 0
        for poly in me.polygons:
            if poly.normal.dot(poly.center) <= 0:
                bad += 1
        if bad:
            problems.append(f"{key}: {bad} inward-facing polygons")
    # continuity at the limbus: cornea outer ring vs sclera first ring
    cme, sme = objs["cornea"].data, objs["sclera"].data
    c_ring = [v.co for v in cme.vertices[-128:]]
    s_ring = [v.co for v in sme.vertices[:129]]
    dmax = max(min((c - s).length for s in s_ring) for c in c_ring)
    if dmax > 1e-6:
        problems.append(f"cornea/sclera limbus seam gap {dmax:.2e} m")
    return problems


# --------------------------------------------------------------------- lid margin / tear film
def eye_frame_from_bone(rig, bone_name: str):
    b = rig.data.bones[bone_name]
    c = rig.matrix_world @ b.head_local
    f = ((rig.matrix_world @ b.tail_local) - c).normalized()
    up = mathutils.Vector((0, 0, 1))
    right = f.cross(up).normalized()
    up2 = right.cross(f).normalized()
    return c, f, up2, right


def opening_profile(bvh: BVHTree, c, f, up, right, Rs: float, n_phi: int = 72, max_deg: float = 80.0) -> list[tuple[float, float]]:
    """For each azimuth phi around the optical axis, the largest polar angle alpha whose ray from
    the eyeball surface outwards is not blocked by the lids within 12 mm."""
    prof = []
    for k in range(n_phi):
        phi = 2 * math.pi * k / n_phi
        amax = 0.0
        a = 0.0
        while a < max_deg:
            ar = math.radians(a)
            d = (f * math.cos(ar) + (right * math.cos(phi) + up * math.sin(phi)) * math.sin(ar)).normalized()
            hit = bvh.ray_cast(c + d * (Rs * 0.97), d, 0.012)
            if hit[0] is not None:
                break
            amax = a
            a += 0.5
        prof.append((phi, math.radians(amax)))
    return prof


def build_tear_meniscus(name: str, body, rig, frame: tuple, Rs: float, mat, coll=None, radius_ratio: float = 0.025,
                        skip_faces: set[int] | None = None) -> tuple:
    """Thin water tube along the lower lid margin, weighted to the lid so it follows blinks.
    frame = (centre, forward, up, right) of the fitted eyeball at rest (world)."""
    bvh = body_bvh_excluding(body, skip_faces or set())
    c, f, up, right = frame
    prof = opening_profile(bvh, c, f, up, right, Rs)
    pts = []
    for phi, a in prof:
        s = math.sin(phi)
        if s > -0.25:  # lower lid only (below the horizontal), excluding the canthi
            continue
        d = (f * math.cos(a) + (right * math.cos(phi) + up * math.sin(phi)) * math.sin(a)).normalized()
        pts.append((phi, c + d * (Rs * 1.004)))
    pts.sort(key=lambda t: t[0])
    if len(pts) < 4:
        return None, prof
    r = Rs * radius_ratio
    verts, faces = [], []
    n_s = 8
    P = [p for _, p in pts]
    for i, p in enumerate(P):
        t = (P[min(i + 1, len(P) - 1)] - P[max(i - 1, 0)]).normalized()
        n1 = (p - c).normalized()
        n2 = t.cross(n1).normalized()
        for k in range(n_s):
            ang = 2 * math.pi * k / n_s
            verts.append(p + (n1 * math.cos(ang) + n2 * math.sin(ang)) * r)
    for i in range(len(P) - 1):
        for k in range(n_s):
            k2 = (k + 1) % n_s
            a0, a1 = i * n_s, (i + 1) * n_s
            faces.append((a0 + k, a1 + k, a1 + k2, a0 + k2))
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(v) for v in verts], [], faces)
    me.polygons.foreach_set("use_smooth", [True] * len(me.polygons))
    obj = bpy.data.objects.new(name, me)
    obj.data.materials.append(mat)
    link(obj, coll)
    # copy armature weights from the nearest rest-pose body vertex (modifiers disabled so the
    # evaluated vertex indices match the original mesh)
    saved = [(m, m.show_viewport) for m in body.modifiers]
    for m, _ in saved:
        m.show_viewport = False
    dg = bpy.context.evaluated_depsgraph_get()
    bev = body.evaluated_get(dg)
    bme = bev.to_mesh()
    kd = KDTree(len(bme.vertices))
    for v in bme.vertices:
        kd.insert(body.matrix_world @ v.co, v.index)
    kd.balance()
    groups = {vg.index: vg.name for vg in body.vertex_groups}
    for vi, v in enumerate(verts):
        _, idx, _ = kd.find(v)
        for ge in body.data.vertices[idx].groups:
            gname = groups[ge.group]
            if gname not in obj.vertex_groups:
                obj.vertex_groups.new(name=gname)
            obj.vertex_groups[gname].add([vi], ge.weight, "REPLACE")
    bev.to_mesh_clear()
    for m, vis in saved:
        m.show_viewport = vis
    mod = obj.modifiers.new("Armature", "ARMATURE")
    mod.object = rig
    return obj, prof


def world_bvh(obj, dg=None) -> BVHTree:
    """BVH in WORLD coordinates (BVHTree.FromObject uses object-local coordinates)."""
    dg = dg or bpy.context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(dg)
    me = ev.to_mesh()
    mw = obj.matrix_world
    verts = [mw @ v.co for v in me.vertices]
    polys = [tuple(p.vertices) for p in me.polygons]
    ev.to_mesh_clear()
    return BVHTree.FromPolygons(verts, polys)


def overlap_count(a, b) -> int:
    dg = bpy.context.evaluated_depsgraph_get()
    return len(world_bvh(a, dg).overlap(world_bvh(b, dg)))


def fit_sphere(points: list) -> tuple:
    """Least-squares sphere |p|^2 = 2 c.p + k (tiny 4x4 normal equations; metadata math)."""
    A = [[2 * p.x, 2 * p.y, 2 * p.z, 1.0] for p in points]
    b = [p.length_squared for p in points]
    AtA = [[sum(A[k][i] * A[k][j] for k in range(len(A))) for j in range(4)] for i in range(4)]
    Atb = [sum(A[k][i] * b[k] for k in range(len(A))) for i in range(4)]
    m = mathutils.Matrix(AtA)
    sol = m.inverted() @ mathutils.Vector(Atb)
    c = mathutils.Vector(sol[:3])
    r = math.sqrt(max(1e-12, sol[3] + c.length_squared))
    return c, r


def robust_sphere(points: list, iters: int = 3, tol_ratio: float = 0.02) -> tuple:
    pts = list(points)
    c, r = fit_sphere(pts)
    for _ in range(iters):
        keep = [p for p in pts if abs((p - c).length - r) < tol_ratio * r]
        if len(keep) < 20:
            break
        c, r = fit_sphere(keep)
        pts = keep
    return c, r

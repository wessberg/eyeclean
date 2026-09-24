"""Deterministic dataset planning: identities, scenes, stills, clips and demo cases.

Split integrity: identities, their textures (iris/sclera seeds), MakeHuman skin textures, scene
seeds and whole clips are drawn from split-specific seed streams and never shared.  The
held-out iris structure family (``spoke``) and light family (``ring_light``) appear only in the
test split.  Blinks and partial lid closure are sampled explicitly (user requirement): clips get
natural blink events, stills sample open / partly closed / nearly closed / closed lids.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass

from ..procedural.iris import COLOR_FAMILIES, HELDOUT_STRUCTURE, STRUCTURE_FAMILIES, sample_identity_iris, sample_sclera

SPLITS = ("train", "validation", "test")
SPLIT_TAG = {"train": "tr", "validation": "va", "test": "te"}

ALL_SKINS = [
    "blindsaypatten_uniform_skin_texture", "bobby_03_young_female_hairless", "callharvey3d_midtoned_female",
    "cutoff3d_indian_female_enhanced", "cutoff3d_indian_female_skin", "darthfurby_caucasian_female", "flower-angel_red_head_skin",
    "jartur69_middleage_slavic_male_with_genitals_and_beard", "jartur69_old_slavic_male_with_genitals_and_beard",
    "ken1138_caucasian_male_tattooed_skin", "middleage_african_female", "middleage_african_male", "middleage_asian_female",
    "middleage_asian_male", "middleage_caucasian_female", "middleage_caucasian_male", "mindfront_aksel_skin",
    "mindfront_skin_male_african_middleage", "nyloseth_zoeyskin", "old_african_female", "old_african_male", "old_asian_female",
    "old_asian_male", "old_caucasian_female", "old_caucasian_male", "onlytheghosts_middle_aged_eurasian_female",
    "onlytheghosts_old_eurasian_female", "onlytheghosts_old_eurasian_male", "onlytheghosts_young_eurasian_female",
    "rehmanpolanski_skin_viking_tattoos", "saltycowdawg_genuine_caucasian_female_with_genitals", "skalldyrssuppe_creamy_female",
    "spreadcore_alana_caucasian_female_with_genitals_skin", "toigo_light_skin_female_bronze", "toigo_light_skin_female_bronze_with_makeup",
    "toigo_light_skin_female_freckles", "toigo_light_skin_female_ginger", "toigo_light_skin_female_ginger_2",
    "toigo_light_skin_female_ginger_with_makeup", "toigo_light_skin_female_with_violet_makeup", "toigo_light_skin_male_bronze",
    "toigo_light_skin_male_freckles", "toigo_light_skin_male_ginger", "toigo_light_skin_male_with_emo_eyes",
    "toigo_light_skin_male_with_eyeliner", "toigo_light_skin_male_with_goth_makeup", "toigo_light_skin_with_makeup",
    "toigo_light_skin_with_natural_makeup", "young_african_female", "young_african_male", "young_asian_female", "young_asian_male",
    "young_caucasian_female", "young_caucasian_female2", "young_caucasian_male", "young_caucasian_male2",
]  # 'toon01' and costume '*_special_suit' skins excluded (not natural skin)


def skin_meta(name: str) -> dict:
    """Sex / age group / ancestry / cosmetics tags parsed from the MakeHuman skin names (v2 matching)."""
    n = name.lower()
    sex = "f" if ("female" in n or any(k in n for k in ("zoey", "creamy", "alana", "red_head", "bobby"))) else ("m" if ("male" in n or "aksel" in n or "viking" in n) else "any")
    age = "old" if "old" in n else ("middle" if "middle" in n else ("young" if "young" in n else "any"))
    anc = next((a for a in ("african", "asian", "indian", "eurasian", "caucasian", "slavic") if a in n), "any")
    if anc == "slavic":
        anc = "caucasian"
    costume = any(k in n for k in ("eyeliner", "emo", "goth", "violet", "tattoo", "viking"))
    natural_makeup = not costume and "makeup" in n  # 'with_makeup', 'natural_makeup', '..._with_makeup'
    return {"sex": sex, "age": age, "ancestry": anc, "cosmetic": costume or natural_makeup, "costume": costume,
            "natural_makeup": natural_makeup}


# MakeHuman macro age: 0 -> 1 y, 0.1875 -> 11 y, 0.5 -> 25 y, 1 -> 90 y (piecewise linear)
def years_to_macro_age(y: float) -> float:
    if y <= 11:
        return max(0.0, (y - 1) / 10 * 0.1875)
    if y <= 25:
        return 0.1875 + (y - 11) / 14 * 0.3125
    return min(1.0, 0.5 + (y - 25) / 65 * 0.5)


AGE_BINS = (("child", 6, 12), ("20s", 20, 29.9), ("30s", 30, 39.9), ("40s", 40, 49.9), ("50s", 50, 59.9), ("60s", 60, 69.9),
            ("70s", 70, 79.9), ("80s", 80, 89), ("90", 89, 90))  # MakeHuman's age range ends at 90 years


EYEBROWS = [f"eyebrow{i:03d}" for i in range(1, 13)]
EYELASHES = [f"eyelashes{i:02d}" for i in range(1, 5)]
EYE_TARGET_CATEGORIES = ["eye-height1", "eye-height2", "eye-height3", "eye-push1", "eye-push2", "eye-scale", "eye-trans-in-out",
                         "eye-trans-up-down", "eye-epicanthus", "eye-eyefold", "eye-eyefold-angle", "eye-bag", "eye-corner1", "eye-corner2"]
_TARGET_PAIRS = {
    "eye-height1": ("height1-decr", "height1-incr"), "eye-height2": ("height2-decr", "height2-incr"),
    "eye-height3": ("height3-decr", "height3-incr"), "eye-push1": ("push1-in", "push1-out"), "eye-push2": ("push2-in", "push2-out"),
    "eye-scale": ("scale-decr", "scale-incr"), "eye-trans-in-out": ("trans-in", "trans-out"), "eye-trans-up-down": ("trans-down", "trans-up"),
    "eye-epicanthus": ("epicanthus-in", "epicanthus-out"), "eye-eyefold": ("eyefold-down", "eyefold-up"),
    "eye-eyefold-angle": ("eyefold-angle-down", "eyefold-angle-up"), "eye-bag": ("bag-decr", "bag-incr"),
    "eye-corner1": ("corner1-down", "corner1-up"), "eye-corner2": ("corner2-down", "corner2-up"),
}
TRAIN_LIGHTS = {"rect_softbox": 3.0, "octa_softbox": 1.5, "strip": 1.5, "window": 1.5, "grid_panel": 1.0, "bulb": 2.0,
                "colored_led": 1.5, "sun": 0.8, "flame": 0.8, "emissive_card": 1.0}
HELDOUT_LIGHT = "ring_light"


def stable_seed(*parts) -> int:
    h = hashlib.sha256("/".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(h[:8], "little") & 0x7FFFFFFF


def partition_skins(seed: int) -> dict:
    r = random.Random(stable_seed(seed, "skins"))
    skins = sorted(ALL_SKINS)
    r.shuffle(skins)
    return {"test": skins[:8], "validation": skins[8:14], "train": skins[14:]}


def _cct_rgb(kelvin: float) -> list[float]:
    """Approximate blackbody RGB (normalized) for light colours (metadata only)."""
    t = kelvin / 100.0
    r = 1.0 if t <= 66 else min(1.0, 1.2929 * ((t - 60) ** -0.1332))
    g = min(1.0, max(0.0, 0.3901 * math.log(t) - 0.6318)) if t <= 66 else min(1.0, 1.1298 * ((t - 60) ** -0.0755))
    b = 1.0 if t >= 66 else (0.0 if t <= 19 else min(1.0, max(0.0, 0.5432 * math.log(t - 10) - 1.1963)))
    m = max(r, g, b)
    return [round(r / m, 4), round(g / m, 4), round(b / m, 4)]


def sample_identity(split: str, idx: int, base_seed: int, skins: list[str], family_override: str | None = None,
                    color_override: str | None = None, demographics: dict | None = None) -> dict:
    s = stable_seed(base_seed, "identity", split, idx)
    r = random.Random(s)
    iid = f"{SPLIT_TAG[split]}{idx:04d}"
    race = [r.random() + 1e-3 for _ in range(3)]
    tot = sum(race)
    macro = {"gender": r.random(), "age": r.uniform(0.42, 0.95), "muscle": r.uniform(0.3, 0.75), "weight": r.uniform(0.3, 0.75),
             "height": r.uniform(0.3, 0.7), "proportions": r.uniform(0.35, 0.75),
             "race": {"african": race[0] / tot, "asian": race[1] / tot, "caucasian": race[2] / tot}}
    targets = []
    for cat in r.sample(EYE_TARGET_CATEGORIES, k=r.randint(2, 6)):
        neg, pos = _TARGET_PAIRS[cat]
        w = r.uniform(-0.8, 0.8)
        asym = r.uniform(-0.25, 0.25) if r.random() < 0.4 else 0.0  # asymmetric lids
        for side in ("l", "r"):
            ww = w + (asym if side == "l" else -asym)
            name = pos if ww > 0 else neg
            if abs(ww) > 0.05:
                targets.append([f"eyes/{side}-eye-{name}", round(abs(ww), 3)])
    allow_heldout = split == "test"
    fam = family_override
    if fam is None and allow_heldout and r.random() < 0.3:
        fam = HELDOUT_STRUCTURE
    iris = sample_identity_iris(stable_seed(s, "iris"), allow_heldout, fam, color_override)
    sclera = sample_sclera(stable_seed(s, "sclera"))
    base_eye = dict(
        radius_fit=0.985, limbus_ratio=r.uniform(0.46, 0.52), cornea_R_ratio=r.uniform(0.62, 0.69), asphericity=r.uniform(-0.4, -0.12),
        iris_recess_ratio=r.uniform(0.03, 0.06), iris_bulge_ratio=r.uniform(0.01, 0.03), pupil_min_ratio=r.uniform(0.18, 0.27),
        pupil_max_ratio=r.uniform(0.5, 0.64), n_eff=1.3375, ior_surface=1.376, ior_tear=1.336, cornea_roughness=r.uniform(0.004, 0.03),
        sclera_sss=r.uniform(0.3, 0.7), sclera_sss_scale=r.uniform(0.001, 0.004), iris_bump=r.uniform(0.1, 0.4))
    eyes = {}
    for side in ("L", "R"):
        e = dict(base_eye)
        e["limbus_ratio"] = base_eye["limbus_ratio"] * r.uniform(0.985, 1.015)
        e["cornea_roughness"] = base_eye["cornea_roughness"] * r.uniform(0.8, 1.2)
        eyes[side] = e
    ident = {
        "identity_id": iid, "split": split, "seed": s, "macro": macro, "targets": targets,
        "skin": r.choice(skins), "eyebrows": r.choice(EYEBROWS), "eyelashes": r.choice(EYELASHES),
        "eyes": eyes, "iris": iris, "sclera": sclera, "tear_roughness": r.uniform(0.015, 0.05),
        "iris_family": iris["L"]["structure"], "iris_color": iris["L"]["color"],
    }
    if demographics:
        _apply_demographics(ident, idx, skins, demographics, random.Random(stable_seed(s, "demographics")))
    return ident


def _apply_demographics(ident: dict, idx: int, skins: list[str], dm: dict, rd: random.Random) -> None:
    """v2: balanced age bins, female/male heads, dominant ancestry and a skin texture matched to
    sex / age / ancestry (relaxed step by step when the split's skin pool has no exact match)."""
    macro = ident["macro"]
    years = None
    if dm["ages"] == "child_to_90s_balanced":
        bin_name, lo, hi = AGE_BINS[idx % len(AGE_BINS)]
        years = rd.uniform(lo, hi)
        macro["age"] = round(years_to_macro_age(years), 4)
        ident["age_bin"] = bin_name
    sex = None
    if dm["sex"] == "female_male":
        sex = "f" if (idx // len(AGE_BINS)) % 2 == 0 else "m"  # alternates within every age bin
        macro["gender"] = round(rd.uniform(0.0, 0.08) if sex == "f" else rd.uniform(0.92, 1.0), 4)
    anc = rd.choice(["african", "asian", "caucasian"])
    dom = rd.uniform(0.65, 0.95)
    others = [a for a in ("african", "asian", "caucasian") if a != anc]
    split_o = rd.random()
    macro["race"] = {anc: dom, others[0]: (1 - dom) * split_o, others[1]: (1 - dom) * (1 - split_o)}
    if years is not None and years < 14:  # children: slimmer build ranges
        macro["muscle"], macro["weight"] = round(rd.uniform(0.3, 0.55), 4), round(rd.uniform(0.3, 0.6), 4)
    want_age = None if years is None else ("young" if years < 36 else ("middle" if years < 60 else "old"))
    # plausible skin ages: never wrinkled "old" textures on children / young adults
    age_ok_set = {None: None, "young": ("young", "any"), "middle": ("middle", "any", "young", "old"), "old": ("old", "any", "middle")}[want_age]
    anc_ok = {"african": ("african",), "asian": ("asian", "eurasian"), "caucasian": ("caucasian", "eurasian", "indian")}[anc]
    pool = [k for k in skins if not (dm["exclude_makeup_and_tattoo_skins"] and skin_meta(k)["cosmetic"])] or list(skins)
    p_mk = float(dm.get("natural_makeup_prob_female", 0.0))
    if sex == "f" and p_mk > 0 and rd.random() < p_mk:  # everyday eye makeup (mascara/liner) on a share of women
        mk = [k for k in skins if skin_meta(k)["natural_makeup"] and skin_meta(k)["sex"] in ("f", "any")]
        if mk:
            pool = mk
            ident["natural_makeup"] = True
    if dm["match_skin_to_age_and_ancestry"]:
        sex_ok = lambda m: sex is None or m["sex"] in (sex, "any")  # noqa: E731
        exact_age = lambda m: want_age is None or m["age"] in (want_age, "any")  # noqa: E731
        age_ok = lambda m: age_ok_set is None or m["age"] in age_ok_set  # noqa: E731
        tests = [lambda m: sex_ok(m) and exact_age(m) and m["ancestry"] in anc_ok,
                 lambda m: sex_ok(m) and age_ok(m) and m["ancestry"] in anc_ok,
                 lambda m: sex_ok(m) and exact_age(m),
                 lambda m: sex_ok(m) and age_ok(m),
                 lambda m: age_ok(m),
                 sex_ok]
        for t in tests:
            cand = [k for k in pool if t(skin_meta(k))]
            if cand:
                pool = cand
                break
    ident["skin"] = rd.choice(sorted(pool))
    # tone variation within the texture's range (applied to the skin albedo in Blender)
    ident["skin_tone"] = round(rd.uniform(0.8, 1.2), 4)
    ident["demographics"] = {"age_years": None if years is None else round(years, 1), "sex": sex, "ancestry": anc,
                             "skin_match": skin_meta(ident["skin"])}


# ----------------------------------------------------------------------------- lights & world
_FAMILY_RANGES = {
    # family: (distance m, size range [w,h] m, power range)  -- power in W (area/point) or strength
    "rect_softbox": ((0.7, 3.0), ((0.25, 1.4), (0.25, 1.4)), (40.0, 400.0)),
    "octa_softbox": ((0.8, 3.0), ((0.4, 1.5), None), (1.5, 25.0)),
    "strip": ((0.7, 2.5), ((0.08, 0.2), (0.6, 1.6)), (30.0, 250.0)),
    "window": ((1.5, 4.0), ((0.8, 2.2), (0.8, 2.0)), (1.0, 12.0)),
    "grid_panel": ((0.6, 2.0), ((0.2, 0.6), (0.2, 0.6)), (3.0, 40.0)),
    "bulb": ((0.5, 2.5), ((0.005, 0.04), None), (2.0, 60.0)),
    "colored_led": ((0.4, 2.0), ((0.003, 0.02), None), (1.0, 25.0)),
    "sun": ((10.0, 10.0), ((0.4, 3.0), None), (2.0, 12.0)),
    "flame": ((0.4, 1.5), ((0.05, 0.15), (0.12, 0.35)), (8.0, 80.0)),
    "emissive_card": ((0.8, 3.0), ((0.2, 0.9), (0.15, 0.7)), (0.5, 6.0)),
    "ring_light": ((0.4, 1.5), ((0.25, 0.6), None), (4.0, 40.0)),
}


def sample_light(r: random.Random, family: str, key: bool) -> dict:
    (d0, d1), sizes, (p0, p1) = _FAMILY_RANGES[family]
    w = r.uniform(*sizes[0])
    h = r.uniform(*sizes[1]) if sizes[1] else w
    az = r.uniform(-65, 65) if key else r.uniform(-90, 90)
    el = r.uniform(-10, 55) if key else r.uniform(-25, 70)
    power = math.exp(r.uniform(math.log(p0), math.log(p1)))
    if family == "colored_led":
        hue = r.random()
        col = [max(0.0, min(1.0, abs(math.sin(math.pi * (hue + k / 3.0))) ** 2)) for k in range(3)]
        m = max(col) or 1.0
        col = [c / m for c in col]
    elif family == "flame":
        col = [1.0, 0.62, 0.3]
    elif r.random() < 0.15:
        col = [round(r.uniform(0.3, 1.0), 3) for _ in range(3)]
        m = max(col)
        col = [c / m for c in col]
    else:
        col = _cct_rgb(r.uniform(2700, 7000))
    spec = {"family": family, "azimuth_deg": round(az, 3), "elevation_deg": round(el, 3), "distance_m": round(r.uniform(d0, d1), 4),
            "size": [round(w, 4), round(h, 4)], "power": round(power, 4), "color": col, "roll": round(r.uniform(-0.5, 0.5), 4),
            "seed": r.randint(0, 10_000)}
    if family == "sun":
        spec["angle_deg"] = spec["size"][0]
    if family == "window":
        spec.update(cols=r.randint(2, 5), rows=r.randint(1, 4), mullion=round(r.uniform(0.03, 0.12), 3))
    if family == "grid_panel":
        spec["dots"] = r.randint(6, 20)
    if family == "rect_softbox":
        spec["spread_deg"] = round(r.uniform(60, 180), 2)
    return spec


def sample_key_softbox(r: random.Random) -> dict:
    """v2: the interview/portrait key: a large rectangular (or octa) softbox in front of and above
    the subject, the dominant light on the face, reflected as a big bright catchlight in the upper
    iris/pupil (often near the upper lid).  Power scales with distance^2 so the face irradiance,
    not the distance, sets its dominance."""
    spec = sample_light(r, "rect_softbox", key=True)  # area light (watts): power/(pi d^2) = face irradiance
    w = r.uniform(0.4, 1.2)
    a = r.uniform(1.0, 1.6)  # square to 1:1.6
    spec["size"] = [round(w, 4), round(w * a, 4)] if r.random() < 0.6 else [round(w * a, 4), round(w, 4)]
    d = r.uniform(0.8, 2.5)
    spec.update(azimuth_deg=round(r.uniform(-35, 35), 3), elevation_deg=round(r.uniform(5, 38), 3), distance_m=round(d, 4),
                roll=round(r.uniform(-0.15, 0.15), 4), key_softbox=True, spread_deg=round(r.uniform(90, 180), 2))
    # the top of the family's power range at 1 m, scaled by d^2 so distance changes the catchlight's
    # size and relative brightness (solid angle) but not the face irradiance / dominance
    spec["power"] = round(_FAMILY_RANGES["rect_softbox"][2][1] * r.uniform(0.35, 1.0) * d * d, 4)
    return spec


def sample_lights(r: random.Random, split: str, force_heldout: bool = False, catch: dict | None = None) -> list[dict]:
    fams, weights = zip(*TRAIN_LIGHTS.items())
    n = r.choices([1, 2, 3, 4], weights=[3, 4, 2, 1])[0]
    lights = []
    key_box = bool(catch) and not force_heldout and r.random() < catch["key_softbox_prob"]
    for k in range(n):
        if k == 0 and key_box:
            lights.append(sample_key_softbox(r))
            continue
        if k == 0 and (force_heldout or (split == "test" and r.random() < 0.3)):
            fam = HELDOUT_LIGHT
        else:
            fam = r.choices(fams, weights=weights)[0]
        L = sample_light(r, fam, key=(k == 0))
        if key_box:  # fills: dimmer than the key
            L["power"] = round(L["power"] * r.uniform(0.1, 0.5), 4)
        lights.append(L)
    return lights


def sample_world(r: random.Random) -> dict:
    return {"seed": r.randint(0, 100000), "strength": round(math.exp(r.uniform(math.log(0.03), math.log(1.0))), 4),
            "panel_density": round(r.uniform(0.04, 0.35), 3), "panel_gain": round(r.uniform(1.0, 9.0), 3),
            "panel_scale": round(r.uniform(1.5, 6.0), 3), "colorful": round(r.uniform(0.0, 0.9), 3),
            "dark_blobs": round(r.uniform(0.03, 0.3), 3), "noise_scale": round(r.uniform(1.0, 5.0), 3),
            "blob_scale": round(r.uniform(0.8, 3.0), 3),
            "tint": [round(r.uniform(0.8, 1.0), 3) for _ in range(3)],
            "floor": [round(r.uniform(0.01, 0.12), 3)] * 3, "ceiling": [round(r.uniform(0.1, 0.5), 3)] * 3}


def sample_silhouettes(r: random.Random) -> list[dict]:
    out = []
    if r.random() < 0.4:
        for _ in range(r.randint(1, 2)):
            out.append({"azimuth_deg": r.uniform(-25, 25), "elevation_deg": r.uniform(-10, 15), "distance_m": r.uniform(0.6, 2.5),
                        "size": [r.uniform(0.15, 0.5), r.uniform(0.3, 1.1)], "color": [r.uniform(0.005, 0.05)] * 3})
    return out


# ----------------------------------------------------------------------------- lids / blinks
def still_openness(r: random.Random, relaxed_prob: float = 0.0) -> float:
    """Lid aperture for stills: open 70%, partly closed 15%, nearly closed 8%, closed 7%.
    v2 relaxed_prob: first, with this probability, a relaxed upper lid resting over the top of
    the iris (as in typical interview footage)."""
    if relaxed_prob > 0 and r.random() < relaxed_prob:
        return round(r.uniform(0.55, 0.82), 3)
    u = r.random()
    if u < 0.70:
        return round(r.uniform(0.85, 1.0), 3)
    if u < 0.85:
        return round(r.uniform(0.3, 0.85), 3)
    if u < 0.93:
        return round(r.uniform(0.05, 0.3), 3)
    return 0.0


def blink_curve(n: int, start: int, depth: float, r: random.Random) -> list[float]:
    """Natural blink: fast close (3 frames), short hold, slower reopen (5-7 frames) at 24 fps."""
    o = [1.0] * n
    close = [0.62, 0.25, 0.0]
    hold = r.randint(0, 2)
    reopen_n = r.randint(5, 7)
    seq = [1 - (1 - c) * depth for c in close] + [1 - depth] * hold
    seq += [1 - depth * (1 - (1 - math.cos(math.pi * (k + 1) / reopen_n)) / 2) for k in range(reopen_n)]
    for k, v in enumerate(seq):
        if 0 <= start + k < n:
            o[start + k] = round(max(0.0, min(1.0, v)), 4)
    return o


def gaze_track(n: int, r: random.Random, yaw_range=25.0, pitch_range=15.0) -> list[list[float]]:
    """Fixations with saccades (2-3 frames) and slight drift."""
    g = [r.uniform(-yaw_range, yaw_range) * 0.6, r.uniform(-pitch_range, pitch_range) * 0.6]
    out = []
    target = list(g)
    t_next = r.randint(4, 12)
    for k in range(n):
        if k == t_next:
            target = [r.uniform(-yaw_range, yaw_range), r.uniform(-pitch_range, pitch_range)]
            t_next = k + r.randint(5, 14)
        g = [g[0] + (target[0] - g[0]) * 0.55 + r.gauss(0, 0.15), g[1] + (target[1] - g[1]) * 0.55 + r.gauss(0, 0.1)]
        out.append([round(g[0], 3), round(g[1], 3)])
    return out


# ----------------------------------------------------------------------------- shots
def _dim_ambient_for_key(shot: dict, r: random.Random) -> dict:
    """v2: with a dominant key softbox the procedural ambient is dimmed (draw only in that case,
    so v1 plans are unchanged)."""
    if any(L.get("key_softbox") for L in shot["lights"]):
        shot["world"]["strength"] = round(shot["world"]["strength"] * r.uniform(0.15, 0.5), 4)
    return shot


def _camera_still(r: random.Random, kind: str) -> dict:
    if kind == "two_eyes":
        return {"lens_mm": round(r.uniform(50, 135), 2), "target": "eyes_mid", "iris_px": round(r.uniform(34, 72), 2),
                "azimuth_deg": round(r.uniform(-35, 35), 3), "elevation_deg": round(r.uniform(-18, 22), 3),
                "roll_deg": round(r.uniform(-10, 10), 3), "offset_right_m": round(r.uniform(-0.006, 0.006), 5),
                "offset_up_m": round(r.uniform(-0.005, 0.005), 5)}
    side = r.choice(["eye_L", "eye_R"])
    return {"lens_mm": round(r.uniform(60, 180), 2), "target": side, "iris_px": round(r.uniform(58, 175), 2),
            "azimuth_deg": round(r.uniform(-38, 38), 3), "elevation_deg": round(r.uniform(-20, 25), 3),
            "roll_deg": round(r.uniform(-12, 12), 3), "offset_right_m": round(r.uniform(-0.004, 0.004), 5),
            "offset_up_m": round(r.uniform(-0.003, 0.003), 5)}


def _camera_portrait(r: random.Random, kind: str) -> dict:
    """Near-axis camera for the deployment case: subject facing the lens.

    kind selects how close we are.  ``ultra_closeup`` goes well past the ordinary close-up range -
    lashes stay in frame, but the iris fills it.  ``one_eye`` frames a single eye, left or right
    interchangeably, which is how a lot of delivered footage is cropped.
    """
    if kind == "two_eyes":
        iris_px, target = r.uniform(34, 80), "eyes_mid"
    elif kind == "one_eye":
        iris_px, target = r.uniform(70, 150), r.choice(["eye_L", "eye_R"])
    else:  # ultra_closeup
        iris_px, target = r.uniform(150, 300), r.choice(["eye_L", "eye_R"])
    lens = r.uniform(50, 135) if kind == "two_eyes" else r.uniform(85, 200)
    return {"lens_mm": round(lens, 2), "target": target, "iris_px": round(iris_px, 2),
            "azimuth_deg": round(r.uniform(-10, 10), 3), "elevation_deg": round(r.uniform(-8, 8), 3),
            "roll_deg": round(r.uniform(-6, 6), 3), "offset_right_m": round(r.uniform(-0.004, 0.004), 5),
            "offset_up_m": round(r.uniform(-0.003, 0.003), 5)}


def sample_portrait_key(r: random.Random) -> dict:
    """A clear, near-frontal key whose catchlight lands ON the iris/pupil - but any light family, at
    any position around it.  The prototype's key was always a softbox above the subject, so every
    bright catchlight sat in the same place; here the family is drawn from the full training set and
    the azimuth/elevation sweep the catchlight around the iris.
    """
    fams, weights = zip(*TRAIN_LIGHTS.items())
    fam = r.choices(fams, weights=weights)[0]
    spec = sample_light(r, fam, key=True)
    # sweep the catchlight around the iris instead of parking it above: on a near-frontal camera the
    # highlight's position on the cornea follows the light's angular position
    spec["azimuth_deg"] = round(r.uniform(-48, 48), 3)
    spec["elevation_deg"] = round(r.uniform(-28, 52), 3)
    spec["roll"] = round(r.uniform(-0.6, 0.6), 4)          # a tilted softbox is not a level rectangle
    # distance is sampled log-uniformly over a wide span and the power tracks d^2, so face
    # irradiance stays comparable while the source's ANGULAR size - and therefore the catchlight's
    # size and edge softness - varies from a hard point to a wall-sized wrap
    d = math.exp(r.uniform(math.log(0.45), math.log(6.0)))
    spec["distance_m"] = round(d, 4)
    if fam in ("rect_softbox", "octa_softbox", "strip", "window", "grid_panel", "emissive_card"):
        w = math.exp(r.uniform(math.log(0.12), math.log(2.4)))
        a = r.uniform(1.0, 2.6)
        spec["size"] = [round(w, 4), round(w * a, 4)] if r.random() < 0.5 else [round(w * a, 4), round(w, 4)]
    lo, hi = _FAMILY_RANGES[fam][2]
    spec["power"] = round(math.exp(r.uniform(math.log(max(lo, hi * 0.35)), math.log(hi))) * (d ** 2), 4)
    spec["portrait_key"] = True
    return spec


def plan_portrait_still(ident: dict, split: str, idx: int, base_seed: int, render: dict,
                        catch: dict | None = None, wide_open: bool = False) -> dict:
    """Deployment-shaped still: eyes into the lens, one clear catchlight on the iris.

    Covers portrait photography, a reporter talking to camera and green-screen performance - the
    cases this model is actually for.  Only the framing and the key light are constrained; iris
    colour, resolution, lens, exposure, noise, codec and blur keep their full training variety, and
    the ordinary `plan_still` population still supplies the off-axis and awkward cases.

    wide_open (Round 2, config ``data.portrait_open_pairs``): lids fully open (0.92-1.0) instead of
    the ordinary openness law, whose ``relaxed_lid_prob`` puts 40% of shots in a 0.55-0.82 band
    where the upper lid rests over the iris and casts a heavy shadow across the eye.  These shots
    use their own seed namespace and index space, so adding them re-seeds nothing already rendered.
    """
    s = stable_seed(base_seed, "portrait_open" if wide_open else "portrait", split, idx)
    r = random.Random(s)
    kind = r.choices(["two_eyes", "one_eye", "ultra_closeup"], weights=[3, 4, 3])[0]
    if wide_open:
        op = round(r.uniform(0.92, 1.0), 3)
        op_r = op if r.random() < 0.85 else round(r.uniform(0.92, 1.0), 3)
    else:
        rp = catch["relaxed_lid_prob"] if catch else 0.0
        op = still_openness(r, rp)
        op_r = op if r.random() < 0.85 else still_openness(r, rp)
    g = [round(r.uniform(-6, 6), 3), round(r.uniform(-5, 4), 3)]   # into the lens, small natural drift
    verg = r.uniform(0, 1.5)
    frame = {"frame": 1, "time_s": 0.0, "gaze": {"L": [g[0] - verg, g[1]], "R": [g[0] + verg, g[1]]},
             "openness": {"L": op, "R": op_r}, "pupil": {"L": round(r.random(), 3), "R": None},
             "head": [round(r.uniform(-5, 5), 3), round(r.uniform(-4, 4), 3), round(r.uniform(-3, 3), 3)],
             "env_phase": round(r.uniform(0, 10), 3), "flame_phase": round(r.uniform(0, 10), 3)}
    frame["pupil"]["R"] = round(min(1.0, max(0.0, frame["pupil"]["L"] + r.uniform(-0.05, 0.05))), 3)
    lights = [sample_portrait_key(r)]
    for _ in range(r.choices([0, 1, 2], weights=[3, 4, 2])[0]):      # fill / rim, never dominant
        f = r.choices(*zip(*[(k, v) for k, v in TRAIN_LIGHTS.items()]))[0]
        lights.append(sample_light(r, f, key=False))
    world = sample_world(r)
    tag = "portraitopen" if wide_open else "portrait"
    shot = {"shot_id": f"{SPLIT_TAG[split]}_{tag}_{idx:05d}", "identity_id": ident["identity_id"], "split": split,
            "kind": "still", "framing": kind, "seed": s, "width": render["width"], "height": render["height"],
            "samples": render["samples"], "adaptive": render["adaptive_sampling"], "label_samples": 16,
            "camera": _camera_portrait(r, kind), "lights": lights, "silhouettes": sample_silhouettes(r),
            "world": world, "negative": stable_seed(s, "neg") % 1000 < 250, "frames": [frame],
            "portrait": True}
    if wide_open:
        shot["wide_open"] = True
    return shot


def plan_still(ident: dict, split: str, idx: int, base_seed: int, render: dict, catch: dict | None = None) -> dict:
    s = stable_seed(base_seed, "still", split, idx)
    r = random.Random(s)
    kind = "two_eyes" if r.random() < 0.4 else "eye_closeup"
    rp = catch["relaxed_lid_prob"] if catch else 0.0
    op = still_openness(r, rp)
    op_r = op if r.random() < 0.8 else still_openness(r, rp)
    g = [round(r.uniform(-25, 25), 3), round(r.uniform(-16, 14), 3)]
    verg = r.uniform(0, 3)
    frame = {"frame": 1, "time_s": 0.0, "gaze": {"L": [g[0] - verg, g[1]], "R": [g[0] + verg, g[1]]},
             "openness": {"L": op, "R": op_r}, "pupil": {"L": round(r.random(), 3), "R": None},
             "head": [round(r.uniform(-8, 8), 3), round(r.uniform(-6, 6), 3), round(r.uniform(-4, 4), 3)],
             "env_phase": round(r.uniform(0, 10), 3), "flame_phase": round(r.uniform(0, 10), 3)}
    frame["pupil"]["R"] = round(min(1.0, max(0.0, frame["pupil"]["L"] + r.uniform(-0.05, 0.05))), 3)
    shot = {"shot_id": f"{SPLIT_TAG[split]}_still_{idx:05d}", "identity_id": ident["identity_id"], "split": split, "kind": "still",
            "framing": kind, "seed": s, "width": render["width"], "height": render["height"], "samples": render["samples"],
            "adaptive": render["adaptive_sampling"], "label_samples": 16, "camera": _camera_still(r, kind),
            "lights": sample_lights(r, split, catch=catch), "silhouettes": sample_silhouettes(r), "world": sample_world(r),
            "negative": stable_seed(s, "neg") % 1000 < 250, "frames": [frame]}
    return _dim_ambient_for_key(shot, r)


def plan_clip(ident: dict, split: str, idx: int, base_seed: int, render: dict, n_frames: int, fps: float = 24.0,
              catch: dict | None = None) -> dict:
    s = stable_seed(base_seed, "clip", split, idx)
    r = random.Random(s)
    kind = "two_eyes" if r.random() < 0.4 else "eye_closeup"
    cam = _camera_still(r, kind)
    gaze = gaze_track(n_frames, r)
    blink = r.random() < 0.45
    depth = 1.0 if r.random() < 0.7 else r.uniform(0.4, 0.8)  # full or half blink
    start = r.randint(0, max(0, n_frames - 6))
    open_L = blink_curve(n_frames, start, depth, r) if blink else [1.0] * n_frames
    if catch and r.random() < catch["relaxed_lid_prob"]:
        base_open = r.uniform(0.55, 0.82)
    else:
        base_open = r.uniform(0.85, 1.0) if r.random() < 0.85 else r.uniform(0.35, 0.85)
    open_L = [round(o * base_open, 4) for o in open_L]
    head_motion = r.random() < 0.5
    cam_motion = r.random() < 0.4
    light_motion = r.random() < 0.5
    env_anim = r.random() < 0.3
    hy0, hp0 = r.uniform(-6, 6), r.uniform(-5, 5)
    dhy, dhp = (r.uniform(-8, 8), r.uniform(-5, 5)) if head_motion else (0.0, 0.0)
    daz, ddist = (r.uniform(-6, 6), r.uniform(-0.1, 0.1)) if cam_motion else (0.0, 0.0)
    lights = sample_lights(r, split, catch=catch)
    lm = {str(k): {"d_az": r.uniform(-12, 12), "d_el": r.uniform(-6, 6)} for k in range(len(lights)) if light_motion and r.random() < 0.7}
    pupil0 = r.random()
    dp = r.uniform(-0.25, 0.25)
    frames = []
    for k in range(n_frames):
        t = k / max(1, n_frames - 1)
        fr = {"frame": k + 1, "time_s": round(k / fps, 6),
              "gaze": {"L": gaze[k], "R": [gaze[k][0] + 1.5, gaze[k][1]]},
              "openness": {"L": open_L[k], "R": open_L[k]},
              "pupil": {"L": round(min(1, max(0, pupil0 + dp * t)), 4), "R": round(min(1, max(0, pupil0 + dp * t)), 4)},
              "head": [round(hy0 + dhy * t, 4), round(hp0 + dhp * t, 4), 0.0],
              "flame_phase": round(k * 0.35, 4)}
        if env_anim:
            fr["env_phase"] = round(k * 0.08, 4)
        if cam_motion and k > 0:
            fr["camera"] = {"azimuth_deg": cam["azimuth_deg"] + daz * t}
            if "iris_px" in cam:
                fr["camera"]["iris_px"] = cam["iris_px"] * (1 + ddist * t)
        if lm:
            fr["light_motion"] = {kk: {"d_az": v["d_az"] * t, "d_el": v["d_el"] * t} for kk, v in lm.items()}
        frames.append(fr)
    shot = {"shot_id": f"{SPLIT_TAG[split]}_clip_{idx:05d}", "identity_id": ident["identity_id"], "split": split, "kind": "clip",
            "framing": kind, "seed": s, "width": render["width"], "height": render["height"], "samples": render["samples"],
            "adaptive": render["adaptive_sampling"], "label_samples": 16, "camera": cam, "lights": lights,
            "silhouettes": sample_silhouettes(r), "world": sample_world(r), "fps": fps,
            "negative": stable_seed(s, "neg") % 1000 < 200, "has_blink": blink, "blink_depth": depth if blink else 0.0,
            "head_motion": head_motion, "camera_motion": cam_motion, "light_motion": bool(lm), "frames": frames}
    return _dim_ambient_for_key(shot, r)


# ----------------------------------------------------------------------------- demo cases
DEMO_CASES = [
    ("small_moving_softbox", {}), ("saturated_pupil_crossing_no_donor", {}), ("colored_multiple_lights", {}),
    ("animated_flame_reflection", {}), ("dark_brown_iris", {"color": "dark_brown"}),
    ("blue_green_irregular_iris", {"color": "blue", "family": "crypt_rich"}), ("blink_eyelash_occlusion", {}),
    ("small_eye_wide_shot", {}), ("no_reflection_negative", {}), ("non_eye_bright_highlights", {}),
    ("head_motion", {}), ("shot_cut_identity_change", {}),
]


def plan_demo(case: str, ident: dict, ident_b: dict | None, idx: int, base_seed: int, render: dict, data: dict, fps: float = 24.0) -> list[dict]:
    s = stable_seed(base_seed, "demo", case)
    r = random.Random(s)
    n = data["demo_frames_per_case"]
    W, H = data["demo_width"], data["demo_height"]
    face_frac = r.uniform(0.45, 0.7)
    cam = {"lens_mm": round(r.uniform(70, 110), 2), "target": "face", "face_frac": face_frac, "azimuth_deg": round(r.uniform(-15, 15), 2),
           "elevation_deg": round(r.uniform(-5, 8), 2), "roll_deg": 0.0}
    key = sample_light(r, "rect_softbox", True)
    key.update(azimuth_deg=r.uniform(-25, 25), elevation_deg=r.uniform(10, 35), distance_m=r.uniform(1.0, 1.8), size=[0.5, 0.7], power=140.0)
    lights = [key]
    world = sample_world(r)
    world["strength"] = r.uniform(0.08, 0.3)
    gaze = gaze_track(n, r, 12, 8)
    openness = [1.0] * n
    head = [[0.0, 0.0, 0.0] for _ in range(n)]
    negative = False
    light_motion = {}
    extra = {}
    if case == "small_moving_softbox":
        lights = [sample_light(r, "rect_softbox", True)]
        lights[0].update(size=[0.15, 0.2], power=40.0, distance_m=0.9, azimuth_deg=-20.0, elevation_deg=15.0)
        light_motion = {"0": {"d_az": 40.0, "d_el": 8.0}}
        lights.append(dict(sample_light(r, "window", False), power=1.5))
    elif case == "saturated_pupil_crossing_no_donor":
        lights = [sample_light(r, "bulb", True)]
        lights[0].update(size=[0.03, 0.03], power=120.0, distance_m=0.8, azimuth_deg=4.0, elevation_deg=3.0)
        lights.append(dict(key, power=60.0))
        gaze = [[2.0, 1.0]] * n  # static gaze and light: the covered iris region is never revealed
        extra["saturate"] = True
    elif case == "colored_multiple_lights":
        lights = [dict(sample_light(r, "colored_led", True), azimuth_deg=a, elevation_deg=e, power=18.0)
                  for a, e in ((-30, 10), (-5, 25), (25, 5))]
        lights[0]["color"], lights[1]["color"], lights[2]["color"] = [1, 0.1, 0.1], [0.1, 1, 0.2], [0.2, 0.3, 1.0]
        lights.append(dict(key, power=70.0))
    elif case == "animated_flame_reflection":
        lights = [dict(sample_light(r, "flame", True), azimuth_deg=10.0, elevation_deg=-5.0, distance_m=0.6, power=60.0)]
        lights.append(dict(key, power=30.0))
    elif case == "blink_eyelash_occlusion":
        openness = blink_curve(n, n // 3, 1.0, r)
        gaze = [[g[0], g[1] - 8.0] for g in gaze]
    elif case == "small_eye_wide_shot":
        cam["face_frac"] = 0.12
        cam["lens_mm"] = 50.0
    elif case == "no_reflection_negative":
        negative = True
    elif case == "non_eye_bright_highlights":
        lights += [dict(sample_light(r, "emissive_card", False), azimuth_deg=a, elevation_deg=e, power=8.0, camera_visible=True)
                   for a, e in ((170, 5), (160, -10))]
        world.update(panel_gain=9.0, panel_density=0.35, strength=0.6)
    elif case == "head_motion":
        head = [[round(-15 + 30 * k / max(1, n - 1), 3), round(6 * math.sin(k / n * math.pi), 3), 0.0] for k in range(n)]
    frames = []
    for k in range(n):
        fr = {"frame": k + 1, "time_s": round(k / fps, 6), "gaze": {"L": gaze[k], "R": [gaze[k][0] + 1.5, gaze[k][1]]},
              "openness": {"L": openness[k], "R": openness[k]}, "pupil": {"L": 0.45, "R": 0.45}, "head": head[k],
              "flame_phase": round(k * 0.35, 4)}
        if light_motion:
            fr["light_motion"] = {kk: {"d_az": v["d_az"] * k / max(1, n - 1), "d_el": v["d_el"] * k / max(1, n - 1)} for kk, v in light_motion.items()}
        frames.append(fr)
    shot = {"shot_id": f"demo_{idx:02d}_{case}", "identity_id": ident["identity_id"], "split": "test", "kind": "demo", "case": case,
            "seed": s, "width": W, "height": H, "samples": render["samples"], "adaptive": render["adaptive_sampling"],
            "label_samples": 16, "camera": cam, "lights": lights, "silhouettes": [], "world": world, "fps": fps,
            "negative": negative, "frames": frames, "save_blend": idx in (0, 1, 7, 11), "extra": extra}
    if case != "shot_cut_identity_change" or ident_b is None:
        return [shot]
    half = n // 2
    a = dict(shot, frames=frames[:half], shot_id=f"demo_{idx:02d}_{case}_a", cut_part=0)
    b_frames = [dict(f, frame=f["frame"], time_s=f["time_s"]) for f in frames[half:]]
    cam_b = dict(cam, azimuth_deg=-cam["azimuth_deg"] + r.uniform(-5, 5), face_frac=r.uniform(0.4, 0.65))
    b = dict(shot, frames=b_frames, shot_id=f"demo_{idx:02d}_{case}_b", identity_id=ident_b["identity_id"], camera=cam_b,
             lights=[dict(key, azimuth_deg=-key["azimuth_deg"], power=120.0)], world=sample_world(r), cut_part=1, seed=stable_seed(s, "b"))
    return [a, b]


# ----------------------------------------------------------------------------- whole plan
@dataclass
class Plan:
    identities: dict
    shots: list
    config_profile: str

    def to_json(self) -> dict:
        return {"schema": "eyeclean.dataset_plan/1", "profile": self.config_profile,
                "identities": self.identities, "shots": self.shots}


def plan_dataset(cfg: dict) -> Plan:
    seed = cfg["seed"]
    data = cfg["data"]
    render = cfg["render"]
    skins = partition_skins(seed)
    identities: dict[str, dict] = {}
    by_split: dict[str, list[dict]] = {}
    for split in SPLITS:
        lst = []
        for i in range(data["identity_counts"][split]):
            ident = sample_identity(split, i, seed, skins[split], demographics=data.get("demographics"))
            identities[ident["identity_id"]] = ident
            lst.append(ident)
        by_split[split] = lst
    shots = []
    for split in SPLITS:
        ids = by_split[split]
        r = random.Random(stable_seed(seed, "assign", split))
        for i in range(data["still_pairs"][split]):
            ident = ids[i % len(ids)] if i < len(ids) else r.choice(ids)
            shots.append(plan_still(ident, split, i, seed, render, catch=data.get("catchlights")))
        for i in range(data["clips"][split]):
            ident = ids[(i * 2 + 1) % len(ids)]
            shots.append(plan_clip(ident, split, i, seed, render, data["frames_per_clip"], catch=data.get("catchlights")))
        # deployment-shaped stills, appended with their own index space and RNG so adding or growing
        # them never re-seeds an already-rendered shot
        npor = (data.get("portrait_pairs") or {}).get(split, 0)
        if npor:
            rp_ = random.Random(stable_seed(seed, "assign_portrait", split))
            for i in range(npor):
                ident = ids[i % len(ids)] if i < len(ids) else rp_.choice(ids)
                shots.append(plan_portrait_still(ident, split, i, seed, render, catch=data.get("catchlights")))
        # fully open lids: the ordinary openness law leaves 40% of shots with the upper lid resting
        # over the iris (relaxed_lid_prob) and shadowing the eye.  Own index space and RNG again.
        nopen = (data.get("portrait_open_pairs") or {}).get(split, 0)
        if nopen:
            ro_ = random.Random(stable_seed(seed, "assign_portrait_open", split))
            for i in range(nopen):
                ident = ids[i % len(ids)] if i < len(ids) else ro_.choice(ids)
                shots.append(plan_portrait_still(ident, split, i, seed, render, catch=data.get("catchlights"),
                                                 wide_open=True))
    # demo identities: dedicated test-split identities (disjoint from train/validation)
    ncase = data["full_head_demo_cases"]
    base = data["identity_counts"]["test"]
    cases = DEMO_CASES[:ncase] if ncase <= len(DEMO_CASES) else DEMO_CASES + DEMO_CASES[: ncase - len(DEMO_CASES)]
    for k, (case, opts) in enumerate(cases):
        ident = sample_identity("test", base + k, seed, skins["test"], opts.get("family"), opts.get("color"))
        identities[ident["identity_id"]] = ident
        ident_b = None
        if case == "shot_cut_identity_change":
            ident_b = sample_identity("test", base + ncase + k, seed, skins["test"])
            identities[ident_b["identity_id"]] = ident_b
        shots.extend(plan_demo(case, ident, ident_b, k, seed, render, data))
    if (data.get("catchlights") or {}).get("clean_iris_render"):
        for sh in shots:  # every split and demo case renders the clean layer the same way
            sh["clean_iris_render"] = True
    return Plan(identities, shots, cfg["profile"])


def plan_digest(plan: Plan) -> str:
    return hashlib.sha256(json.dumps(plan.to_json(), sort_keys=True).encode()).hexdigest()


def split_integrity(plan: Plan) -> dict:
    """Verify identities, texture seeds, skins, scene seeds and clips are split-disjoint."""
    by = {s: {"ids": set(), "iris_seeds": set(), "skins": set(), "scene_seeds": set(), "families": set(), "lights": set()} for s in SPLITS}
    for iid, ident in plan.identities.items():
        b = by[ident["split"]]
        b["ids"].add(iid)
        b["iris_seeds"] |= {ident["iris"]["L"]["seed"], ident["iris"]["R"]["seed"]}
        b["skins"].add(ident["skin"])
        b["families"].add(ident["iris_family"])
    for sh in plan.shots:
        b = by[sh["split"]]
        b["scene_seeds"].add(sh["seed"])
        for L in sh["lights"]:
            b["lights"].add(L["family"])
        if plan.identities[sh["identity_id"]]["split"] != sh["split"]:
            return {"passed": False, "reason": f"shot {sh['shot_id']} uses identity from another split"}
    problems = []
    for key in ("ids", "iris_seeds", "skins", "scene_seeds"):
        for a in SPLITS:
            for b2 in SPLITS:
                if a < b2 and by[a][key] & by[b2][key]:
                    problems.append(f"{key} shared between {a} and {b2}: {sorted(by[a][key] & by[b2][key])[:5]}")
    for s in ("train", "validation"):
        if HELDOUT_STRUCTURE in by[s]["families"]:
            problems.append(f"held-out iris family in {s}")
        if HELDOUT_LIGHT in by[s]["lights"]:
            problems.append(f"held-out light family in {s}")
    return {"passed": not problems, "problems": problems,
            "counts": {s: {k: len(v) for k, v in by[s].items()} for s in SPLITS},
            "heldout_iris_in_test": HELDOUT_STRUCTURE in by["test"]["families"],
            "heldout_light_in_test": HELDOUT_LIGHT in by["test"]["lights"]}

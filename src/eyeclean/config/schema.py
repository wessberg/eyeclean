"""The single public EyeClean configuration schema.

Every key in ``configs/*.yaml`` is declared here with its type and allowed values.  Unknown or
misspelled keys are rejected with the dotted path and a suggestion; they are never ignored.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from typing import Any


class ConfigError(ValueError):
    """Raised for invalid configuration files (exit code 2 at the CLI)."""


@dataclass(frozen=True)
class F:
    """Field specification."""

    kind: str  # 'int' | 'float' | 'bool' | 'str' | 'map' | 'list_int' | 'split_counts' | 'stage'
    choices: tuple[Any, ...] | None = None
    min: float | None = None
    max: float | None = None
    children: dict[str, "F"] | None = None
    required: bool = True
    const: Any = None  # value required by the contract (e.g. cpu_image_fallback must be false)


def _split(min_: int = 0, required: bool = True) -> F:
    return F("map", required=required,
             children={"train": F("int", min=min_), "validation": F("int", min=min_), "test": F("int", min=min_)})


def _stage() -> F:
    return F("map", children={"steps": F("int", min=1), "microbatch": F("int", min=1), "effective_batch": F("int", min=1)})


SCHEMA: dict[str, F] = {
    "schema_version": F("str", choices=("1.0",)),
    "profile": F("str", choices=("smoke", "prototype", "quality", "custom")),
    "run_dir": F("str"),
    "seed": F("int", min=0),
    "gpu": F("map", children={
        "policy": F("str", choices=("strict",)),
        "device_index": F("int", min=0),
        "target_working_gib": F("float", min=1.0),
        "reserve_free_gib": F("float", min=0.0),
        "cpu_image_fallback": F("bool", const=False),
        "sequential_render_train": F("bool"),
        "network_precision": F("str", choices=("bf16", "fp32")),
        "reference_precision": F("str", choices=("float32",)),
        "compile": F("str", choices=("auto_cuda_only", "disabled")),
    }),
    "resources": F("map", children={
        "workspace_cap_gib": F("float", min=0.1),
        "min_free_disk_gib": F("float", min=0.0),
        "max_repair_attempts": F("int", min=0, max=2),
        "max_total_optimizer_steps": F("int", min=1),
        "allow_paid_services": F("bool", const=False),
        "allow_uploads": F("bool", const=False),
    }),
    "render": F("map", children={
        "engine": F("str", choices=("CYCLES",)),
        "device": F("str", choices=("OPTIX",)),
        "width": F("int", min=64),
        "height": F("int", min=64),
        "samples": F("int", min=1),
        "adaptive_sampling": F("bool"),
        "denoising": F("bool"),
        "training_cache_dtype": F("str", choices=("float16", "float32")),
        "diagnostic_dtype": F("str", choices=("float32",)),
        "paired_target": F("str", choices=("camera_ocular_reflection_closure_off",)),
        "preserve_refraction": F("bool", const=True),
        "preserve_transmission_weight": F("bool", const=True),
        "save_representative_blend_scenes": F("bool"),
        "external_assets": F("str", choices=("verified_public_ungated_only",)),
    }),
    "data": F("map", children={
        "identity_counts": _split(1),
        "still_pairs": _split(1),
        # deployment-shaped stills (eyes into the lens, clear catchlight on the iris); optional so
        # older configs stay valid, and a separate index space so adding them re-renders nothing
        "portrait_pairs": _split(0, required=False),
        # deployment-shaped stills with fully open lids (0.92-1.0), i.e. no relaxed-lid shadow over
        # the iris; optional, and again a separate index space so adding them re-renders nothing
        "portrait_open_pairs": _split(0, required=False),
        "clips": _split(1),
        "frames_per_clip": F("int", min=5),
        "full_head_demo_cases": F("int", min=1),
        "demo_frames_per_case": F("int", min=5),
        "demo_width": F("int", min=160),
        "demo_height": F("int", min=120),
        "negative_fraction_min": F("float", min=0.0, max=1.0),
        "saturated_fraction_min": F("float", min=0.0, max=1.0),
        "disjoint_identities_textures_scenes": F("bool", const=True),
        "heldout_light_family": F("bool"),
        "heldout_material_family": F("bool"),
        "augmentation_device": F("str", choices=("cuda",)),
        "same_pair_noise_realization": F("bool"),
        # optional v2 data options (absent = the original prototype behaviour, bit-for-bit)
        "camera_sim": F("map", required=False, children={
            "exposure_midgray_min": F("float", min=0.005, max=1.0),  # crop median as a fraction of the white level
            "exposure_midgray_max": F("float", min=0.005, max=1.0),
            "highlight_rolloff_prob": F("float", min=0.0, max=1.0),  # camera knee instead of a hard clip
            "codec_prob": F("float", min=0.0, max=1.0),  # 8x8 block-DCT (H.264/JPEG-like) quantisation
            "forced_saturation_prob": F("float", min=0.0, max=1.0),
            "sclera_exposure_prob": F("float", min=0.0, max=1.0, required=False),  # expose for the sclera, not the skin
            "monochrome_prob": F("float", min=0.0, max=1.0, required=False),  # black-and-white delivery
            # Round 2: draw portrait shots at deployment (1080p portrait) native iris sizes with this
            # probability; the sampler otherwise draws the native size log-uniformly from 8 px up to
            # the rendered size, which makes the size mix independent of the rendered framing
            "portrait_native_iris": F("map", required=False, children={
                "prob": F("float", min=0.0, max=1.0),
                "min_px": F("float", min=4.0, max=512.0),
                "max_px": F("float", min=4.0, max=512.0),
            }),
            # Round 4: translate the rendered reflection layer within the eye by up to max_frac iris
            # diameters (random direction) with this probability, so bright reflections land over the
            # iris at every position relative to the pupil.  Measured need: the converged restorer
            # fills a catchlight above/level with the pupil pupil-dark on real footage (0.28 of the
            # true iris) while one below the pupil is filled correctly (1.05); in CG, strong
            # reflections lie over the pupil 46-82% of the time by sector.  No re-render needed.
            "reflection_shift": F("map", required=False, children={
                "prob": F("float", min=0.0, max=1.0),
                "max_frac": F("float", min=0.0, max=2.0),
            }),
        }),
        "demographics": F("map", required=False, children={
            "ages": F("str", choices=("adult_uniform", "child_to_90s_balanced")),
            "sex": F("str", choices=("continuous", "female_male")),
            "match_skin_to_age_and_ancestry": F("bool"),
            "exclude_makeup_and_tattoo_skins": F("bool"),
            "natural_makeup_prob_female": F("float", min=0.0, max=1.0, required=False),  # everyday makeup, female heads only
        }),
        "catchlights": F("map", required=False, children={
            "key_softbox_prob": F("float", min=0.0, max=1.0),  # bright large key reflected in the upper iris / near the lid
            "relaxed_lid_prob": F("float", min=0.0, max=1.0),  # upper lid resting over the top of the iris
            "clean_iris_render": F("bool"),  # suppress refractive fireflies (shared clean layer)
        }),
    }),
    "model": F("map", children={
        "baseline": F("str", choices=("eyeguard_plus_temporal_naf_style",)),
        "widths": F("list_int"),
        "encoder_blocks": F("list_int"),
        "crop_size": F("int", choices=(128, 192, 256, 384, 512)),
        "temporal_frames": F("int", choices=(5,)),
        "temporal_fusion": F("str", choices=("local_sdpa_or_temporal_convolution",)),
        "max_parameters_target": F("int", min=1),
        "canonical_prior": F("str", choices=("fixed_per_shot_cuda_procedural_atlas",)),
        "require_pretrained_reflection_weights": F("bool", const=False),
        "use_external_flow": F("bool"),
        # optional v2: per-eye visible-reflection presence head on EyeGuard + visibility-based reflection targets
        "reflection_presence": F("bool", required=False),
        "reflection_visibility": F("float", min=0.0, max=1.0, required=False),
        "conf_target": F("str", choices=("tolerance", "improvement"), required=False),  # v2: 'improvement'
    }),
    "training": F("map", children={
        "optimizer": F("str", choices=("adamw",)),
        "learning_rate": F("float", min=0.0),
        "weight_decay": F("float", min=0.0),
        "warmup_steps": F("int", min=0),
        "schedule": F("str", choices=("cosine",)),
        "gradient_clip": F("float", min=0.0),
        "ema_decay": F("float", min=0.0, max=1.0),
        "activation_checkpointing": F("bool"),
        "guard": _stage(),
        "image": _stage(),
        "temporal": _stage(),
        "presence": F("map", required=False, children={"steps": F("int", min=1), "microbatch": F("int", min=1), "effective_batch": F("int", min=1)}),
        "checkpoint_every_steps": F("int", min=1),
        "validation_every_steps": F("int", min=1),
        "resume": F("str", choices=("auto", "never", "require")),
    }),
    "inference": F("map", children={
        "default_mode": F("str", choices=("conservative", "hybrid")),
        "synthesis_requires_explicit_flag": F("bool", const=True),
        "max_faces": F("int", min=1),
        "minimum_iris_diameter_pixels": F("int", min=1),
        "frame_buffer_count": F("int", min=5),
        "canonical_analysis_pass": F("bool"),
        "hard_gate_after_inverse_warp": F("bool", const=True),
        "preserve_source_alpha": F("bool", const=True),
        "fail_closed_on_uncertain_anatomy": F("bool", const=True),
        # optional v2: choose gate thresholds on validation data at bundle time (test never used)
        "gate_selection": F("str", choices=("fixed", "validation"), required=False),
    }),
    "color": F("map", children={
        "working_color": F("str", choices=("linear-acescg", "linear-rec709")),
        "input_color": F("str", choices=("require_metadata_or_explicit",)),
        "input_range": F("str", choices=("require_metadata_or_explicit",)),
        "model_mapping": F("str", choices=("invertible_exposure_asinh",)),
    }),
    "output": F("map", children={
        "authoritative": F("str", choices=("cropped_signed_float_exr",)),
        "exr_compression": F("str", choices=("none",)),
        "preview_encoder": F("str", choices=("nvenc",)),
        "preview_codec": F("str", choices=("auto_supported", "h264", "hevc", "av1")),
        "include_ground_truth_in_demo_only": F("bool", const=True),
        "preserve_source_hash": F("bool", const=True),
        "export_full_frame_master_by_default": F("bool"),
    }),
    "validation": F("map", children={
        "contract": F("str"),
        "report_schema": F("str"),
        "require_gpu_trace": F("bool"),
        "require_independent_full_frame_detection": F("bool"),
        "run_4k_preservation_fixture": F("bool"),
        "user_footage_sample_max_frames": F("int", min=1),
        "real_quality_without_footage": F("str", choices=("not_tested",)),
        "smoke_only": F("bool", required=False),
    }),
}


def _suggest(key: str, options: list[str]) -> str:
    m = difflib.get_close_matches(key, options, n=1)
    return f" (did you mean '{m[0]}'?)" if m else ""


def validate(cfg: dict[str, Any]) -> None:
    """Validate a fully merged configuration mapping in place; raise ConfigError on any problem."""
    _validate_map(cfg, SCHEMA, "")
    # cross-field invariants
    tr = cfg["training"]
    for st in ("guard", "image", "temporal"):
        s = tr[st]
        if s["effective_batch"] % s["microbatch"] != 0:
            raise ConfigError(f"training.{st}.effective_batch must be a multiple of microbatch")
    total = sum(tr[st]["steps"] for st in ("guard", "image", "temporal")) + (tr["presence"]["steps"] if "presence" in tr else 0)
    if total > cfg["resources"]["max_total_optimizer_steps"]:
        raise ConfigError(
            f"base optimizer steps {total} exceed resources.max_total_optimizer_steps "
            f"{cfg['resources']['max_total_optimizer_steps']}"
        )
    m = cfg["model"]
    if len(m["widths"]) != len(m["encoder_blocks"]):
        raise ConfigError("model.widths and model.encoder_blocks must have equal length")
    if cfg["render"]["denoising"]:
        raise ConfigError(
            "render.denoising=true is not supported: training pairs must stay undenoised "
            "(independent denoising would corrupt the reflection decomposition)"
        )
    if cfg["gpu"]["reserve_free_gib"] >= cfg["gpu"]["target_working_gib"] + 16:
        raise ConfigError("gpu.reserve_free_gib is implausibly large")


def _validate_map(obj: Any, spec: dict[str, F], path: str) -> None:
    if not isinstance(obj, dict):
        raise ConfigError(f"{path or '<root>'}: expected a mapping, got {type(obj).__name__}")
    for k in obj:
        if k not in spec:
            raise ConfigError(f"unknown configuration key '{path + k}'{_suggest(k, list(spec))}")
    for k, f in spec.items():
        p = path + k
        if k not in obj:
            if f.required:
                raise ConfigError(f"missing required configuration key '{p}'")
            continue
        _validate_value(obj[k], f, p)


def _validate_value(v: Any, f: F, p: str) -> None:
    if f.kind == "map":
        _validate_map(v, f.children or {}, p + ".")
        return
    if f.kind == "bool":
        if not isinstance(v, bool):
            raise ConfigError(f"{p}: expected true/false, got {v!r}")
    elif f.kind == "int":
        if isinstance(v, bool) or not isinstance(v, int):
            raise ConfigError(f"{p}: expected an integer, got {v!r}")
    elif f.kind == "float":
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ConfigError(f"{p}: expected a number, got {v!r}")
    elif f.kind == "str":
        if not isinstance(v, str):
            raise ConfigError(f"{p}: expected a string, got {v!r}")
    elif f.kind == "list_int":
        if not isinstance(v, list) or not v or not all(isinstance(x, int) and not isinstance(x, bool) and x > 0 for x in v):
            raise ConfigError(f"{p}: expected a non-empty list of positive integers, got {v!r}")
    else:  # pragma: no cover - schema bug
        raise ConfigError(f"{p}: unknown schema kind {f.kind}")
    if f.choices is not None and v not in f.choices:
        raise ConfigError(f"{p}: {v!r} is not one of {list(f.choices)}")
    if f.min is not None and isinstance(v, (int, float)) and v < f.min:
        raise ConfigError(f"{p}: {v!r} is below the minimum {f.min}")
    if f.max is not None and isinstance(v, (int, float)) and v > f.max:
        raise ConfigError(f"{p}: {v!r} is above the maximum {f.max}")
    if f.const is not None and v != f.const:
        raise ConfigError(f"{p}: must be {f.const!r} under the implementation contract, got {v!r}")


@dataclass
class Config:
    """Validated, merged configuration with convenient attribute access."""

    raw: dict[str, Any]
    source_paths: list[str] = field(default_factory=list)

    def __getitem__(self, k: str) -> Any:
        return self.raw[k]

    def get(self, dotted: str, default: Any = None) -> Any:
        cur: Any = self.raw
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

"""Strict GPU hot-path auditing.

Three independent mechanisms:

1. ``GpuAudit`` - a ``TorchDispatchMode`` that sees every ATen operator executed while it is
   active.  Any operator touching a CPU tensor larger than a scalar-sized threshold is a
   *CPU numerical fallback* and raises in strict mode.  Host/device copies are recorded and
   must happen inside an explicitly declared boundary (e.g. ``exr_payload_upload``); the
   allowlist below explains each one.  Scalar read-backs (``.item()``) are counted as
   control/logging transfers.
2. ``profile_stage`` - torch.profiler CPU+CUDA traces per stage with kernel/memcpy counts.
3. ``static_hot_path_scan`` - AST scan proving hot-path modules import no NumPy/OpenCV/PIL/
   SciPy image libraries (NumPy is allowed only in the declared raw-byte helper).
"""

from __future__ import annotations

import ast
import json
import time
import traceback
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

from .runtime import GpuPolicyError

# Declared host<->device boundaries (the transfer allowlist).
TRANSFER_ALLOWLIST: dict[str, str] = {
    "exr_payload_upload": "raw EXR scanline bytes (uncompressed, or zlib-inflated ZIP/ZIPS chunks still pre-filtered) and their chunk indices copied to CUDA before GPU unpacking",
    "exr_payload_download": "GPU-packed (and for ZIP/ZIPS GPU-pre-filtered) EXR scanline payload copied to host for file writing / zlib deflate",
    "render_buffer_upload": "Blender/Cycles render EXR bytes (renderer I/O boundary) copied to CUDA",
    "training_cache_upload": "float16/uint8 training-cache tensors read from safetensors bytes to CUDA",
    "checkpoint_io": "model weights loaded from / saved to safetensors",
    "codec_bitstream": "NVENC compressed bitstream returned to host for container muxing",
    "preview_png": "8-bit display preview image bytes downloaded for PNG/HTML gallery writing",
    "metadata_readback": "small per-frame results (boxes, track ids, scalar metrics) read back for JSON/logging",
    "mask_download": "uint8/float mask tensors downloaded for EXR/PNG layer writing",
    "test_reference": "offline unit-test comparison against CPU reference implementations",
}

# Operators that do not compute on pixel values (metadata/view/allocation only).
_NON_NUMERIC_OPS = {
    "aten.detach.default", "aten.alias.default", "aten.view.default", "aten._unsafe_view.default",
    "aten.view.dtype", "aten.lift_fresh.default", "aten.set_.source_Storage_storage_offset",
    "aten.set_.source_Storage", "aten.empty.memory_format", "aten.empty_strided.default",
    "aten.as_strided.default", "aten.slice.Tensor", "aten.select.int", "aten.t.default",
    "aten.permute.default", "aten.reshape.default", "aten.expand.default", "aten.unsqueeze.default",
    "aten.squeeze.dim", "aten.squeeze.default", "aten.split.Tensor", "aten.unbind.int",
    "aten.is_pinned.default", "aten._pin_memory.default", "aten.resize_.default",
    "aten._local_scalar_dense.default", "aten.item.default", "aten.equal.default",
}
_COPY_OPS = {"aten._to_copy.default", "aten.copy_.default", "aten._copy_from.default", "aten._copy_from_and_resize.default"}


@dataclass
class Violation:
    op: str
    detail: str
    stack: str

    def as_dict(self) -> dict:
        return {"op": self.op, "detail": self.detail, "stack": self.stack}


@dataclass
class GpuAudit:
    strict: bool = True
    numel_threshold: int = 64
    violations: list[Violation] = field(default_factory=list)
    transfers: Counter = field(default_factory=Counter)
    transfer_bytes: Counter = field(default_factory=Counter)
    op_counts: Counter = field(default_factory=Counter)
    scalar_readbacks: int = 0
    _boundaries: list[str] = field(default_factory=list)

    @contextmanager
    def boundary(self, name: str) -> Iterator[None]:
        if name not in TRANSFER_ALLOWLIST:
            raise ValueError(f"undeclared transfer boundary {name!r}")
        self._boundaries.append(name)
        try:
            yield
        finally:
            self._boundaries.pop()

    def current_boundary(self) -> str | None:
        return self._boundaries[-1] if self._boundaries else None

    def _flag(self, op: str, detail: str) -> None:
        stack = "".join(traceback.format_stack(limit=12)[:-3])
        v = Violation(op, detail, stack[-2000:])
        self.violations.append(v)
        if self.strict:
            raise GpuPolicyError(f"strict GPU audit: {detail} in {op}\n{stack[-1200:]}")

    def record(self, func: Any, args: tuple, kwargs: dict, out: Any) -> None:
        name = str(func)
        self.op_counts[name] += 1
        flat_in, _ = tree_flatten((args, kwargs))
        flat_out, _ = tree_flatten(out)
        tin = [t for t in flat_in if isinstance(t, torch.Tensor)]
        tout = [t for t in flat_out if isinstance(t, torch.Tensor)]
        if name in ("aten._local_scalar_dense.default", "aten.item.default"):
            if tin and tin[0].is_cuda:
                self.scalar_readbacks += 1
            return
        if name in _COPY_OPS:
            src = dst = None
            if name == "aten._to_copy.default" and tin and tout:
                src, dst = tin[0], tout[0]
            elif name == "aten.copy_.default" and len(tin) >= 2:
                dst, src = tin[0], tin[1]
            elif name.startswith("aten._copy_from") and len(tin) >= 2:
                src, dst = tin[0], tin[1]
            if src is not None and dst is not None and src.device.type != dst.device.type:
                direction = "HtoD" if src.device.type == "cpu" else "DtoH"
                nbytes = src.numel() * src.element_size()
                b = self.current_boundary()
                key = f"{direction}:{b or 'UNDECLARED'}"
                self.transfers[key] += 1
                self.transfer_bytes[key] += nbytes
                if b is None and src.numel() > self.numel_threshold:
                    self._flag(name, f"undeclared {direction} transfer of {nbytes} bytes")
                return
            if src is not None and dst is not None and src.device.type == "cuda":
                return  # device-to-device copy / dtype cast on GPU
        if name in _NON_NUMERIC_OPS:
            return
        big_cpu = [t for t in tin + tout if t.device.type == "cpu" and t.numel() > self.numel_threshold]
        if big_cpu:
            # CPU-side arithmetic inside a declared byte boundary is still not allowed, except
            # pure copies (handled above).  Offline tests declare 'test_reference'.
            if self.current_boundary() == "test_reference":
                return
            shapes = [tuple(t.shape) for t in big_cpu][:3]
            self._flag(name, f"CPU numerical op on tensors {shapes}")

    def summary(self) -> dict:
        return {
            "strict": self.strict,
            "numel_threshold": self.numel_threshold,
            "cpu_numerical_image_fallbacks": len(self.violations),
            "violations": [v.as_dict() for v in self.violations[:20]],
            "transfers": dict(self.transfers),
            "transfer_bytes": dict(self.transfer_bytes),
            "undeclared_transfers": sum(v for k, v in self.transfers.items() if k.endswith("UNDECLARED")),
            "scalar_readbacks": self.scalar_readbacks,
            "distinct_ops": len(self.op_counts),
            "total_ops": int(sum(self.op_counts.values())),
            "allowlist": TRANSFER_ALLOWLIST,
        }


class _AuditMode(TorchDispatchMode):
    def __init__(self, audit: GpuAudit):
        super().__init__()
        self.audit = audit

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        self.audit.record(func, args, kwargs, out)
        return out


_ACTIVE: list[GpuAudit] = []


@contextmanager
def audited(audit: GpuAudit) -> Iterator[GpuAudit]:
    """Run a block under the dispatch-mode audit."""
    _ACTIVE.append(audit)
    try:
        with _AuditMode(audit):
            yield audit
    finally:
        _ACTIVE.pop()


@contextmanager
def boundary(name: str) -> Iterator[None]:
    """Declare a host/device transfer boundary for the innermost active audit (no-op if none)."""
    if name not in TRANSFER_ALLOWLIST:
        raise ValueError(f"undeclared transfer boundary {name!r}")
    if not _ACTIVE:
        yield
        return
    with _ACTIVE[-1].boundary(name):
        yield


def active_audit() -> GpuAudit | None:
    return _ACTIVE[-1] if _ACTIVE else None


@contextmanager
def profile_stage(name: str, out_dir: Path | None, enabled: bool = True) -> Iterator[dict]:
    """torch.profiler CPU+CUDA trace for a stage; writes <out_dir>/<name>.trace.json and summary."""
    info: dict = {"stage": name}
    if not enabled or out_dir is None:
        t0 = time.perf_counter()
        yield info
        info["seconds"] = time.perf_counter() - t0
        return
    from torch.profiler import ProfilerActivity, profile

    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False, with_stack=False) as prof:
        yield info
        torch.cuda.synchronize()
    info["seconds"] = time.perf_counter() - t0
    trace = out_dir / f"{name}.trace.json"
    prof.export_chrome_trace(str(trace))
    kernels = 0
    memcpy = Counter()
    cuda_us = 0.0
    for ev in prof.events():
        dt = getattr(ev, "device_type", None)
        if dt is not None and "CUDA" in str(dt):
            nm = ev.name
            if "Memcpy" in nm or "memcpy" in nm:
                if "HtoD" in nm:
                    memcpy["HtoD"] += 1
                elif "DtoH" in nm:
                    memcpy["DtoH"] += 1
                elif "DtoD" in nm:
                    memcpy["DtoD"] += 1
                else:
                    memcpy["other"] += 1
            elif "Memset" not in nm:
                kernels += 1
            cuda_us += float(getattr(ev, "device_time", 0.0) or 0.0)
    info.update({"trace": str(trace), "cuda_kernels": kernels, "memcpy": dict(memcpy), "cuda_time_ms": round(cuda_us / 1000.0, 3)})
    (out_dir / f"{name}.summary.json").write_text(json.dumps(info, indent=2), encoding="utf-8")


# ---------------------------------------------------------------- static scan
HOT_PATH_PACKAGES = (
    "eyeclean.io", "eyeclean.color", "eyeclean.tracking", "eyeclean.models", "eyeclean.inference",
    "eyeclean.gpu", "eyeclean.training.augment", "eyeclean.procedural",
)
FORBIDDEN_IMPORTS = {"numpy", "cv2", "PIL", "scipy", "skimage", "imageio", "OpenEXR", "colour", "av"}
# Modules allowed to import NumPy purely as a byte-buffer view (no arithmetic), with reason.
NUMPY_BUFFER_ALLOWLIST = {
    "eyeclean.io.rawbytes": "tensor<->bytes buffer views for raw file I/O; no arithmetic",
}


def static_hot_path_scan(src_root: Path) -> dict:
    findings = []
    scanned = []
    for py in sorted(src_root.rglob("*.py")):
        rel = py.relative_to(src_root.parent).with_suffix("")
        mod = ".".join(rel.parts)
        if not any(mod == p or mod.startswith(p + ".") for p in HOT_PATH_PACKAGES):
            continue
        scanned.append(mod)
        tree = ast.parse(py.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names = [node.module.split(".")[0]]
            for n in names:
                if n in FORBIDDEN_IMPORTS:
                    if n == "numpy" and mod in NUMPY_BUFFER_ALLOWLIST:
                        continue
                    findings.append({"module": mod, "import": n, "line": node.lineno})
    return {"scanned_modules": scanned, "forbidden_imports": findings, "numpy_buffer_allowlist": NUMPY_BUFFER_ALLOWLIST,
            "passed": not findings}

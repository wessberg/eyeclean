"""CUDA device policy, memory budgeting and host/device boundary helpers.

Strict policy: every numerical image operation runs on a CUDA device.  There is no CPU
fallback: if CUDA is unavailable the command fails with a precise error.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field

import torch


class GpuPolicyError(RuntimeError):
    """Raised when the strict GPU policy cannot be satisfied (CLI exit code 3)."""


@dataclass
class DeviceInfo:
    index: int
    name: str
    capability: tuple[int, int]
    total_gib: float
    free_gib_at_start: float
    sm_count: int
    driver_version: str | None
    torch_version: str
    torch_cuda: str | None
    cudnn: int | None


@dataclass
class MemoryBudget:
    """Measured device-memory budget (not only allocator statistics)."""

    total_gib: float
    free_at_start_gib: float
    reserve_free_gib: float
    target_working_gib: float
    limit_gib: float
    adaptations: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "total_gib": round(self.total_gib, 3),
            "free_at_start_gib": round(self.free_at_start_gib, 3),
            "reserve_free_gib": self.reserve_free_gib,
            "target_working_gib": self.target_working_gib,
            "limit_gib": round(self.limit_gib, 3),
            "adaptations": list(self.adaptations),
        }


_DEVICE: torch.device | None = None


def nvidia_driver_version() -> str | None:
    """Query the driver version through NVML if available, else nvidia-smi (metadata only)."""
    try:
        import subprocess

        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20,
        )
        if out.returncode == 0:
            return out.stdout.strip().splitlines()[0].strip()
    except Exception:
        pass
    return None


def require_cuda(policy: str = "strict", device_index: int = 0) -> torch.device:
    """Return the CUDA device or raise GpuPolicyError. Never returns a CPU device."""
    global _DEVICE
    if policy != "strict":
        raise GpuPolicyError(f"unsupported --gpu-policy {policy!r}; only 'strict' is implemented")
    if not torch.cuda.is_available():
        raise GpuPolicyError(
            "strict GPU policy: CUDA is not available to PyTorch "
            f"(torch {torch.__version__}, built for CUDA {torch.version.cuda}). "
            "No CPU fallback exists for image processing."
        )
    if device_index >= torch.cuda.device_count():
        raise GpuPolicyError(f"CUDA device index {device_index} not present ({torch.cuda.device_count()} devices)")
    dev = torch.device("cuda", device_index)
    torch.cuda.set_device(dev)
    cc = torch.cuda.get_device_capability(dev)
    arch = f"sm_{cc[0]}{cc[1]}"
    if arch not in torch.cuda.get_arch_list():
        raise GpuPolicyError(f"PyTorch build lacks kernels for {arch}; arch list {torch.cuda.get_arch_list()}")
    _DEVICE = dev
    return dev


def device() -> torch.device:
    if _DEVICE is None:
        return require_cuda()
    return _DEVICE


def device_info(dev: torch.device | None = None) -> DeviceInfo:
    dev = dev or device()
    p = torch.cuda.get_device_properties(dev)
    free, total = torch.cuda.mem_get_info(dev)
    return DeviceInfo(
        index=dev.index or 0,
        name=p.name,
        capability=(p.major, p.minor),
        total_gib=total / 2**30,
        free_gib_at_start=free / 2**30,
        sm_count=p.multi_processor_count,
        driver_version=nvidia_driver_version(),
        torch_version=torch.__version__,
        torch_cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(),
    )


def configure_memory_budget(target_working_gib: float, reserve_free_gib: float, dev: torch.device | None = None) -> MemoryBudget:
    """Set a per-process memory fraction from *measured* free device memory.

    limit = min(target_working_gib, free_now - reserve_free_gib); the adaptive lower limit
    applies when other applications already hold device memory.
    """
    dev = dev or device()
    free, total = torch.cuda.mem_get_info(dev)
    free_g, total_g = free / 2**30, total / 2**30
    limit = min(target_working_gib, free_g - reserve_free_gib)
    budget = MemoryBudget(total_g, free_g, reserve_free_gib, target_working_gib, limit)
    if limit < target_working_gib:
        budget.adaptations.append(
            f"working limit lowered from {target_working_gib:.2f} to {limit:.2f} GiB because only "
            f"{free_g:.2f} GiB was free at start (reserve {reserve_free_gib:.2f} GiB)"
        )
    if limit < 2.0:
        raise GpuPolicyError(f"only {free_g:.2f} GiB free on {torch.cuda.get_device_name(dev)}; cannot keep a {reserve_free_gib} GiB reserve")
    torch.cuda.set_per_process_memory_fraction(min(1.0, limit / total_g), dev)
    return budget


def measured_device_used_gib(dev: torch.device | None = None) -> float:
    """Whole-device used memory measured by the driver (includes other processes)."""
    free, total = torch.cuda.mem_get_info(dev or device())
    return (total - free) / 2**30


class PeakMemoryMonitor:
    """Track peak allocator and device-level memory over a stage (measured, not estimated)."""

    def __init__(self, dev: torch.device | None = None):
        self.dev = dev or device()
        self.peak_device_used_gib = 0.0
        self.baseline_device_used_gib = measured_device_used_gib(self.dev)
        torch.cuda.reset_peak_memory_stats(self.dev)

    def sample(self) -> None:
        self.peak_device_used_gib = max(self.peak_device_used_gib, measured_device_used_gib(self.dev))

    def result(self) -> dict:
        self.sample()
        return {
            "peak_allocated_gib": round(torch.cuda.max_memory_allocated(self.dev) / 2**30, 4),
            "peak_reserved_gib": round(torch.cuda.max_memory_reserved(self.dev) / 2**30, 4),
            "peak_device_used_gib": round(self.peak_device_used_gib, 4),
            "baseline_device_used_gib": round(self.baseline_device_used_gib, 4),
        }


def cuda_kernel_smoke(dev: torch.device | None = None) -> dict:
    """Execute real kernels on the device and verify results; returns evidence."""
    dev = dev or device()
    t0 = time.perf_counter()
    x = torch.arange(1 << 20, device=dev, dtype=torch.float32)
    s = float((x * 2 + 1).sum().item())
    expect = float((1 << 20) * ((1 << 20) - 1) + (1 << 20))
    a = torch.randn(1024, 1024, device=dev, dtype=torch.bfloat16)
    b = torch.randn(1024, 1024, device=dev, dtype=torch.bfloat16)
    c = (a @ b).float()
    ref = a.float() @ b.float()
    rel = float(((c - ref).abs().max() / ref.abs().max()).item())
    q = torch.randn(1, 4, 256, 64, device=dev, dtype=torch.bfloat16)
    o = torch.nn.functional.scaled_dot_product_attention(q, q, q)
    torch.cuda.synchronize(dev)
    cc = torch.cuda.get_device_capability(dev)
    return {
        "device": torch.cuda.get_device_name(dev),
        "capability": f"{cc[0]}.{cc[1]}",
        "arch": f"sm_{cc[0]}{cc[1]}",
        "blackwell": cc[0] >= 10,
        "sum_exact": s == expect,
        "bf16_matmul_rel_err": rel,
        "sdpa_ok": bool(torch.isfinite(o).all().item()),
        "seconds": round(time.perf_counter() - t0, 4),
        "in_torch_arch_list": f"sm_{cc[0]}{cc[1]}" in torch.cuda.get_arch_list(),
        "passed": bool(s == expect and rel < 0.05 and f"sm_{cc[0]}{cc[1]}" in torch.cuda.get_arch_list()),
    }


def ensure_cuda12_runtime_on_dll_path() -> None:
    """PyNvVideoCodec 2.2.2 links cudart64_12.dll; torch cu130 bundles only CUDA 13.

    The project depends on NVIDIA's redistributable ``nvidia-cuda-runtime-cu12`` wheel; this
    adds its ``bin`` directory to the Windows DLL search path before importing the codec.
    """
    if sys.platform != "win32":
        # Linux adapter: preload the wheel's libcudart.so.12 globally if present (untested on this host)
        import ctypes
        import glob

        for lib in glob.glob(os.path.join(sys.prefix, "lib", "python3*", "site-packages", "nvidia", "cuda_runtime", "lib", "libcudart.so.12*")):
            try:
                ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
                break
            except OSError:
                pass
        return
    cand = os.path.join(sys.prefix, "Lib", "site-packages", "nvidia", "cuda_runtime", "bin")
    if os.path.isdir(cand):
        os.add_dll_directory(cand)

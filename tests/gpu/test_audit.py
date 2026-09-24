from pathlib import Path

import pytest
import torch

from eyeclean.gpu.audit import GpuAudit, audited, boundary, static_hot_path_scan
from eyeclean.gpu.runtime import GpuPolicyError, cuda_kernel_smoke

ROOT = Path(__file__).resolve().parents[2]


def test_cuda_kernel_smoke(dev):
    r = cuda_kernel_smoke(dev)
    assert r["passed"] and r["in_torch_arch_list"] and r["sdpa_ok"]


def test_cpu_numerical_op_is_caught(dev):
    a = GpuAudit(strict=True)
    with pytest.raises(GpuPolicyError):
        with audited(a):
            x = torch.rand((64, 64))  # CPU image-sized tensor
            _ = torch.nn.functional.avg_pool2d(x[None, None], 2)


def test_undeclared_transfer_is_caught(dev):
    a = GpuAudit(strict=True)
    x = torch.rand((256, 256))
    with pytest.raises(GpuPolicyError):
        with audited(a):
            x.to(dev)


def test_declared_boundary_allowed(dev):
    a = GpuAudit(strict=True)
    x = torch.frombuffer(bytearray(4096), dtype=torch.uint8)
    with audited(a):
        with boundary("exr_payload_upload"):
            y = x.to(dev)
        z = (y.float() * 2).sum()
        float(z)
    s = a.summary()
    assert s["cpu_numerical_image_fallbacks"] == 0 and s["transfers"] == {"HtoD:exr_payload_upload": 1} and s["scalar_readbacks"] >= 1


def test_static_scan_hot_path():
    r = static_hot_path_scan(ROOT / "src" / "eyeclean")
    assert r["passed"], r["forbidden_imports"]
    assert "eyeclean.inference.pipeline" in r["scanned_modules"]

"""`gpu.compile: auto_cuda_only` — try torch.compile on CUDA; otherwise CUDA eager with a recorded reason.

Compilation is an optimisation, never a prerequisite, and never a reason to move work to the CPU.
"""

from __future__ import annotations

import functools

import torch


@functools.lru_cache(maxsize=1)
def probe_compile() -> dict:
    try:
        m = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3, padding=1), torch.nn.ReLU()).cuda()
        c = torch.compile(m)
        c(torch.randn(1, 3, 16, 16, device="cuda"))
        torch.cuda.synchronize()
        return {"available": True, "mode": "torch.compile(inductor)", "reason": None}
    except Exception as e:  # e.g. no Triton on Windows
        return {"available": False, "mode": "cuda_eager", "reason": f"{type(e).__name__}: {str(e).splitlines()[0][:240]}"}


def maybe_compile(model: torch.nn.Module, policy: str) -> tuple[torch.nn.Module, dict]:
    if policy == "disabled":
        return model, {"available": None, "mode": "cuda_eager", "reason": "disabled by configuration"}
    info = probe_compile()
    if info["available"]:
        return torch.compile(model), info
    return model, info

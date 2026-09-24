"""Declared raw-byte boundary between files and CUDA tensors.

This is the only hot-path module allowed to import NumPy, and only to expose tensor memory as
a byte buffer for ``file.write`` / to wrap ``bytes`` without copying.  No arithmetic happens here.
"""

from __future__ import annotations

import numpy as np  # buffer views only (see eyeclean.gpu.audit.NUMPY_BUFFER_ALLOWLIST)
import torch

from ..gpu.audit import boundary


def bytes_to_cuda(buf: bytes | bytearray | memoryview, device: torch.device, name: str = "exr_payload_upload") -> torch.Tensor:
    """Wrap raw bytes as a uint8 CPU tensor and copy them to the device (declared boundary)."""
    if len(buf) == 0:
        return torch.empty(0, dtype=torch.uint8, device=device)
    host = torch.frombuffer(bytearray(buf) if isinstance(buf, bytes) else buf, dtype=torch.uint8)
    with boundary(name):
        return host.to(device, non_blocking=False)


def cuda_to_bytes(t: torch.Tensor, name: str = "exr_payload_download") -> memoryview:
    """Copy a contiguous uint8 CUDA tensor to host and expose it as a memoryview for raw writing."""
    if t.dtype != torch.uint8:
        raise TypeError("cuda_to_bytes expects a uint8 tensor (pack on GPU first)")
    with boundary(name):
        host = t.contiguous().cpu()
    return memoryview(np.asarray(host.numpy()).reshape(-1))

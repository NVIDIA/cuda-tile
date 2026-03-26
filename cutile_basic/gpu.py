"""GPU utilities (architecture detection, etc.)."""

from __future__ import annotations

from cuda.core import Device


def detect_gpu_arch() -> str:
    """Return the GPU architecture string (e.g. 'sm_120') for device 0."""
    dev = Device(0)
    cc = dev.compute_capability
    return f"sm_{cc[0] * 10 + cc[1]}"

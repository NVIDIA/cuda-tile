"""GPU kernel launch via cuda.core and cupy."""

from __future__ import annotations

import cupy as cp
from cuda.core import Device, LaunchConfig, ObjectCode, launch


class GpuRunnerError(Exception):
    pass


def detect_gpu_arch() -> str:
    """Return the GPU architecture string (e.g. 'sm_80') for device 0."""
    dev = Device(0)
    cc = dev.compute_capability
    return f"sm_{cc[0] * 10 + cc[1]}"


def launch_kernel(
    cubin_path: str,
    inputs: dict[str, list[float]] | None = None,
    outputs: list[str] | None = None,
    param_order: list[str] | None = None,
    sizes: dict[str, int] | None = None,
    grid_size: int = 1,
    kernel_name: str = "main",
) -> dict[str, list[float]]:
    """Launch a compiled .cubin kernel with host-device memory transfers.

    Args:
        cubin_path: Path to compiled .cubin file.
        inputs: name -> list of float values to copy to device before launch.
        outputs: names of arrays to copy back from device after launch.
        param_order: ordered list of kernel parameter names (must match
            the kernel signature). If None, inferred from inputs + outputs.
        sizes: name -> number of f32 elements for each parameter.
            If None, inferred from the lengths of the input arrays.
        grid_size: number of thread blocks (grid X dimension).
        kernel_name: entry point name in the cubin.

    Returns:
        dict mapping each output name to its list of float results.
        Empty dict when outputs is None or empty.
    """
    inputs = inputs or {}
    outputs = outputs or []

    if param_order is None:
        seen: set[str] = set()
        param_order = []
        for name in list(inputs.keys()) + outputs:
            if name not in seen:
                seen.add(name)
                param_order.append(name)

    if sizes is None:
        sizes = {name: len(data) for name, data in inputs.items()}

    if not param_order:
        return _launch_no_params(cubin_path, grid_size, kernel_name)

    dev = Device(0)
    dev.set_current()
    s = dev.create_stream()

    obj = ObjectCode.from_cubin(cubin_path)
    kernel = obj.get_kernel(kernel_name)

    arrays: dict[str, cp.ndarray] = {}
    for name in param_order:
        n_elems = sizes.get(name, 0)
        if name in inputs:
            arrays[name] = cp.array(inputs[name], dtype=cp.float32)
        else:
            arrays[name] = cp.zeros(n_elems, dtype=cp.float32)

    config = LaunchConfig(grid=(grid_size, 1, 1), block=(1, 1, 1))
    kernel_args = [arrays[name].data.ptr for name in param_order]
    launch(s, config, kernel, *kernel_args)
    s.sync()

    results: dict[str, list[float]] = {}
    for name in outputs:
        results[name] = cp.asnumpy(arrays[name]).tolist()

    return results


def _launch_no_params(cubin_path: str, grid_size: int, kernel_name: str) -> dict:
    """Launch a kernel that takes no parameters."""
    dev = Device(0)
    dev.set_current()
    s = dev.create_stream()

    obj = ObjectCode.from_cubin(cubin_path)
    kernel = obj.get_kernel(kernel_name)

    config = LaunchConfig(grid=(grid_size, 1, 1), block=(1, 1, 1))
    launch(s, config, kernel)
    s.sync()

    return {}

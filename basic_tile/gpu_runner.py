"""GPU kernel launch via CUDA driver API (ctypes)."""

from __future__ import annotations

import array
import ctypes
from pathlib import Path


class GpuRunnerError(Exception):
    pass


def _get_libcuda():
    try:
        return ctypes.CDLL("libcuda.so")
    except OSError:
        raise GpuRunnerError(
            "Cannot load libcuda.so. Ensure NVIDIA driver is installed."
        )


def _check(name: str, ret: int):
    if ret != 0:
        raise GpuRunnerError(f"CUDA driver error in {name}: code {ret}")


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

    # Infer param_order if not given
    if param_order is None:
        seen: set[str] = set()
        param_order = []
        for name in list(inputs.keys()) + outputs:
            if name not in seen:
                seen.add(name)
                param_order.append(name)

    # Infer sizes from input data if not given
    if sizes is None:
        sizes = {}
        for name, data in inputs.items():
            sizes[name] = len(data)

    # No-param kernel (e.g. print-only)
    if not param_order:
        return _launch_no_params(cubin_path, grid_size, kernel_name)

    libcuda = _get_libcuda()
    _check("cuInit", libcuda.cuInit(0))

    device = ctypes.c_int()
    _check("cuDeviceGet", libcuda.cuDeviceGet(ctypes.byref(device), 0))

    ctx = ctypes.c_void_p()
    _check("cuCtxCreate", libcuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, device))

    d_ptrs: dict[str, ctypes.c_uint64] = {}
    h_arrays: dict[str, array.array] = {}

    try:
        # Load module
        module = ctypes.c_void_p()
        cubin_data = Path(cubin_path).read_bytes()
        _check(
            "cuModuleLoadData",
            libcuda.cuModuleLoadData(ctypes.byref(module), cubin_data),
        )
        func = ctypes.c_void_p()
        _check(
            "cuModuleGetFunction",
            libcuda.cuModuleGetFunction(
                ctypes.byref(func), module, kernel_name.encode("utf-8")
            ),
        )

        # Allocate device memory for each parameter
        for name in param_order:
            n_elems = sizes.get(name, 0)
            nbytes = n_elems * 4  # f32
            d_ptr = ctypes.c_uint64()
            _check(f"cuMemAlloc({name})", libcuda.cuMemAlloc_v2(ctypes.byref(d_ptr), nbytes))
            d_ptrs[name] = d_ptr

        # Copy inputs host → device
        for name, data in inputs.items():
            if name not in d_ptrs:
                continue
            h_arr = array.array("f", data)
            h_arrays[name] = h_arr
            nbytes = sizes[name] * 4
            h_buf = (ctypes.c_char * nbytes).from_buffer(h_arr)
            _check(
                f"cuMemcpyHtoD({name})",
                libcuda.cuMemcpyHtoD_v2(
                    d_ptrs[name], ctypes.addressof(h_buf), nbytes,
                ),
            )

        # Build kernel params (array of pointers to device pointers)
        n_params = len(param_order)
        param_values = (ctypes.c_uint64 * n_params)(
            *(d_ptrs[name].value for name in param_order)
        )
        param_ptrs = (ctypes.c_void_p * n_params)(
            *(ctypes.addressof(param_values) + i * 8 for i in range(n_params))
        )

        # Launch
        _check(
            "cuLaunchKernel",
            libcuda.cuLaunchKernel(
                func,
                grid_size, 1, 1,
                1, 1, 1,
                0, None,
                param_ptrs, None,
            ),
        )
        _check("cuCtxSynchronize", libcuda.cuCtxSynchronize())

        # Copy outputs device → host
        results: dict[str, list[float]] = {}
        for name in outputs:
            n_elems = sizes[name]
            nbytes = n_elems * 4
            h_arr = h_arrays.get(name) or array.array("f", [0.0] * n_elems)
            h_buf = (ctypes.c_char * nbytes).from_buffer(h_arr)
            _check(
                f"cuMemcpyDtoH({name})",
                libcuda.cuMemcpyDtoH_v2(
                    ctypes.addressof(h_buf), d_ptrs[name], nbytes,
                ),
            )
            results[name] = list(h_arr)

        return results

    finally:
        for d_ptr in d_ptrs.values():
            libcuda.cuMemFree_v2(d_ptr)
        libcuda.cuCtxDestroy_v2(ctx)


def _launch_no_params(cubin_path: str, grid_size: int, kernel_name: str) -> dict:
    """Launch a kernel that takes no parameters."""
    libcuda = _get_libcuda()
    _check("cuInit", libcuda.cuInit(0))

    device = ctypes.c_int()
    _check("cuDeviceGet", libcuda.cuDeviceGet(ctypes.byref(device), 0))

    ctx = ctypes.c_void_p()
    _check("cuCtxCreate", libcuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, device))

    try:
        module = ctypes.c_void_p()
        cubin_data = Path(cubin_path).read_bytes()
        _check(
            "cuModuleLoadData",
            libcuda.cuModuleLoadData(ctypes.byref(module), cubin_data),
        )
        func = ctypes.c_void_p()
        _check(
            "cuModuleGetFunction",
            libcuda.cuModuleGetFunction(
                ctypes.byref(func), module, kernel_name.encode("utf-8")
            ),
        )
        _check(
            "cuLaunchKernel",
            libcuda.cuLaunchKernel(
                func,
                grid_size, 1, 1,
                1, 1, 1,
                0, None,
                None, None,
            ),
        )
        _check("cuCtxSynchronize", libcuda.cuCtxSynchronize())
    finally:
        libcuda.cuCtxDestroy_v2(ctx)

    return {}

"""GPU compilation and launch pipeline: .mlir -> .tilebc -> .cubin -> GPU launch."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


class RunnerError(Exception):
    pass


def find_tools(cuda_tile_translate_path: str | None = None) -> dict[str, Path]:
    """Locate cuda-tile-translate and tileiras binaries.

    Search order: explicit path, PATH, /usr/local/cuda/bin/, /tmp/cuda-tile/build/bin/.
    """
    tools: dict[str, Path] = {}

    # cuda-tile-translate
    if cuda_tile_translate_path:
        p = Path(cuda_tile_translate_path)
        if p.is_file() and os.access(p, os.X_OK):
            tools["cuda-tile-translate"] = p
        else:
            raise RunnerError(f"cuda-tile-translate not found or not executable: {p}")
    else:
        for candidate in _search_paths("cuda-tile-translate"):
            if candidate.is_file() and os.access(candidate, os.X_OK):
                tools["cuda-tile-translate"] = candidate
                break
        else:
            raise RunnerError(
                "cuda-tile-translate not found. Run 'make build-tools' or "
                "pass --cuda-tile-translate <path>."
            )

    # tileiras
    for candidate in _search_paths("tileiras"):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            tools["tileiras"] = candidate
            break
    else:
        raise RunnerError(
            "tileiras not found. Ensure CUDA toolkit is installed and "
            "/usr/local/cuda/bin is in PATH."
        )

    return tools


def _search_paths(name: str) -> list[Path]:
    """Return candidate paths for a binary."""
    candidates: list[Path] = []

    # shutil.which checks PATH
    found = shutil.which(name)
    if found:
        candidates.append(Path(found))

    # Common locations
    candidates.append(Path("/usr/local/cuda/bin") / name)
    candidates.append(Path("/tmp/cuda-tile/build/bin") / name)

    return candidates


def compile_mlir_to_tilebc(
    mlir_text: str, output_path: Path, tools: dict[str, Path]
) -> Path:
    """Run cuda-tile-translate to convert MLIR text to .tilebc bytecode."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".mlir", delete=False
    ) as f:
        f.write(mlir_text)
        mlir_path = Path(f.name)

    try:
        cmd = [
            str(tools["cuda-tile-translate"]),
            "--mlir-to-cudatilebc",
            "--bytecode-version=13.1",
            "--no-implicit-module",
            str(mlir_path),
            "-o",
            str(output_path),
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=False
        )
        if result.returncode != 0:
            raise RunnerError(
                f"cuda-tile-translate failed (exit {result.returncode}):\n"
                f"{result.stderr}"
            )
    finally:
        mlir_path.unlink(missing_ok=True)

    if not output_path.exists():
        raise RunnerError(f"cuda-tile-translate produced no output at {output_path}")

    return output_path


def compile_tilebc_to_cubin(
    tilebc_path: Path, output_path: Path, gpu_arch: str, tools: dict[str, Path]
) -> Path:
    """Run tileiras to assemble .tilebc into a .cubin."""
    cmd = [
        str(tools["tileiras"]),
        f"--gpu-name={gpu_arch}",
        str(tilebc_path),
        "-o",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RunnerError(
            f"tileiras failed (exit {result.returncode}):\n{result.stderr}"
        )

    if not output_path.exists():
        raise RunnerError(f"tileiras produced no output at {output_path}")

    return output_path


def detect_gpu_arch() -> str:
    """Query nvidia-smi for the GPU compute capability and return sm_XXX string."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=compute_cap",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError:
        raise RunnerError("nvidia-smi not found. Is the NVIDIA driver installed?")
    except subprocess.CalledProcessError as e:
        raise RunnerError(f"nvidia-smi failed: {e.stderr}")

    # Output like "12.0" -> "sm_120"
    line = result.stdout.strip().splitlines()[0].strip()
    parts = line.split(".")
    if len(parts) != 2:
        raise RunnerError(f"Unexpected compute_cap format: {line!r}")

    major, minor = parts
    arch = f"sm_{major}{minor}"
    return arch


def launch_cubin(cubin_path: Path, kernel_name: str = "main") -> None:
    """Load and launch a .cubin kernel via the CUDA driver API (ctypes)."""
    try:
        libcuda = ctypes.CDLL("libcuda.so")
    except OSError:
        raise RunnerError(
            "Cannot load libcuda.so. Ensure NVIDIA driver is installed."
        )

    # Type aliases
    CUresult = ctypes.c_int
    CUdevice = ctypes.c_int
    CUcontext = ctypes.c_void_p
    CUmodule = ctypes.c_void_p
    CUfunction = ctypes.c_void_p

    def _check(name: str, ret: int):
        if ret != 0:
            raise RunnerError(f"CUDA driver error in {name}: code {ret}")

    # cuInit
    _check("cuInit", libcuda.cuInit(0))

    # cuDeviceGet
    device = CUdevice()
    _check("cuDeviceGet", libcuda.cuDeviceGet(ctypes.byref(device), 0))

    # cuCtxCreate
    ctx = CUcontext()
    _check(
        "cuCtxCreate",
        libcuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, device),
    )

    try:
        # cuModuleLoad
        module = CUmodule()
        cubin_bytes = cubin_path.read_bytes()
        _check(
            "cuModuleLoadData",
            libcuda.cuModuleLoadData(
                ctypes.byref(module), cubin_bytes
            ),
        )

        # cuModuleGetFunction
        func = CUfunction()
        _check(
            "cuModuleGetFunction",
            libcuda.cuModuleGetFunction(
                ctypes.byref(func),
                module,
                kernel_name.encode("utf-8"),
            ),
        )

        # cuLaunchKernel(func, gridDimX=1, gridDimY=1, gridDimZ=1,
        #                blockDimX=1, blockDimY=1, blockDimZ=1,
        #                sharedMemBytes=0, stream=0, kernelParams=NULL, extra=NULL)
        _check(
            "cuLaunchKernel",
            libcuda.cuLaunchKernel(
                func,
                1, 1, 1,  # grid
                1, 1, 1,  # block
                0,         # shared mem
                None,      # stream
                None,      # kernel params
                None,      # extra
            ),
        )

        # cuCtxSynchronize
        _check("cuCtxSynchronize", libcuda.cuCtxSynchronize())

    finally:
        # cuCtxDestroy
        libcuda.cuCtxDestroy_v2(ctx)


def compile_and_run(
    mlir_text: str,
    gpu_arch: str | None = None,
    cuda_tile_translate_path: str | None = None,
    compile_only: bool = False,
    output_dir: Path | None = None,
) -> Path | None:
    """Full pipeline: MLIR text -> .tilebc -> .cubin -> optional GPU launch.

    Returns the .cubin path if compile_only, else None.
    """
    tools = find_tools(cuda_tile_translate_path)

    if gpu_arch is None:
        gpu_arch = detect_gpu_arch()

    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="cutile_basic_"))

    tilebc_path = output_dir / "program.tilebc"
    cubin_path = output_dir / "program.cubin"

    print(f"[1/3] Compiling MLIR -> .tilebc ...", flush=True)
    compile_mlir_to_tilebc(mlir_text, tilebc_path, tools)

    print(f"[2/3] Assembling .tilebc -> .cubin (arch={gpu_arch}) ...", flush=True)
    compile_tilebc_to_cubin(tilebc_path, cubin_path, gpu_arch, tools)

    if compile_only:
        print(f"[done] Compiled to {cubin_path}")
        return cubin_path

    print(f"[3/3] Launching kernel on GPU ...", flush=True)
    launch_cubin(cubin_path)
    print(f"[done] Kernel execution complete.")
    return None

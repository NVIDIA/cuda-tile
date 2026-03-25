"""GPU compilation and launch pipeline: .mlir -> .tilebc -> .cubin -> GPU launch."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from cuda.core import Device, LaunchConfig, ObjectCode, launch

from .gpu import detect_gpu_arch


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
                "cuda-tile-translate not found. "
                "Build it from https://github.com/NVIDIA/cuda-tile "
                "or pass --cuda-tile-translate <path>."
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
    try:
        dev = Device(0)
        dev.set_current()
        stream = dev.create_stream()

        kernel = ObjectCode.from_cubin(str(cubin_path)).get_kernel("main")
        config = LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1))
        launch(stream, config, kernel)
        stream.sync()
    except Exception as e:
        raise RunnerError(f"GPU launch failed: {e}") from e
    print(f"[done] Kernel execution complete.")
    return None

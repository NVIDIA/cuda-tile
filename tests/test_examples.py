"""Tests for the public API using example .bas files."""

import subprocess
import sys
from pathlib import Path

import pytest
from cutile_basic import compile_basic_to_mlir


EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _read_example(name: str) -> str:
    return (EXAMPLES_DIR / name).read_text()


class TestCompileBasicToMlir:
    """Test compile_basic_to_mlir with all example programs."""

    def test_hello(self):
        mlir = compile_basic_to_mlir(_read_example("hello.bas"))
        assert "cuda_tile.module @basic_program" in mlir
        assert "entry @main()" in mlir
        assert "Hello, World!" in mlir
        assert "return" in mlir

    def test_vector_add(self):
        mlir = compile_basic_to_mlir(_read_example("vector_add.bas"))
        assert "cuda_tile.module @basic_program" in mlir
        assert "entry @main(" in mlir
        assert "return" in mlir

    def test_gemm(self):
        mlir = compile_basic_to_mlir(_read_example("gemm.bas"))
        assert "cuda_tile.module @basic_program" in mlir
        assert "entry @main(" in mlir
        assert "for %" in mlir
        assert "return" in mlir

    def test_return_type_is_str(self):
        result = compile_basic_to_mlir(_read_example("hello.bas"))
        assert isinstance(result, str)

    def test_all_examples_compile(self):
        """Every .bas file in examples/ should compile without error."""
        for bas_file in sorted(EXAMPLES_DIR.glob("*.bas")):
            source = bas_file.read_text()
            mlir = compile_basic_to_mlir(source)
            assert "cuda_tile.module @basic_program" in mlir, (
                f"{bas_file.name} did not produce valid MLIR"
            )


def _run_example(script_name: str, timeout: int = 120) -> subprocess.CompletedProcess:
    """Run an example .py script as a subprocess and return the result."""
    script = EXAMPLES_DIR / script_name
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return result


class TestExampleScripts:
    """End-to-end tests that run the example .py scripts on the GPU."""

    def test_hello(self):
        result = _run_example("hello.py")
        assert result.returncode == 0, (
            f"hello.py failed (rc={result.returncode}):\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert "Kernel execution complete" in result.stdout

    def test_vector_add(self):
        result = _run_example("vector_add.py")
        assert result.returncode == 0, (
            f"vector_add.py failed (rc={result.returncode}):\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert "VERIFICATION PASSED" in result.stdout

    def test_gemm(self):
        result = _run_example("gemm.py")
        assert result.returncode == 0, (
            f"gemm.py failed (rc={result.returncode}):\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert "VERIFICATION PASSED" in result.stdout

    def test_all_example_scripts_run(self):
        """Every .py file in examples/ should run successfully."""
        for py_file in sorted(EXAMPLES_DIR.glob("*.py")):
            result = _run_example(py_file.name)
            assert result.returncode == 0, (
                f"{py_file.name} failed (rc={result.returncode}):\n"
                f"stdout: {result.stdout}\nstderr: {result.stderr}"
            )

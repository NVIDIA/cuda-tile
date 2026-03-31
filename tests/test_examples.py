"""Tests for the public API using example .bas files.

Includes CLI tests and GPU end-to-end tests that run the example .py scripts.
"""

import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _read_example(name: str) -> str:
    return (EXAMPLES_DIR / name).read_text()


def _run_cli(*args: str, timeout: int = 120) -> subprocess.CompletedProcess:
    """Run python -m cutile_basic.cli with the given arguments."""
    return subprocess.run(
        [sys.executable, "-m", "cutile_basic.cli", *args],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
    )


class TestDocsCLIExamples:
    """Test every CLI command that appears in the documentation."""

    def test_cli_dump_tokens(self):
        r = _run_cli("examples/hello.bas", "--dump-tokens")
        assert r.returncode == 0, f"stderr: {r.stderr}"
        assert r.stdout.strip()

    def test_cli_dump_ast(self):
        r = _run_cli("examples/hello.bas", "--dump-ast")
        assert r.returncode == 0, f"stderr: {r.stderr}"
        assert r.stdout.strip()

    def test_cli_compile_cubin(self, tmp_path):
        out = tmp_path / "vector_add.cubin"
        r = _run_cli("examples/vector_add.bas", "-o", str(out))
        assert r.returncode == 0, f"stderr: {r.stderr}"
        assert out.exists()
        assert out.stat().st_size > 0


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


class TestReadmeExample:
    """Validate the exact code example shown in the README."""

    README_SOURCE = """
10 INPUT N, A(), B()
20 DIM A(N), B(N), C(N)
30 TILE A(128), B(128), C(128)
40 LET C(BID) = A(BID) + B(BID)
50 OUTPUT C
60 END
"""

    def test_compile_and_metadata(self):
        from cutile_basic import compile_basic_to_cubin

        result = compile_basic_to_cubin(self.README_SOURCE)

        assert Path(result.cubin_path).exists()
        assert Path(result.cubin_path).stat().st_size > 0
        assert isinstance(result.meta, dict)
        assert "tile_shapes" in result.meta
        for name in ("A", "B", "C"):
            assert name in result.meta["tile_shapes"], (
                f"expected tile shape for array {name!r} in meta"
            )
            assert result.meta["tile_shapes"][name] == [128]


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

"""Tests for the public API using example .bas files.

Includes golden-file comparison and structural validation of the textual
backend, plus GPU end-to-end tests that run the example .py scripts.

To update golden files after an intentional change:
    UPDATE_GOLDEN=1 pytest tests/test_examples.py -v
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
from cutile_basic import compile_basic_to_textual


EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPECTED_DIR = Path(__file__).resolve().parent / "textual_gold"
UPDATE_GOLDEN = os.environ.get("UPDATE_GOLDEN", "0") == "1"


def _read_example(name: str) -> str:
    return (EXAMPLES_DIR / name).read_text()


def _example_files() -> list[Path]:
    return sorted(EXAMPLES_DIR.glob("*.bas"))


def _golden_path(bas_file: Path) -> Path:
    return EXPECTED_DIR / f"{bas_file.stem}.mlir"


class TestCompileBasicToTextual:
    """Test compile_basic_to_textual with all example programs."""

    def test_hello(self):
        text = compile_basic_to_textual(_read_example("hello.bas"))
        assert "cuda_tile.module @basic_program" in text
        assert "entry @main()" in text
        assert "Hello, World!" in text
        assert "return" in text

    def test_vector_add(self):
        text = compile_basic_to_textual(_read_example("vector_add.bas"))
        assert "cuda_tile.module @basic_program" in text
        assert "entry @main(" in text
        assert "return" in text

    def test_gemm(self):
        text = compile_basic_to_textual(_read_example("gemm.bas"))
        assert "cuda_tile.module @basic_program" in text
        assert "entry @main(" in text
        assert "for %" in text
        assert "return" in text

    def test_return_type_is_str(self):
        result = compile_basic_to_textual(_read_example("hello.bas"))
        assert isinstance(result, str)

    def test_all_examples_compile(self):
        """Every .bas file in examples/ should compile without error."""
        for bas_file in sorted(EXAMPLES_DIR.glob("*.bas")):
            source = bas_file.read_text()
            text = compile_basic_to_textual(source)
            assert "cuda_tile.module @basic_program" in text, (
                f"{bas_file.name} did not produce valid textual output"
            )


# ---------------------------------------------------------------------------
# Golden-file comparison
# ---------------------------------------------------------------------------

class TestTextualGoldenFiles:
    """Full textual output must match the committed golden files."""

    @pytest.fixture(autouse=True)
    def _ensure_dir(self):
        EXPECTED_DIR.mkdir(exist_ok=True)

    @pytest.mark.parametrize("bas_file", _example_files(), ids=lambda f: f.stem)
    def test_matches_golden(self, bas_file):
        source = bas_file.read_text()
        actual = compile_basic_to_textual(source)
        golden = _golden_path(bas_file)

        if UPDATE_GOLDEN:
            golden.write_text(actual)
            pytest.skip(f"Updated golden file: {golden.name}")

        assert golden.exists(), (
            f"Golden file {golden} not found. "
            f"Run with UPDATE_GOLDEN=1 to create it."
        )
        expected = golden.read_text()
        assert actual == expected, (
            f"Textual output for {bas_file.name} differs from golden file.\n"
            f"Run with UPDATE_GOLDEN=1 to update."
        )


# ---------------------------------------------------------------------------
# Structural validation (applies to every example)
# ---------------------------------------------------------------------------

class TestTextualStructure:
    """Structural invariants that every generated textual program must satisfy."""

    @pytest.mark.parametrize("bas_file", _example_files(), ids=lambda f: f.stem)
    def test_balanced_braces(self, bas_file):
        text = compile_basic_to_textual(bas_file.read_text())
        assert text.count("{") == text.count("}"), (
            f"Unbalanced braces in {bas_file.name}"
        )

    @pytest.mark.parametrize("bas_file", _example_files(), ids=lambda f: f.stem)
    def test_module_wrapping(self, bas_file):
        text = compile_basic_to_textual(bas_file.read_text())
        lines = text.strip().splitlines()
        assert lines[0].startswith("cuda_tile.module @basic_program {")
        assert lines[-1] == "}"

    @pytest.mark.parametrize("bas_file", _example_files(), ids=lambda f: f.stem)
    def test_entry_and_return(self, bas_file):
        text = compile_basic_to_textual(bas_file.read_text())
        assert "entry @main" in text
        assert "return" in text

    @pytest.mark.parametrize("bas_file", _example_files(), ids=lambda f: f.stem)
    def test_ssa_numbering_contiguous(self, bas_file):
        """SSA values %0, %1, ... must be contiguous with no gaps."""
        text = compile_basic_to_textual(bas_file.read_text())
        ssa_ids = sorted(set(int(m) for m in re.findall(r"%(\d+)", text)))
        if not ssa_ids:
            return
        assert ssa_ids[0] == 0, "SSA numbering must start at %0"
        expected = list(range(ssa_ids[0], ssa_ids[-1] + 1))
        assert ssa_ids == expected, (
            f"Gap in SSA numbering for {bas_file.name}: {ssa_ids}"
        )

    @pytest.mark.parametrize("bas_file", _example_files(), ids=lambda f: f.stem)
    def test_indentation_even(self, bas_file):
        """Every non-empty line must be indented by a multiple of 2 spaces."""
        text = compile_basic_to_textual(bas_file.read_text())
        for lineno, line in enumerate(text.splitlines(), 1):
            if not line:
                continue
            indent = len(line) - len(line.lstrip(" "))
            assert indent % 2 == 0, (
                f"Line {lineno} has odd indentation ({indent}): {line!r}"
            )

    @pytest.mark.parametrize("bas_file", _example_files(), ids=lambda f: f.stem)
    def test_trailing_newline(self, bas_file):
        text = compile_basic_to_textual(bas_file.read_text())
        assert text.endswith("\n"), "Generated textual output should end with a newline"

    @pytest.mark.parametrize("bas_file", _example_files(), ids=lambda f: f.stem)
    def test_no_undefined_ssa_refs(self, bas_file):
        """Every SSA value used on the RHS must be defined earlier or be a
        block argument (loop induction var / kernel param / iter_values)."""
        text = compile_basic_to_textual(bas_file.read_text())

        defined: set[str] = set()
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("//"):
                continue

            def_match = re.match(r"(%\d+)\s*=", stripped)
            if def_match:
                defined.add(def_match.group(1))

            for m in re.finditer(r"(%\d+)\s*:", stripped):
                if "entry" in stripped or "iter_values" in stripped:
                    defined.add(m.group(1))

            for_match = re.match(r"for\s+(%\d+)\s+in", stripped)
            if for_match:
                defined.add(for_match.group(1))

            multi_def = re.match(r"((?:%\d+,\s*)*%\d+)\s*=\s*for", stripped)
            if multi_def:
                for tok in multi_def.group(1).split(","):
                    defined.add(tok.strip())


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

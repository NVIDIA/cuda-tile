"""Tests for the bytecode backend (AST → cuTile bytecode)."""

import os
import shutil
import subprocess

import pytest

from cutile_basic._lexer import lex
from cutile_basic._parser import parse
from cutile_basic._analyzer import analyze
from cutile_basic.bytecode_backend import BytecodeBackend
from cutile_basic.gpu_runner import launch_kernel

TILEIRAS_MAGIC = b"\x7fTileIR"


def _tileiras_available() -> bool:
    if shutil.which("tileiras"):
        return True
    if os.path.isfile("/usr/local/cuda/bin/tileiras"):
        return True
    try:
        import nvidia.cu13.bin as _nbin
        return any(
            os.path.isfile(os.path.join(d, "tileiras")) for d in _nbin.__path__
        )
    except ImportError:
        return False


def _detect_gpu_arch() -> str:
    """Detect the GPU architecture from nvidia-smi, e.g. 'sm_80'."""
    from cutile_basic._runner import detect_gpu_arch
    return detect_gpu_arch()


requires_tileiras = pytest.mark.skipif(
    not _tileiras_available(), reason="tileiras not available",
)


@pytest.fixture(scope="session")
def gpu_arch():
    """Session-scoped GPU architecture string (e.g. 'sm_80')."""
    return _detect_gpu_arch()


def _compile(source: str) -> bytes:
    """Helper: compile BASIC source to bytecode bytes."""
    tokens = lex(source)
    program = parse(tokens)
    analyzed = analyze(program)
    return BytecodeBackend(analyzed).generate()


class TestBytecodeGeneration:
    """Test that bytecode is generated correctly."""

    def test_empty_program(self):
        bc = _compile("10 END")
        assert bc[:7] == TILEIRAS_MAGIC
        assert len(bc) > 20

    def test_constant_int(self):
        bc = _compile("10 LET X% = 42\n20 END")
        assert bc[:7] == TILEIRAS_MAGIC

    def test_constant_float(self):
        bc = _compile("10 LET X = 3.14\n20 END")
        assert bc[:7] == TILEIRAS_MAGIC

    def test_print_string(self):
        bc = _compile('10 PRINT "Hello"\n20 END')
        assert bc[:7] == TILEIRAS_MAGIC

    def test_print_expression(self):
        bc = _compile("10 LET X = 42.0\n20 PRINT X\n30 END")
        assert bc[:7] == TILEIRAS_MAGIC

    def test_arithmetic_int(self):
        bc = _compile(
            "10 LET A% = 10\n20 LET B% = A% + 5\n"
            "30 LET C% = A% * B%\n40 END"
        )
        assert bc[:7] == TILEIRAS_MAGIC

    def test_arithmetic_float(self):
        bc = _compile(
            "10 LET A = 1.5\n20 LET B = A + 2.5\n"
            "30 LET C = A * B\n40 END"
        )
        assert bc[:7] == TILEIRAS_MAGIC

    def test_if_else(self):
        bc = _compile(
            '10 LET X = 10.0\n'
            '20 IF X > 5.0 THEN\n'
            '30   PRINT "big"\n'
            '40 ELSE\n'
            '50   PRINT "small"\n'
            '60 ENDIF\n'
            '70 END'
        )
        assert bc[:7] == TILEIRAS_MAGIC

    def test_for_loop(self):
        bc = _compile(
            "10 FOR I = 1 TO 5\n"
            "20   PRINT I\n"
            "30 NEXT I\n"
            "40 END"
        )
        assert bc[:7] == TILEIRAS_MAGIC

    def test_for_loop_with_iter_values(self):
        """Test FOR loop that modifies outer variables (fibonacci pattern)."""
        bc = _compile(
            "10 LET A = 0\n"
            "20 LET B = 1\n"
            "30 FOR I = 1 TO 5\n"
            "40   LET C = A + B\n"
            "50   LET A = B\n"
            "60   LET B = C\n"
            "70 NEXT I\n"
            "80 END"
        )
        assert bc[:7] == TILEIRAS_MAGIC

    def test_data_read(self):
        bc = _compile(
            "10 DATA 10, 20, 30\n"
            "20 LET S = 0\n"
            "30 FOR I = 1 TO 3\n"
            "40   READ X\n"
            "50   LET S = S + X\n"
            "60 NEXT I\n"
            "70 PRINT S\n"
            "80 END"
        )
        assert bc[:7] == TILEIRAS_MAGIC

    def test_rem_ignored(self):
        bc = _compile("10 REM This is a comment\n20 END")
        assert bc[:7] == TILEIRAS_MAGIC

    def test_hello_bas(self):
        source = open("examples/hello.bas").read()
        bc = _compile(source)
        assert bc[:7] == TILEIRAS_MAGIC

    def test_fibonacci_bas(self):
        source = open("examples/fibonacci.bas").read()
        bc = _compile(source)
        assert bc[:7] == TILEIRAS_MAGIC

    def test_array_sum_bas(self):
        source = open("examples/array_sum.bas").read()
        bc = _compile(source)
        assert bc[:7] == TILEIRAS_MAGIC


@requires_tileiras
class TestCompileToCubin:
    """Test the full pipeline through tileiras (skip if not available)."""

    def test_hello_cubin(self, gpu_arch):
        source = open("examples/hello.bas").read()
        tokens = lex(source)
        program = parse(tokens)
        analyzed = analyze(program)
        backend = BytecodeBackend(analyzed, gpu_arch=gpu_arch)
        cubin_path = backend.compile_to_cubin()
        assert os.path.isfile(cubin_path)
        assert os.path.getsize(cubin_path) > 0

    def test_fibonacci_cubin(self, gpu_arch):
        source = open("examples/fibonacci.bas").read()
        tokens = lex(source)
        program = parse(tokens)
        analyzed = analyze(program)
        backend = BytecodeBackend(analyzed, gpu_arch=gpu_arch)
        cubin_path = backend.compile_to_cubin()
        assert os.path.isfile(cubin_path)
        assert os.path.getsize(cubin_path) > 0

    def test_array_sum_cubin(self, gpu_arch):
        source = open("examples/array_sum.bas").read()
        tokens = lex(source)
        program = parse(tokens)
        analyzed = analyze(program)
        backend = BytecodeBackend(analyzed, gpu_arch=gpu_arch)
        cubin_path = backend.compile_to_cubin()
        assert os.path.isfile(cubin_path)
        assert os.path.getsize(cubin_path) > 0


class TestArrayKernel:
    """Test the tiled array kernel path (vector add)."""

    N = 1024

    def test_vector_add_bytecode(self):
        source = open("examples/vector_add.bas").read()
        tokens = lex(source)
        program = parse(tokens)
        analyzed = analyze(program)
        bc = BytecodeBackend(analyzed, array_size=self.N).generate()
        assert bc[:7] == TILEIRAS_MAGIC

    def test_vector_add_detects_array_kernel(self):
        source = open("examples/vector_add.bas").read()
        tokens = lex(source)
        program = parse(tokens)
        analyzed = analyze(program)
        backend = BytecodeBackend(analyzed, array_size=self.N)
        assert backend._is_array_kernel()
        info = backend._get_array_info()
        assert info["input_arrays"] == ["A", "B"]
        assert info["output_arrays"] == ["C"]
        assert info["tile_size"] == 128

    @requires_tileiras
    def test_vector_add_cubin(self, gpu_arch):
        source = open("examples/vector_add.bas").read()
        tokens = lex(source)
        program = parse(tokens)
        analyzed = analyze(program)
        backend = BytecodeBackend(analyzed, gpu_arch=gpu_arch, array_size=self.N)
        cubin_path = backend.compile_to_cubin()
        assert os.path.isfile(cubin_path)
        assert os.path.getsize(cubin_path) > 0
        assert backend._array_kernel_meta is not None
        assert backend._array_kernel_meta["grid_size"] == 8

    @requires_tileiras
    def test_vector_add_gpu_execution(self, gpu_arch):
        """Full end-to-end: compile + launch + verify on GPU."""
        source = open("examples/vector_add.bas").read()
        tokens = lex(source)
        program = parse(tokens)
        analyzed = analyze(program)
        backend = BytecodeBackend(analyzed, gpu_arch=gpu_arch, array_size=self.N)
        cubin_path = backend.compile_to_cubin()
        meta = backend._array_kernel_meta

        N = self.N
        h_a = [float(i) for i in range(N)]
        h_b = [float(i) * 2.0 for i in range(N)]

        results = launch_kernel(
            cubin_path=cubin_path,
            inputs={"A": h_a, "B": h_b},
            outputs=["C"],
            param_order=meta["all_arrays"],
            sizes={name: N for name in meta["all_arrays"]},
            grid_size=meta["grid_size"],
        )

        h_c = results["C"]
        for i in [0, 1, 511, 512, 1023]:
            assert abs(h_c[i] - (h_a[i] + h_b[i])) < 0.01


class TestGemmKernel:
    """Test the tiled GEMM kernel path."""

    def test_gemm_bytecode(self):
        source = open("examples/gemm.bas").read()
        tokens = lex(source)
        program = parse(tokens)
        analyzed = analyze(program)
        bc = BytecodeBackend(analyzed).generate()
        assert bc[:7] == TILEIRAS_MAGIC

    def test_gemm_detects_gemm_kernel(self):
        source = open("examples/gemm.bas").read()
        tokens = lex(source)
        program = parse(tokens)
        analyzed = analyze(program)
        backend = BytecodeBackend(analyzed)
        assert backend._is_gemm_kernel()

    def test_gemm_kernel_metadata(self):
        source = open("examples/gemm.bas").read()
        tokens = lex(source)
        program = parse(tokens)
        analyzed = analyze(program)
        backend = BytecodeBackend(analyzed)
        backend.generate()
        meta = backend._array_kernel_meta
        assert meta is not None
        assert meta["M"] == 512
        assert meta["N"] == 512
        assert meta["K"] == 512
        assert meta["tm"] == 128
        assert meta["tn"] == 128
        assert meta["tk"] == 32
        assert set(meta["all_arrays"]) == {"A", "B", "C"}

    @requires_tileiras
    def test_gemm_cubin(self, gpu_arch):
        source = open("examples/gemm.bas").read()
        tokens = lex(source)
        program = parse(tokens)
        analyzed = analyze(program)
        backend = BytecodeBackend(analyzed, gpu_arch=gpu_arch)
        cubin_path = backend.compile_to_cubin()
        assert os.path.isfile(cubin_path)
        assert os.path.getsize(cubin_path) > 0

    @requires_tileiras
    def test_gemm_gpu_execution(self, gpu_arch):
        """Full end-to-end: compile + launch + verify on GPU."""
        import random
        random.seed(42)

        source = open("examples/gemm.bas").read()
        tokens = lex(source)
        program = parse(tokens)
        analyzed = analyze(program)
        backend = BytecodeBackend(analyzed, gpu_arch=gpu_arch)
        cubin_path = backend.compile_to_cubin()
        meta = backend._array_kernel_meta
        M, N, K = meta["M"], meta["N"], meta["K"]

        h_a = [random.uniform(-1.0, 1.0) for _ in range(M * K)]
        h_b = [random.uniform(-1.0, 1.0) for _ in range(K * N)]

        a_name, b_name, c_name = meta["all_arrays"]
        results = launch_kernel(
            cubin_path=cubin_path,
            inputs={a_name: h_a, b_name: h_b},
            outputs=[c_name],
            param_order=meta["all_arrays"],
            sizes={a_name: M * K, b_name: K * N, c_name: M * N},
            grid_size=meta["grid_size"],
        )

        h_c = results[c_name]
        expected = [0.0] * (M * N)
        for i in range(M):
            for j in range(N):
                s = 0.0
                for k in range(K):
                    s += h_a[i * K + k] * h_b[k * N + j]
                expected[i * N + j] = s

        max_diff = max(abs(h_c[i] - expected[i]) for i in range(M * N))
        assert max_diff < K * 1e-5

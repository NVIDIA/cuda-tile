"""Tests for the GPU compilation and launch pipeline."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import pytest

from cutile_basic.runner import (
    RunnerError,
    find_tools,
    compile_mlir_to_tilebc,
    compile_tilebc_to_cubin,
    detect_gpu_arch,
    launch_cubin,
    compile_and_run,
)


# --- find_tools ---


class TestFindTools:
    def test_explicit_cuda_tile_translate(self, tmp_path):
        fake_bin = tmp_path / "cuda-tile-translate"
        fake_bin.write_text("#!/bin/sh\n")
        fake_bin.chmod(0o755)

        tileiras = tmp_path / "tileiras"
        tileiras.write_text("#!/bin/sh\n")
        tileiras.chmod(0o755)

        with patch("cutile_basic.runner._search_paths") as mock_search:
            # Only mock tileiras search; cuda-tile-translate uses explicit path
            def side_effect(name):
                if name == "tileiras":
                    return [tileiras]
                return []
            mock_search.side_effect = side_effect

            tools = find_tools(cuda_tile_translate_path=str(fake_bin))
            assert tools["cuda-tile-translate"] == fake_bin
            assert tools["tileiras"] == tileiras

    def test_missing_cuda_tile_translate_raises(self):
        with patch("cutile_basic.runner._search_paths", return_value=[]):
            with pytest.raises(RunnerError, match="cuda-tile-translate not found"):
                find_tools()

    def test_explicit_path_not_executable(self, tmp_path):
        fake_bin = tmp_path / "cuda-tile-translate"
        fake_bin.write_text("not executable")
        fake_bin.chmod(0o644)

        with pytest.raises(RunnerError, match="not found or not executable"):
            find_tools(cuda_tile_translate_path=str(fake_bin))

    def test_missing_tileiras_raises(self, tmp_path):
        fake_ctt = tmp_path / "cuda-tile-translate"
        fake_ctt.write_text("#!/bin/sh\n")
        fake_ctt.chmod(0o755)

        with patch("cutile_basic.runner._search_paths", return_value=[]):
            with pytest.raises(RunnerError, match="tileiras not found"):
                find_tools(cuda_tile_translate_path=str(fake_ctt))


# --- compile_mlir_to_tilebc ---


class TestCompileMlirToTilebc:
    def test_success(self, tmp_path):
        output = tmp_path / "out.tilebc"
        tools = {"cuda-tile-translate": Path("/usr/bin/fake-ctt")}

        def fake_run(cmd, **kwargs):
            # Simulate creating the output file
            output.write_bytes(b"tilebc-data")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        with patch("cutile_basic.runner.subprocess.run", side_effect=fake_run):
            result = compile_mlir_to_tilebc("mlir text", output, tools)

        assert result == output
        assert output.exists()

    def test_failure_nonzero_exit(self, tmp_path):
        output = tmp_path / "out.tilebc"
        tools = {"cuda-tile-translate": Path("/usr/bin/fake-ctt")}

        with patch("cutile_basic.runner.subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                [], 1, stdout="", stderr="some error"
            )
            with pytest.raises(RunnerError, match="cuda-tile-translate failed"):
                compile_mlir_to_tilebc("mlir text", output, tools)


# --- compile_tilebc_to_cubin ---


class TestCompileTilebcToCubin:
    def test_success(self, tmp_path):
        tilebc = tmp_path / "in.tilebc"
        tilebc.write_bytes(b"data")
        output = tmp_path / "out.cubin"
        tools = {"tileiras": Path("/usr/local/cuda/bin/tileiras")}

        def fake_run(cmd, **kwargs):
            output.write_bytes(b"cubin-data")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        with patch("cutile_basic.runner.subprocess.run", side_effect=fake_run):
            result = compile_tilebc_to_cubin(tilebc, output, "sm_120", tools)

        assert result == output

    def test_failure(self, tmp_path):
        tilebc = tmp_path / "in.tilebc"
        tilebc.write_bytes(b"data")
        output = tmp_path / "out.cubin"
        tools = {"tileiras": Path("/usr/local/cuda/bin/tileiras")}

        with patch("cutile_basic.runner.subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                [], 1, stdout="", stderr="asm error"
            )
            with pytest.raises(RunnerError, match="tileiras failed"):
                compile_tilebc_to_cubin(tilebc, output, "sm_120", tools)


# --- detect_gpu_arch ---


class TestDetectGpuArch:
    def test_parses_compute_cap(self):
        with patch("cutile_basic.runner.subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                [], 0, stdout="12.0\n", stderr=""
            )
            assert detect_gpu_arch() == "sm_120"

    def test_older_gpu(self):
        with patch("cutile_basic.runner.subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                [], 0, stdout="8.9\n", stderr=""
            )
            assert detect_gpu_arch() == "sm_89"

    def test_nvidia_smi_missing(self):
        with patch(
            "cutile_basic.runner.subprocess.run", side_effect=FileNotFoundError
        ):
            with pytest.raises(RunnerError, match="nvidia-smi not found"):
                detect_gpu_arch()

    def test_bad_format(self):
        with patch("cutile_basic.runner.subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                [], 0, stdout="unknown\n", stderr=""
            )
            with pytest.raises(RunnerError, match="Unexpected compute_cap"):
                detect_gpu_arch()


# --- launch_cubin ---


class TestLaunchCubin:
    def test_driver_api_sequence(self, tmp_path):
        cubin = tmp_path / "test.cubin"
        cubin.write_bytes(b"\x00" * 64)

        mock_libcuda = MagicMock()
        # All calls return 0 (CUDA_SUCCESS)
        mock_libcuda.cuInit.return_value = 0
        mock_libcuda.cuDeviceGet.return_value = 0
        mock_libcuda.cuCtxCreate_v2.return_value = 0
        mock_libcuda.cuModuleLoadData.return_value = 0
        mock_libcuda.cuModuleGetFunction.return_value = 0
        mock_libcuda.cuLaunchKernel.return_value = 0
        mock_libcuda.cuCtxSynchronize.return_value = 0
        mock_libcuda.cuCtxDestroy_v2.return_value = 0

        with patch("cutile_basic.runner.ctypes.CDLL", return_value=mock_libcuda):
            launch_cubin(cubin)

        mock_libcuda.cuInit.assert_called_once_with(0)
        mock_libcuda.cuLaunchKernel.assert_called_once()
        mock_libcuda.cuCtxSynchronize.assert_called_once()
        mock_libcuda.cuCtxDestroy_v2.assert_called_once()

    def test_driver_error(self, tmp_path):
        cubin = tmp_path / "test.cubin"
        cubin.write_bytes(b"\x00" * 64)

        mock_libcuda = MagicMock()
        mock_libcuda.cuInit.return_value = 100  # CUDA_ERROR_NO_DEVICE

        with patch("cutile_basic.runner.ctypes.CDLL", return_value=mock_libcuda):
            with pytest.raises(RunnerError, match="CUDA driver error in cuInit"):
                launch_cubin(cubin)

    def test_libcuda_not_found(self, tmp_path):
        cubin = tmp_path / "test.cubin"
        cubin.write_bytes(b"\x00" * 64)

        with patch("cutile_basic.runner.ctypes.CDLL", side_effect=OSError("not found")):
            with pytest.raises(RunnerError, match="Cannot load libcuda.so"):
                launch_cubin(cubin)


# --- compile_and_run (integration with mocks) ---


class TestCompileAndRun:
    def test_compile_only(self, tmp_path):
        tools = {
            "cuda-tile-translate": Path("/fake/cuda-tile-translate"),
            "tileiras": Path("/fake/tileiras"),
        }

        call_count = 0

        def fake_run(cmd, **kwargs):
            nonlocal call_count
            call_count += 1
            # Create expected output files
            for i, arg in enumerate(cmd):
                if arg == "-o" and i + 1 < len(cmd):
                    Path(cmd[i + 1]).write_bytes(b"data")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        with (
            patch("cutile_basic.runner.find_tools", return_value=tools),
            patch("cutile_basic.runner.detect_gpu_arch", return_value="sm_120"),
            patch("cutile_basic.runner.subprocess.run", side_effect=fake_run),
        ):
            result = compile_and_run(
                "mlir text", compile_only=True, output_dir=tmp_path
            )

        assert result is not None
        assert result.suffix == ".cubin"
        assert call_count == 2  # cuda-tile-translate + tileiras

    def test_full_run(self, tmp_path):
        tools = {
            "cuda-tile-translate": Path("/fake/cuda-tile-translate"),
            "tileiras": Path("/fake/tileiras"),
        }

        def fake_run(cmd, **kwargs):
            for i, arg in enumerate(cmd):
                if arg == "-o" and i + 1 < len(cmd):
                    Path(cmd[i + 1]).write_bytes(b"data")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        with (
            patch("cutile_basic.runner.find_tools", return_value=tools),
            patch("cutile_basic.runner.detect_gpu_arch", return_value="sm_120"),
            patch("cutile_basic.runner.subprocess.run", side_effect=fake_run),
            patch("cutile_basic.runner.launch_cubin") as mock_launch,
        ):
            result = compile_and_run(
                "mlir text", compile_only=False, output_dir=tmp_path
            )

        assert result is None
        mock_launch.assert_called_once()

"""Tests for the GPU compilation and launch pipeline."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cutile_basic.gpu import detect_gpu_arch
from cutile_basic._runner import (
    RunnerError,
    find_tools,
    compile_mlir_to_tilebc,
    compile_tilebc_to_cubin,
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

        with patch("cutile_basic._runner._search_paths") as mock_search:
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
        with patch("cutile_basic._runner._search_paths", return_value=[]):
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

        with patch("cutile_basic._runner._search_paths", return_value=[]):
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

        with patch("cutile_basic._runner.subprocess.run", side_effect=fake_run):
            result = compile_mlir_to_tilebc("mlir text", output, tools)

        assert result == output
        assert output.exists()

    def test_failure_nonzero_exit(self, tmp_path):
        output = tmp_path / "out.tilebc"
        tools = {"cuda-tile-translate": Path("/usr/bin/fake-ctt")}

        with patch("cutile_basic._runner.subprocess.run") as mock_run:
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

        with patch("cutile_basic._runner.subprocess.run", side_effect=fake_run):
            result = compile_tilebc_to_cubin(tilebc, output, "sm_120", tools)

        assert result == output

    def test_failure(self, tmp_path):
        tilebc = tmp_path / "in.tilebc"
        tilebc.write_bytes(b"data")
        output = tmp_path / "out.cubin"
        tools = {"tileiras": Path("/usr/local/cuda/bin/tileiras")}

        with patch("cutile_basic._runner.subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                [], 1, stdout="", stderr="asm error"
            )
            with pytest.raises(RunnerError, match="tileiras failed"):
                compile_tilebc_to_cubin(tilebc, output, "sm_120", tools)


# --- detect_gpu_arch ---


class TestDetectGpuArch:
    def test_parses_compute_cap(self):
        mock_dev = MagicMock()
        mock_dev.compute_capability = (12, 0)
        with patch("cutile_basic.gpu.Device", return_value=mock_dev):
            assert detect_gpu_arch() == "sm_120"

    def test_older_gpu(self):
        mock_dev = MagicMock()
        mock_dev.compute_capability = (8, 9)
        with patch("cutile_basic.gpu.Device", return_value=mock_dev):
            assert detect_gpu_arch() == "sm_89"

    def test_device_error(self):
        with patch(
            "cutile_basic.gpu.Device",
            side_effect=RuntimeError("no device"),
        ):
            with pytest.raises(RuntimeError):
                detect_gpu_arch()


# --- launch_cubin ---


class TestLaunchCubin:
    def test_cuda_core_sequence(self, tmp_path):
        cubin = tmp_path / "test.cubin"
        cubin.write_bytes(b"\x00" * 64)

        mock_dev = MagicMock()
        mock_stream = MagicMock()
        mock_dev.create_stream.return_value = mock_stream
        mock_kernel = MagicMock()
        mock_obj = MagicMock()
        mock_obj.get_kernel.return_value = mock_kernel

        with (
            patch("cutile_basic._runner.Device", return_value=mock_dev),
            patch("cutile_basic._runner.ObjectCode") as mock_oc,
            patch("cutile_basic._runner.launch") as mock_launch,
        ):
            mock_oc.from_cubin.return_value = mock_obj
            launch_cubin(cubin)

        mock_dev.set_current.assert_called_once()
        mock_dev.create_stream.assert_called_once()
        mock_oc.from_cubin.assert_called_once_with(str(cubin))
        mock_obj.get_kernel.assert_called_once_with("main")
        mock_launch.assert_called_once()
        mock_stream.sync.assert_called_once()

    def test_device_error(self, tmp_path):
        cubin = tmp_path / "test.cubin"
        cubin.write_bytes(b"\x00" * 64)

        with patch(
            "cutile_basic._runner.Device",
            side_effect=RuntimeError("no device"),
        ):
            with pytest.raises(RunnerError, match="GPU launch failed"):
                launch_cubin(cubin)

    def test_kernel_load_error(self, tmp_path):
        cubin = tmp_path / "test.cubin"
        cubin.write_bytes(b"\x00" * 64)

        mock_dev = MagicMock()
        mock_oc = MagicMock()
        mock_oc.from_cubin.side_effect = RuntimeError("bad cubin")
        with (
            patch("cutile_basic._runner.Device", return_value=mock_dev),
            patch("cutile_basic._runner.ObjectCode", mock_oc),
        ):
            with pytest.raises(RunnerError, match="GPU launch failed"):
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
            patch("cutile_basic._runner.find_tools", return_value=tools),
            patch("cutile_basic._runner.detect_gpu_arch", return_value="sm_120"),
            patch("cutile_basic._runner.subprocess.run", side_effect=fake_run),
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
            patch("cutile_basic._runner.find_tools", return_value=tools),
            patch("cutile_basic._runner.detect_gpu_arch", return_value="sm_120"),
            patch("cutile_basic._runner.subprocess.run", side_effect=fake_run),
            patch("cutile_basic._runner.launch_cubin") as mock_launch,
        ):
            result = compile_and_run(
                "mlir text", compile_only=False, output_dir=tmp_path
            )

        assert result is None
        mock_launch.assert_called_once()

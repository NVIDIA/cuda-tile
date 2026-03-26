"""Tests for GPU architecture detection."""

from unittest.mock import MagicMock, patch

import pytest

from cutile_basic.gpu import detect_gpu_arch


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

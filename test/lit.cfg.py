# -*- Python -*-

import os

import lit.formats
from lit.llvm import llvm_config

# Configuration file for the 'lit' test runner

# name: The name of this test suite
config.name = "CUDA_TILE"

config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [".mlir", ".c", ".py"]

# excludes: A list of directories/files to exclude from the test suite.
config.excludes = ["lit.cfg.py", "lit.site.cfg.py", "round_trip_test.py"]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.cuda_tile_obj_root, "test")

config.substitutions.append(("%PATH%", config.environment["PATH"]))
config.substitutions.append(("%shlibext", config.llvm_shlib_ext))

capi_tests = ["test-cuda-tile-capi-register"]

llvm_config.add_tool_substitutions(capi_tests, [config.cuda_tile_tool_dir])

tool_dirs = [
    config.cuda_tile_tool_dir,
    config.llvm_tools_dir,
]

# Cross-platform round trip test script substitution
import platform

python_executable = config.python_executable
if platform.system() == "Windows":
    # On Windows, use Python to run the shared cross-platform script
    round_trip_script = (
        f'"{python_executable}" "{config.test_source_root}/round_trip_test.py"'
    )
else:
    # On Unix/Linux, use the shell script (fallback to shared location for consistency)
    round_trip_script = f"{config.test_source_root}/Dialect/CudaTile/round_trip_test.sh"

tools = [
    "cuda-tile-opt",
    "FileCheck",
    "not",
]

llvm_config.add_tool_substitutions(tools, tool_dirs)

# Add the round trip test substitution after the tools are set up
config.substitutions.append(("%round_trip_test", round_trip_script))

llvm_config.with_environment("PATH", config.cuda_tile_tool_dir, append_path=True)

# Python support for running Python tests
quoted_python_executable = (
    f'"{python_executable}"' if " " in python_executable else python_executable
)

# Python configuration with sanitizer requires preloading ASAN runtime on Linux.
# See: https://github.com/google/sanitizers/issues/1086
if config.llvm_use_sanitizer and "linux" in config.host_os.lower():

    def preload(lib_name: str) -> str:
        return f"$({config.host_cxx} -print-file-name={lib_name})"

    preload_libs = [
        preload("libclang_rt.asan.so" if "clang" in config.host_cxx else "libasan.so")
    ]
    preload_path = f'LD_PRELOAD="{" ".join(preload_libs)}"'
    quoted_python_executable = f"{preload_path} {quoted_python_executable}"

config.substitutions.append(("%PYTHON", quoted_python_executable))

# Add the python path for both the source and binary tree.
if config.enable_bindings_python:
    python_paths = [
        # Build directory (always needed for cuda_tile bindings)
        os.path.join(config.cuda_tile_obj_root, "python_packages"),
        # Test source python utilities
        os.path.join(config.test_source_root, "python"),
    ]
    # Also add install directory if available (CI pipelines)
    if config.cuda_tile_install_dir:
        python_paths.insert(
            0, os.path.join(config.cuda_tile_install_dir, "python_packages")
        )
    llvm_config.with_environment("PYTHONPATH", python_paths, append_path=True)

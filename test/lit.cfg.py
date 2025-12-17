# -*- Python -*-

import os

import lit.formats
from lit.llvm import llvm_config

# Configuration file for the 'lit' test runner

# name: The name of this test suite
config.name = "CUDA_TILE"

config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [".mlir", ".c"]

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
]

llvm_config.add_tool_substitutions(tools, tool_dirs)

# Add the round trip test substitution after the tools are set up
config.substitutions.append(("%round_trip_test", round_trip_script))

llvm_config.with_environment("PATH", config.cuda_tile_tool_dir, append_path=True)

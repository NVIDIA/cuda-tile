#!/usr/bin/env python3

"""
Cross-platform replacement for round_trip_test.sh
Tests MLIR -> CUDA Tile BC -> MLIR round-trip conversion
"""

import sys
import subprocess
import os
import difflib


def run_command(cmd, check=True):
    """Run a command and return the result"""
    try:
        result = subprocess.run(
            cmd, shell=True, check=check, capture_output=True, text=True
        )
        return result
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
        print(f"Return code: {e.returncode}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        raise


def main():
    if len(sys.argv) < 3:
        print("Usage: round_trip_test.py <input_file> <output_base> [extra_flags...]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_base = sys.argv[2]
    extra_flags = sys.argv[3:] if len(sys.argv) > 3 else []

    # Convert extra_flags list to space-separated string for shell commands
    extra_flags_str = " ".join(extra_flags) if extra_flags else ""

    try:
        # Step 1: Convert MLIR to CUDA Tile BC
        tilebc_file = f"{output_base}.out.tilebc"
        cmd1 = f"cuda-tile-translate -mlir-to-cudatilebc -no-implicit-module {input_file} -o {tilebc_file}"
        run_command(cmd1)

        # Step 2: Convert CUDA Tile BC back to MLIR
        roundtrip_file = f"{output_base}.roundtrip.mlir"
        cmd2 = f"cuda-tile-translate -cudatilebc-to-mlir {tilebc_file} -o {roundtrip_file} {extra_flags_str}".strip()
        run_command(cmd2)

        # Step 3: Create reference using cuda-tile-opt
        ref_file = f"{output_base}.ref.mlir"
        cmd3 = f"cuda-tile-opt {input_file} -no-implicit-module -o {ref_file} {extra_flags_str}".strip()
        run_command(cmd3)

        # Step 4: Compare files (equivalent to diff -B)
        with open(ref_file, "r") as f:
            ref_content = f.read()
        with open(roundtrip_file, "r") as f:
            roundtrip_content = f.read()

        # Remove blank lines for comparison (equivalent to diff -B)
        ref_lines = [line for line in ref_content.splitlines() if line.strip()]
        roundtrip_lines = [
            line for line in roundtrip_content.splitlines() if line.strip()
        ]

        if ref_lines != roundtrip_lines:
            print("Round-trip test failed: files differ")
            print("\nDifferences:")
            diff = difflib.unified_diff(
                ref_lines,
                roundtrip_lines,
                fromfile=ref_file,
                tofile=roundtrip_file,
                lineterm="",
            )
            for line in diff:
                print(line)
            sys.exit(1)

        print("Round-trip test passed")

    except Exception as e:
        print(f"Round-trip test failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

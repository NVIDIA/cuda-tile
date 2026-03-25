#!/usr/bin/env python3
"""End-to-end demo: compile a BASIC program and run it on the GPU."""

import sys
from pathlib import Path

from cuda.core import Device, LaunchConfig, ObjectCode, launch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cutile_basic import compile_basic_to_cubin


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <program.bas>")
        sys.exit(1)

    source = Path(sys.argv[1]).read_text()

    print("[1/2] Compiling to cubin ...", flush=True)
    result = compile_basic_to_cubin(source)

    print("[2/2] Launching kernel on GPU ...", flush=True)
    dev = Device(0)
    dev.set_current()
    stream = dev.create_stream()

    kernel = ObjectCode.from_cubin(result.cubin_path).get_kernel("main")
    config = LaunchConfig(grid=(1, 1, 1), block=(1, 1, 1))
    launch(stream, config, kernel)
    stream.sync()

    print("[done] Kernel execution complete.")


if __name__ == "__main__":
    main()

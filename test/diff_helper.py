#!/usr/bin/env python3

"""Small cross-platform diff helpers for lit tests."""

import difflib
import sys
from pathlib import Path


def nonblank_lines(path: Path) -> list[str]:
    return [line for line in path.read_text().splitlines() if line.strip()]


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: diff_helper.py <expected> <actual>", file=sys.stderr)
        return 2

    expected_path = Path(sys.argv[1])
    actual_path = Path(sys.argv[2])
    expected = nonblank_lines(expected_path)
    actual = nonblank_lines(actual_path)

    if expected == actual:
        return 0

    for line in difflib.unified_diff(
        expected,
        actual,
        fromfile=str(expected_path),
        tofile=str(actual_path),
        lineterm="",
    ):
        print(line)
    return 1


if __name__ == "__main__":
    sys.exit(main())

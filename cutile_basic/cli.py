"""CLI entry point for the BASIC to CUDA Tile IR compiler."""

import argparse
import shutil
import sys
from pathlib import Path

from .lexer import lex, LexError
from .parser import parse, ParseError
from .analyzer import analyze, AnalyzeError
from .bytecode import compile_basic_to_cubin, BytecodeBackendError


def main():
    parser = argparse.ArgumentParser(
        prog="cutile_basic",
        description="Compile BASIC source to CUDA Tile IR",
    )
    parser.add_argument("input", help="Input .bas file")
    parser.add_argument("-o", "--output", help="Output file path (default: stdout)")
    parser.add_argument("--dump-tokens", action="store_true", help="Dump tokens and exit")
    parser.add_argument("--dump-ast", action="store_true", help="Dump AST and exit")
    parser.add_argument("--dump-analyzed", action="store_true", help="Dump analyzed program and exit")

    parser.add_argument("--gpu-arch", default=None, help="GPU architecture (e.g. sm_120). Default: auto-detect")

    args = parser.parse_args()

    source = Path(args.input).read_text()

    try:
        tokens = lex(source)

        if args.dump_tokens:
            for tok in tokens:
                print(tok)
            return

        program = parse(tokens)

        if args.dump_ast:
            for i, stmt in enumerate(program.statements):
                print(f"[{i}] {stmt}")
            if program.line_map:
                print(f"\nLine map: {program.line_map}")
            return

        analyzed = analyze(program)

        if args.dump_analyzed:
            print("Symbols:")
            for name, info in sorted(analyzed.symbols.items()):
                arr = f" array[{info.array_size}]" if info.is_array else ""
                print(f"  {name}: {info.type.name}{arr}")
            if analyzed.data_values:
                print(f"\nDATA values: {analyzed.data_values}")
            if analyzed.input_vars:
                print(f"\nINPUT vars: {analyzed.input_vars}")
            if analyzed.output_vars:
                print(f"\nOUTPUT vars: {analyzed.output_vars}")
            print(f"\nHas GOTO: {analyzed.has_goto}")
            print(f"\nStatements: {len(analyzed.statements)}")
            return

        result = compile_basic_to_cubin(source, gpu_arch=args.gpu_arch)

        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            shutil.move(result.cubin_path, args.output)
            print(f"Wrote {args.output}", file=sys.stderr)
        else:
            print(result.cubin_path)

    except (LexError, ParseError, AnalyzeError, BytecodeBackendError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

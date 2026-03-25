"""CLI entry point for the BASIC to CUDA Tile IR compiler."""

import argparse
import sys
from pathlib import Path

from ._lexer import lex, LexError
from ._parser import parse, ParseError
from ._analyzer import analyze, AnalyzeError
from ._codegen import generate, CodeGenError
from ._runner import compile_and_run, RunnerError


def main():
    parser = argparse.ArgumentParser(
        prog="cutile_basic",
        description="Compile BASIC source to CUDA Tile IR MLIR",
    )
    parser.add_argument("input", help="Input .bas file")
    parser.add_argument("-o", "--output", help="Output .mlir file (default: stdout)")
    parser.add_argument("--dump-tokens", action="store_true", help="Dump tokens and exit")
    parser.add_argument("--dump-ast", action="store_true", help="Dump AST and exit")
    parser.add_argument("--dump-analyzed", action="store_true", help="Dump analyzed program and exit")
    parser.add_argument("--compile", action="store_true", help="Compile to .cubin (stop before GPU launch)")
    parser.add_argument("--run", action="store_true", help="Compile and launch kernel on GPU")
    parser.add_argument("--gpu-arch", default=None, help="GPU architecture override (e.g. sm_120). Default: auto-detect")
    parser.add_argument("--cuda-tile-translate", default=None, help="Path to cuda-tile-translate binary")

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
            print(f"\nHas GOTO: {analyzed.has_goto}")
            print(f"\nStatements: {len(analyzed.statements)}")
            return

        mlir = generate(analyzed)

        if args.compile or args.run:
            output_dir = None
            if args.output:
                output_dir = Path(args.output).parent
                output_dir.mkdir(parents=True, exist_ok=True)

            cubin_path = compile_and_run(
                mlir,
                gpu_arch=args.gpu_arch,
                cuda_tile_translate_path=args.cuda_tile_translate,
                compile_only=not args.run,
                output_dir=output_dir,
            )
            if cubin_path and args.output:
                # Move cubin to requested output path
                import shutil
                shutil.move(str(cubin_path), args.output)
                print(f"Wrote {args.output}", file=sys.stderr)
        elif args.output:
            Path(args.output).write_text(mlir)
            print(f"Wrote {args.output}", file=sys.stderr)
        else:
            print(mlir)

    except (LexError, ParseError, AnalyzeError, CodeGenError, RunnerError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

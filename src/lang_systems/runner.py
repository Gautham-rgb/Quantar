from . import ast_builder
from .runtime import runtime_globals
from pathlib import Path
import argparse
from ast import unparse
import sys

def qtr_runner(filename: str | Path, output_path: str | Path | None = None):
    path = Path(filename)

    if not path.exists():
        raise FileNotFoundError(f"Quantar source file {path} not found, either expand the scope or change the file name")

    if path.suffix not in (".qtr", ".py"):
        raise ValueError(f"Invalid file extension {path.suffix}. Expected .qtr, .py")

    source_code = path.read_text(encoding="utf-8")

    tree = ast_builder.parse_ast(source_code, str(filename))

    if output_path:
        python_source = unparse(ast_builder.lower_ast(tree))
        runtime_imports = (
            "from lang_systems.runtime import apply_quantar_operator as __quantar_apply__\n\n"
        )
        final_code = runtime_imports + python_source

        out_path = Path(output_path) if output_path else path.with_suffix(".py")
        out_path.write_text(final_code, "utf-8")
        print(f"[COMPILE] Compiled to {out_path}")

    else:
        python_ast = ast_builder.compile_ast(tree, str(filename))
        exec(python_ast, {"__name__": "__main__", **runtime_globals()})

def main():
    parser = argparse.ArgumentParser("qtr", description = "A CLI tool to run .qtr or .py files (Quantar files)")
    parser.add_argument("filename", help="The name of the file to process")
    parser.add_argument("-c", "--compile", help = "Compile the given file into a .py file at this location", metavar = "OUTPUT_PATH")

    args = parser.parse_args()

    try:
        qtr_runner(args.filename, args.compile)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)



if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
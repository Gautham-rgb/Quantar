from . import ast_builder
from .runtime import runtime_globals
from pathlib import Path
import argparse
from ast import unparse
import sys

INTERNAL_RUNNER_FILE = Path(__file__).resolve()
INTERNAL_PACKAGE_DIR = INTERNAL_RUNNER_FILE.parent.parent.resolve()

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
        runtime_imports = ("from lang_systems.runtime import apply_quantar_operator as __quantar_apply__\n\n")
        final_code = runtime_imports + python_source

        out_path = Path(output_path) if output_path else path.with_suffix(".py")
        out_path.write_text(final_code, "utf-8")
        print(f"[COMPILE] Compiled to {out_path}")

    else:
        python_ast = ast_builder.compile_ast(tree, str(filename))
        exec(python_ast, {"__name__": "__main__", **runtime_globals()})

def main():
    parser = argparse.ArgumentParser("qtr", description="A CLI tool to run .qtr or .py files (Quantar files)")
    parser.add_argument("filename", help="The name of the file to process")
    parser.add_argument("-c", "--compile", help="Compile the given file into a .py file at this location", metavar="OUTPUT_PATH")
    parser.add_argument("--full-traceback", action="store_true", help="Shows full error traceback (including the files for Quantar)")

    args = parser.parse_args()

    try:
        qtr_runner(args.filename, args.compile)

    except KeyboardInterrupt:
        print(f"[PROCESS TEMINATED] Process from {args.filename} has been terminated")
    except Exception as e:
        import traceback

        selected_tb = e.__traceback__

        if not args.full_traceback and selected_tb is not None:
            current = selected_tb
            while current is not None:
                raw_filename = current.tb_frame.f_code.co_filename
                
                # Instantly catch virtual frames generated via exec() or eval()
                if raw_filename.startswith("<") and raw_filename.endswith(">"):
                    selected_tb = current
                    break

                # Evaluate actual file system boundaries explicitly
                frame_path = Path(raw_filename).resolve()
                is_internal = (frame_path == INTERNAL_RUNNER_FILE or INTERNAL_PACKAGE_DIR in frame_path.parents)

                if not is_internal:
                    selected_tb = current
                    break

                current = current.tb_next

        # Instantiated with the exact slice found, avoiding global state mutation
        te = traceback.TracebackException(type(e), e, selected_tb)
        for line in te.format(colorize=True): #type: ignore
            sys.stderr.write(line)
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
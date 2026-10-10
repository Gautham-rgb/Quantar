"""Hatch build hook to generate parser.py from .gram files using pegen.

Also fetches the CPython grammar matching the installer's Python version.
"""

import subprocess
import sys
import urllib.request
from pathlib import Path
from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class PegenBuildHook(BuildHookInterface):
    """Build hook that runs pegen to generate parsers from .gram files."""

    PLUGIN_NAME = "pegen"

    def initialize(self, version, build_data):
        """Generate parsers before building."""
        self._fetch_python_grammar()
        self._generate_parsers()

    def _fetch_python_grammar(self):
        """Download CPython grammar matching the installer's Python version."""
        grammar_dir = Path("Grammar")
        grammar_dir.mkdir(exist_ok=True)
        target = grammar_dir / "python.gram"

        if target.exists():
            print(f"Using existing {target}")
            return

        # Get installer's Python version
        py_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        url = f"https://raw.githubusercontent.com/python/cpython/{py_version}/Grammar/python.gram"

        print(f"Fetching CPython {py_version} grammar from {url}")
        try:
            urllib.request.urlretrieve(url, target)
            print(f"Downloaded {target} ({target.stat().st_size} bytes)")
        except Exception as e:
            print(f"Warning: Could not fetch grammar for {py_version}: {e}", file=sys.stderr)
            # Fallback to latest main branch
            try:
                fallback_url = "https://raw.githubusercontent.com/python/cpython/main/Grammar/python.gram"
                print(f"Trying fallback: {fallback_url}")
                urllib.request.urlretrieve(fallback_url, target)
                print(f"Downloaded fallback {target}")
            except Exception as e2:
                print(f"Error: Could not fetch any grammar: {e2}", file=sys.stderr)

    def _generate_parsers(self):
        """Run pegen on all .gram files in the project."""
        # Generate quantar parser from lang_systems/quantar.gram
        gram_file = Path("lang_systems/quantar.gram")
        if gram_file.exists():
            output_file = Path("lang_systems/quantar_parser.py")
            output_file.parent.mkdir(parents=True, exist_ok=True)

            cmd = [
                sys.executable, "-m", "pegen",
                str(gram_file),
                "-o", str(output_file)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"pegen failed for {gram_file}: {result.stderr}", file=sys.stderr)
                raise RuntimeError(f"Parser generation failed for {gram_file}")
            print(f"Generated {output_file} from {gram_file}")
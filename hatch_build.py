"""Hatch build hook.

Quantar's parser is generated from the vendored CPython grammar
(``Grammar/python.gram``) plus the operator overlay
(``src/lang_systems/quantar.gram``); see ``src/lang_systems/pegen_gram_gen.py``
for the details and the pegen-0.3.0 workarounds. This hook ensures the base
grammar is present, then runs the build.
"""

import sys
import urllib.request
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

ROOT = Path(__file__).resolve().parent


class PegenBuildHook(BuildHookInterface):
    """Generate the Quantar parser before building."""

    PLUGIN_NAME = "pegen"

    def initialize(self, version, build_data):
        self._fetch_python_grammar()
        self._generate_parser()

    def _fetch_python_grammar(self):
        """Download the CPython grammar matching the installer's Python version."""
        grammar_dir = ROOT / "Grammar"
        grammar_dir.mkdir(exist_ok=True)
        target = grammar_dir / "python.gram"

        if target.exists() and target.stat().st_size > 0:
            print(f"Using existing {target}")
            return

        version = f"{sys.version_info.major}.{sys.version_info.minor}"
        base = "https://raw.githubusercontent.com/python/cpython"
        for ref in (version, "main"):
            url = f"{base}/{ref}/Grammar/python.gram"
            temporary = target.with_suffix(".gram.tmp")
            print(f"Fetching CPython grammar from {url}")
            try:
                urllib.request.urlretrieve(url, temporary)
                temporary.replace(target)  # atomic: never leave a partial file
                print(f"Downloaded {target} ({target.stat().st_size} bytes)")
                return
            except Exception as error:  # noqa: BLE001
                temporary.unlink(missing_ok=True)
                print(f"Could not fetch {url}: {error}", file=sys.stderr)

        raise RuntimeError(
            "Could not download Grammar/python.gram and no vendored copy exists."
        )

    def _generate_parser(self):
        # The build hook runs in an isolated environment where the source tree
        # is not necessarily on sys.path, so make our own modules importable.
        source = str(ROOT / "src")
        if source not in sys.path:
            sys.path.insert(0, source)

        from lang_systems.pegen_gram_gen import build

        build()

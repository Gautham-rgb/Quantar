"""Hatch build hook.

Quantar's parser is generated from the vendored CPython grammar
(``Grammar/python.gram``) plus the operator overlay (``lang_systems/quantar.gram``);
see ``lang_systems/build_parser.py`` for the details and the pegen-0.3.0
workarounds. This hook ensures the base grammar is present, then runs the build.
"""

import sys
import urllib.request
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class PegenBuildHook(BuildHookInterface):
    """Generate the Quantar parser before building."""

    PLUGIN_NAME = "pegen"

    def initialize(self, version, build_data):
        self._fetch_python_grammar()
        self._generate_parser()

    def _fetch_python_grammar(self):
        """Download the CPython grammar matching the installer's Python version."""
        grammar_dir = Path("Grammar")
        grammar_dir.mkdir(exist_ok=True)
        target = grammar_dir / "python.gram"

        if target.exists():
            print(f"Using existing {target}")
            return

        version = f"{sys.version_info.major}.{sys.version_info.minor}"
        base = "https://raw.githubusercontent.com/python/cpython"
        for ref in (version, "main"):
            url = f"{base}/{ref}/Grammar/python.gram"
            print(f"Fetching CPython grammar from {url}")
            try:
                urllib.request.urlretrieve(url, target)
                print(f"Downloaded {target} ({target.stat().st_size} bytes)")
                return
            except Exception as error:  # noqa: BLE001
                print(f"Could not fetch {url}: {error}", file=sys.stderr)

        raise RuntimeError(
            "Could not download Grammar/python.gram and no vendored copy exists."
        )

    def _generate_parser(self):
        from lang_systems.build_parser import build

        build()

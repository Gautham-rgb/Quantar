"""Parse Quantar source into a parse tree.

Quantar is a superset of Python, so the parser is generated from CPython's
grammar (see lang_systems/pegen_gram_gen.py) plus a small operator overlay. This
module adds the two custom-operator token matchers the overlay refers to and a
convenient :func:`parse` entry point.

The result is pegen's raw tree (nested lists of tokens and sub-nodes); turning
it into an AST or executing it is intentionally a separate concern.
"""

import sys
import token

from pegen.tokenizer import Tokenizer

from lang_systems.quantar_parser import GeneratedParser
from lang_systems.quantar_tokens import tokenize_source
from lang_systems.operators import is_known_operator

# The generated parser is deep; parsing the CPython grammar needs headroom.
_RECURSION_LIMIT = 200_000


class QuantarParser(GeneratedParser):  # type: ignore[misc]
    """The generated Python parser plus Quantar's custom-operator matchers."""

    def QUANTUM_OP(self):  # type: ignore[override]
        """Match a symbolic operator (e.g. ``<->``) known to the language."""
        peeked = self._tokenizer.peek()
        if peeked.type == token.OP and is_known_operator(peeked.string):
            return self._tokenizer.getnext()
        return None

    def QUANTUM_NAME(self):  # type: ignore[override]
        """Match an identifier operator (e.g. ``CNOT``) known to the language."""
        peeked = self._tokenizer.peek()
        if (
            peeked.type == token.NAME
            and peeked.string not in self.KEYWORDS
            and is_known_operator(peeked.string)
        ):
            return self._tokenizer.getnext()
        return None


def parse(source: str):
    """Parse `source` and return the raw parse tree (the grammar's `file` rule).

    Raises:
        SyntaxError: If `source` is not valid Quantar.
    """
    sys.setrecursionlimit(max(sys.getrecursionlimit(), _RECURSION_LIMIT))
    tokenizer = Tokenizer(tokenize_source(source))
    parser = QuantarParser(tokenizer)
    tree = parser.file()
    if tree is None:
        # Parsing is backtracking, so peek() may point back at the beginning.
        # The tokenizer retains every token explored; diagnose() returns the
        # furthest token reached by the grammar before it failed.
        bad = parser._tokenizer.diagnose()
        line_text = bad.line or ""
        if not line_text and bad.start:
            lines = source.splitlines()
            if 0 < bad.start[0] <= len(lines):
                line_text = lines[bad.start[0] - 1]
        raise SyntaxError(
            f"invalid Quantar syntax near {bad.string!r}",
            ("<quantar>", bad.start[0], bad.start[1] + 1, line_text),
        )
    return tree

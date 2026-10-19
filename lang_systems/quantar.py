"""Parse Quantar source into a parse tree.

Quantar is a superset of Python, so the parser is generated from CPython's
grammar (see lang_systems/build_parser.py) plus a small operator overlay. This
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
from quantar.operators import get_global_operator

# The generated parser is deep; parsing the CPython grammar needs headroom.
_RECURSION_LIMIT = 200_000


class QuantarParser(GeneratedParser):
    """The generated Python parser plus Quantar's custom-operator matchers."""

    def QUANTUM_OP(self):
        """Match a symbolic operator (e.g. ``<->``) registered in the registry."""
        peeked = self._tokenizer.peek()
        if peeked.type == token.OP and get_global_operator(peeked.string)[0] is not None:
            return self._tokenizer.getnext()
        return None

    def QUANTUM_NAME(self):
        """Match an identifier operator (e.g. ``CNOT``) registered in the registry."""
        peeked = self._tokenizer.peek()
        if (
            peeked.type == token.NAME
            and peeked.string not in self.KEYWORDS
            and get_global_operator(peeked.string)[0] is not None
        ):
            return self._tokenizer.getnext()
        return None


def parse(source: str):
    """Parse `source` and return the raw parse tree (the grammar's `file` rule)."""
    sys.setrecursionlimit(max(sys.getrecursionlimit(), _RECURSION_LIMIT))
    tokenizer = Tokenizer(tokenize_source(source))
    return QuantarParser(tokenizer).file()

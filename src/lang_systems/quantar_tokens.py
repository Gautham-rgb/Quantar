"""Tokenizer glue for Quantar's symbolic operators.

Python's tokenizer splits a symbol such as ``<->`` into ``<``, ``-``, ``>``, so
the generated parser could never match it. This module post-processes the token
stream and merges maximal runs of adjacent ``OP``/``NAME`` tokens that together
spell an operator known to the language into a single ``OP`` token.

Identifier operators (``CNOT``) need no help -- they are ordinary ``NAME`` tokens.
"""

import io
import token
import tokenize
from collections.abc import Iterator

from lang_systems.operators import BUILTIN_OPERATOR_KINDS, list_operators

_MERGEABLE = (token.OP, token.NAME)


def symbolic_operators() -> list[str]:
    """Known operator names that are not plain identifiers, longest first."""
    names = set(list_operators()) | set(BUILTIN_OPERATOR_KINDS)
    symbols = [name for name in names if not name.isidentifier()]
    return sorted(symbols, key=len, reverse=True)


def _match(tokens: list[tokenize.TokenInfo], start: int, symbol: str):
    """Try to match `symbol` against adjacent tokens from `start`.

    Returns (merged_token, index_after) or None.
    """
    position = 0
    index = start
    while position < len(symbol):
        if index >= len(tokens):
            return None
        current = tokens[index]
        if current.type not in _MERGEABLE:
            return None
        piece = current.string
        if not symbol.startswith(piece, position):
            return None
        if index > start and tokens[index - 1].end != current.start:
            return None
        position += len(piece)
        index += 1

    if position != len(symbol):
        return None

    first, last = tokens[start], tokens[index - 1]
    merged = tokenize.TokenInfo(
        type=token.OP,
        string=symbol,
        start=first.start,
        end=last.end,
        line=first.line,
    )
    return merged, index


def merge_operators(tokens: list[tokenize.TokenInfo], symbols: list[str]):
    """Yield `tokens`, replacing adjacent runs that spell a symbol with one OP token."""
    index = 0
    total = len(tokens)
    while index < total:
        current = tokens[index]
        if current.type in _MERGEABLE:
            for symbol in symbols:
                matched = _match(tokens, index, symbol)
                if matched is not None:
                    merged, index = matched
                    yield merged
                    break
            else:
                yield current
                index += 1
        else:
            yield current
            index += 1


def tokenize_source(source: str) -> Iterator[tokenize.TokenInfo]:
    """Tokenize Quantar source, merging registered symbolic operators."""
    tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    return merge_operators(tokens, symbolic_operators())

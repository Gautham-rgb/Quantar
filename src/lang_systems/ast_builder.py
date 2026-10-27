"""Build a Python AST for Quantar source.

Quantar adds a handful of infix operators to Python's syntax.  Reimplementing
CPython's grammar would be wasteful, so this module takes the opposite route:

1. recursively find custom infix expressions in the pegen parse tree;
2. replace each one with a unique sentinel call;
3. hand the resulting text to :func:`ast.parse`;
4. replace the sentinel calls with :class:`QuantumOp` nodes.

The result is ordinary Python AST plus custom :class:`QuantumOp` nodes, which
can then be inspected with :mod:`lang_systems.visitors` or lowered back to calls.
"""

from __future__ import annotations

import ast
import token
import tokenize
from dataclasses import dataclass
from types import CodeType
from uuid import uuid4

from lang_systems.operators import BUILTIN_OPERATOR_KINDS, get_known_operator_kind
from lang_systems.quantar import parse
from lang_systems.quantar_tokens import tokenize_source

_SENTINEL_PREFIX = "__quantar_operator_"
_SENTINEL_SUFFIX = "__"


class QuantumOp(ast.expr):
    """A Quantar infix operation, stored as a first-class AST node."""

    _fields = ("operator", "operands", "kind")

    def __init__(
        self,
        operator: str,
        operands: list[ast.expr],
        kind: str = "binary",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.operator = operator
        self.operands = operands
        self.kind = kind


@dataclass
class _Rewrite:
    name: str
    operator: str
    kind: str
    start: tuple[int, int]
    end: tuple[int, int]


class _Source:
    """Map token positions (line, column) to source offsets and slices."""

    def __init__(self, text: str):
        self.text = text
        self.lines = text.splitlines(keepends=True)
        self.starts: list[int] = []
        offset = 0
        for line in self.lines:
            self.starts.append(offset)
            offset += len(line)

    def offset(self, position: tuple[int, int]) -> int:
        row, column = position
        if row < 1 or row > len(self.starts):
            return len(self.text)
        return self.starts[row - 1] + column

    def slice(self, start: tuple[int, int], end: tuple[int, int]) -> str:
        return self.text[self.offset(start) : self.offset(end)]

    def line(self, row: int) -> str:
        if 1 <= row <= len(self.lines):
            return self.lines[row - 1]
        return ""

    def byte_column(self, position: tuple[int, int]) -> int:
        _, column = position
        return len(self.line(position[0])[:column].encode("utf-8"))


def _first_token(node) -> tokenize.TokenInfo | None:
    if isinstance(node, tokenize.TokenInfo):
        return node
    if isinstance(node, list):
        for item in node:
            found = _first_token(item)
            if found is not None:
                return found
    return None


def _last_token(node) -> tokenize.TokenInfo | None:
    if isinstance(node, tokenize.TokenInfo):
        return node
    if isinstance(node, list):
        for item in reversed(node):
            found = _last_token(item)
            if found is not None:
                return found
    return None


def _custom_operator(node) -> tokenize.TokenInfo | None:
    """Return the operator token if `node` is a custom binary expression."""
    if not isinstance(node, list) or len(node) != 3:
        return None
    left, middle, right = node
    if not isinstance(middle, tokenize.TokenInfo):
        return None
    if middle.type not in (token.OP, token.NAME):
        return None
    if get_known_operator_kind(middle.string) is None:
        return None
    if left is None or right is None:
        return None
    return middle


def _direct_custom_nodes(node) -> list:
    """Return the outermost custom expressions inside a standard node."""
    found = []

    def visit(value):
        if _custom_operator(value) is not None:
            found.append(value)
            return
        if isinstance(value, list):
            for item in value:
                visit(item)

    visit(node)
    return found


def _node_text(node, source: _Source) -> str:
    if isinstance(node, tokenize.TokenInfo):
        return node.string
    if not isinstance(node, list):
        return ""
    first = _first_token(node)
    last = _last_token(node)
    if first is None or last is None:
        return ""
    return source.slice(first.start, last.end)


def _render(node, source: _Source, rewrites: list[_Rewrite]) -> str:
    operator = _custom_operator(node)
    if operator is not None:
        left, _, right = node
        first = _first_token(node)
        last = _last_token(node)
        if first is None or last is None:
            return _node_text(node, source)

        name = f"{_SENTINEL_PREFIX}{uuid4().hex}{_SENTINEL_SUFFIX}"
        # The kind describes the syntax (``a <-> b`` is binary), not the
        # runtime implementation, which may be a matrix or a callable.
        kind = BUILTIN_OPERATOR_KINDS.get(operator.string, "binary")
        rewrites.append(_Rewrite(name, operator.string, kind, first.start, last.end))

        left_text = _render(left, source, rewrites)
        right_text = _render(right, source, rewrites)
        return f"{name}({operator.string!r}, ({left_text}), ({right_text}))"

    if isinstance(node, tokenize.TokenInfo):
        return node.string

    text = _node_text(node, source)
    if not text:
        return text

    first = _first_token(node)
    if first is None:
        return text
    base = source.offset(first.start)
    replacements = []
    for child in _direct_custom_nodes(node):
        child_first = _first_token(child)
        child_last = _last_token(child)
        if child_first is None or child_last is None:
            continue
        start = source.offset(child_first.start) - base
        end = source.offset(child_last.end) - base
        replacements.append((start, end, _render(child, source, rewrites)))

    for start, end, replacement in sorted(replacements, reverse=True):
        text = text[:start] + replacement + text[end:]
    return text


def python_compatible_source(source: str) -> str:
    """Replace Quantar infix operators with Python operators, preserving positions.

    This is a lightweight mirror for external Python language servers. Each
    Quantar operator is replaced with ``or`` and padding spaces of equal UTF-16
    width, so lines and editor columns remain aligned with the original file.
    ``or`` accepts every operand type, which keeps type checkers from reporting
    bogus "unsupported operand" errors for Quantar's gate operators.
    """
    try:
        parsed = parse(source)
    except (SyntaxError, tokenize.TokenError):
        try:
            ast.parse(source)
        except SyntaxError:
            return source
        return source

    operators: list[tokenize.TokenInfo] = []

    def collect(node) -> None:
        operator = _custom_operator(node)
        if operator is not None:
            operators.append(operator)
            collect(node[0])
            collect(node[2])
        elif isinstance(node, list):
            for item in node:
                collect(item)

    collect(parsed)
    if not operators:
        return source

    source_map = _Source(source)
    rewritten = source
    replacements = []
    for operator in operators:
        start = source_map.offset(operator.start)
        end = source_map.offset(operator.end)
        utf16_width = len(operator.string.encode("utf-16-le")) // 2
        if utf16_width >= 2:
            replacement = "or" + (" " * (utf16_width - 2))
        else:
            replacement = "+"
        replacements.append((start, end, replacement))

    for start, end, replacement in sorted(replacements, reverse=True):
        rewritten = rewritten[:start] + replacement + rewritten[end:]

    try:
        ast.parse(rewritten)
    except SyntaxError:
        # Keep broken/incomplete user input intact so the Python server can
        # still provide whatever recovery it supports.
        return source
    return rewritten


class _ReplaceSentinels(ast.NodeTransformer):
    def __init__(self, rewrites: list[_Rewrite], source: _Source):
        self.rewrites = {rewrite.name: rewrite for rewrite in rewrites}
        self.source = source

    def visit_Call(self, node: ast.Call):
        if not (isinstance(node.func, ast.Name) and node.func.id in self.rewrites):
            return self.generic_visit(node)

        node = self.generic_visit(node) #type: ignore
        rewrite = self.rewrites[node.func.id] #type: ignore
        if len(node.args) != 3:
            return node

        result = QuantumOp(rewrite.operator, [node.args[1], node.args[2]], rewrite.kind)
        result.lineno = rewrite.start[0]
        result.col_offset = self.source.byte_column(rewrite.start)
        result.end_lineno = rewrite.end[0]
        result.end_col_offset = self.source.byte_column(rewrite.end)
        return result


class LowerQuantumOps(ast.NodeTransformer):
    """Lower :class:`QuantumOp` nodes to ``__quantar_apply__(name, *operands)``."""

    def visit_QuantumOp(self, node: QuantumOp):
        node = self.generic_visit(node) #type: ignore
        call = ast.Call(
            func=ast.Name(id="__quantar_apply__", ctx=ast.Load()),
            args=[ast.Constant(node.operator), *node.operands],
            keywords=[],
        )
        return ast.copy_location(call, node)


def parse_ast(source: str, filename: str = "<quantar>") -> ast.Module:
    """Parse Quantar source into a Python AST containing :class:`QuantumOp` nodes."""
    source_map = _Source(source)
    rewrites: list[_Rewrite] = []
    try:
        parsed = parse(source)
        rewritten = _render(parsed, source_map, rewrites)
        tree = ast.parse(rewritten, filename=filename)
    except SyntaxError as quantar_error:
        # Keep Quantar's parser for custom operators, but let CPython accept
        # ordinary Python syntax supported by the running Python version.
        try:
            return ast.parse(source, filename=filename)
        except SyntaxError:
            raise quantar_error
    tree = _ReplaceSentinels(rewrites, source_map).visit(tree)
    tree._quantar_rewrites = rewrites
    ast.fix_missing_locations(tree)
    return tree


def lower_ast(tree: ast.AST) -> ast.AST:
    """Return `tree` with all :class:`QuantumOp` nodes lowered to calls."""
    lowered = LowerQuantumOps().visit(tree)
    ast.fix_missing_locations(lowered)
    return lowered


def compile_ast(tree: ast.AST, filename: str = "<quantar>", mode: str = "exec") -> CodeType:
    """Compile an AST produced by :func:`parse_ast`."""
    return compile(lower_ast(tree), filename, mode) #type: ignore


def tokenize_ast(source: str):
    """Return the token stream used while building an AST (useful for tooling)."""
    return tokenize_source(source)


build_ast = parse_ast


__all__ = [
    "QuantumOp",
    "LowerQuantumOps",
    "parse_ast",
    "python_compatible_source",
    "build_ast",
    "lower_ast",
    "compile_ast",
    "tokenize_ast",
]

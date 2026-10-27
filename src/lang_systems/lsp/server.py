"""Quantar language server.

The server deliberately reuses the real tokenizer and parser so editor
highlighting and diagnostics cannot drift from the language implementation.
"""

from __future__ import annotations

import ast
import builtins
import keyword
import logging
import os
import sys
import token
import tokenize
import inspect

from lsprotocol import types
from pygls.lsp.server import LanguageServer

from lang_systems import __version__
from lang_systems.ast_builder import parse_ast, python_compatible_source
from lang_systems.operators import (
    BUILTIN_OPERATOR_KINDS,
    get_known_operator_kind,
    get_operator,
    list_known_operators,
    load_operators,
)
from lang_systems.quantar_tokens import tokenize_source
from lang_systems.symbol_table import analyze

TOKEN_TYPES = ("quantarOperator",)
LEGEND = types.SemanticTokensLegend(
    token_types=list(TOKEN_TYPES),
    token_modifiers=[],
)

logger = logging.getLogger(__name__)

server = LanguageServer("quantar-lsp", __version__)


@server.feature("quantar/convertDocument")
def convert_document(_ls: LanguageServer, params) -> str:
    """Return Python-compatible text for Pylance's analysis of a .qtr file.

    Custom request parameters arrive either as a dict (tests) or as an
    ``Object`` namedtuple (pygls). The response is a plain string so it is
    serialized as JSON text rather than as a converted params object.
    """
    if isinstance(params, dict):
        source = params.get("source")
    else:
        source = getattr(params, "source", None)
    if not isinstance(source, str):
        raise TypeError("quantar/convertDocument requires a string 'source'")
    return python_compatible_source(source)

_OPERATOR_USAGE = {
    "nullary": "{name}",
    "unary_pre": "{name}operand",
    "unary_post": "operand{name}",
    "binary": "left {name} right",
    "ternary": "condition ? when_true : when_false",
    "matrix": "{name}",
    "callable": "{name}(...)",
}


def operator_help(name: str) -> str | None:
    function, implementation_kind = get_operator(name)
    syntax_kind = BUILTIN_OPERATOR_KINDS.get(name)
    if function is None and syntax_kind is None:
        return None

    kind = syntax_kind or implementation_kind or "callable"
    usage = _OPERATOR_USAGE.get(kind, "{name}").format(name=name)
    block = f"```quantar\n{usage}  # {kind} operator\n```"

    doc = inspect.getdoc(function) if function is not None else None
    if doc:
        return f"{block}\n\n{doc}"

    if function is not None and kind in {"callable", "nullary"}:
        try:
            signature = str(inspect.signature(function))
        except (TypeError, ValueError):
            signature = "(...)"
        return f"{block}\n\nRuntime implementation: `{name}{signature}`"
    return block

@server.feature(types.TEXT_DOCUMENT_HOVER)
def _hover(ls: LanguageServer, params: types.HoverParams):
    document = ls.workspace.get_text_document(params.text_document.uri)
    pos = params.position

    if pos.line >= len(document.lines):
        return None

    current_line = document.lines[pos.line]
    # LSP positions are UTF-16 code units; Python tokenizer columns count
    # Unicode code points. Convert the requested cursor position accordingly.
    char_idx = 0
    utf16_units = 0
    while char_idx < len(current_line) and utf16_units < pos.character:
        width = len(current_line[char_idx].encode("utf-16-le")) // 2
        if utf16_units + width > pos.character:
            break
        utf16_units += width
        char_idx += 1

    try:
        hovered = next(
            (
                current
                for current in tokenize_source(document.source)
                if current.start[0] == pos.line + 1
                and current.start[1] <= char_idx < current.end[1]
                and current.type in (token.OP, token.NAME)
            ),
            None,
        )
    except (SyntaxError, tokenize.TokenError):
        return None
    if hovered is None:
        return None

    name = hovered.string
    markdown = operator_help(name)
    if markdown is None:
        kind = get_known_operator_kind(name)
        if kind is None:
            return None
        markdown = f"```quantar\n{name}  # {kind} operator\n```\n\nBuilt-in Quantar operator."

    return types.Hover(
        contents=types.MarkupContent(kind=types.MarkupKind.Markdown, value=markdown),
        range=types.Range(
            start=_position(document, hovered.start[0] - 1, hovered.start[1]),
            end=_position(document, hovered.end[0] - 1, hovered.end[1]),
        ),
    )


def _load_default_operators() -> None:
    """Register the quantum gate operators shipped with Quantar."""
    try:
        import quantum.gate_matrix  # noqa: F401
    except ImportError:
        logger.debug("Quantum gate operators are not available")


def _load_configured_operators() -> None:
    """Import operator modules listed in QUANTAR_OPERATOR_MODULES."""
    raw_modules = os.environ.get("QUANTAR_OPERATOR_MODULES", "")
    modules = [module.strip() for module in raw_modules.split(",") if module.strip()]
    if not modules:
        return

    try:
        load_operators(*modules)
    except Exception:
        logger.exception("Failed to load Quantar operator modules: %s", modules)


def _position(document, line: int, character: int) -> types.Position:
    """Convert a Python code-point position into a client position."""
    return document.position_to_client_units(
        types.Position(line=max(line, 0), character=max(character, 0))
    )


def _full_range(document) -> types.Range:
    lines = document.lines
    if not lines:
        return types.Range(
            start=types.Position(line=0, character=0),
            end=types.Position(line=0, character=0),
        )
    last = len(lines) - 1
    return types.Range(
        start=types.Position(line=0, character=0),
        end=_position(document, last, len(lines[last].rstrip("\r\n"))),
    )


def _syntax_error_range(document, error: SyntaxError) -> types.Range:
    lines = document.lines
    if not lines:
        return _full_range(document)

    line = min(max((error.lineno or 1) - 1, 0), len(lines) - 1)
    text = lines[line]
    start = min(max((error.offset or 1) - 1, 0), len(text))

    end_line = min(
        max((getattr(error, "end_lineno", None) or line + 1) - 1, line),
        len(lines) - 1,
    )
    if end_line == line:
        end = max(
            start + 1,
            min(
                max((getattr(error, "end_offset", None) or start + 2) - 1, 0),
                len(text),
            ),
        )
    else:
        end = len(lines[end_line].rstrip("\r\n"))

    return types.Range(
        start=_position(document, line, start),
        end=_position(document, end_line, end),
    )


def _token_error_range(document, error: tokenize.TokenError) -> types.Range:
    location = error.args[1] if len(error.args) > 1 else (1, 0)
    line, character = location
    return types.Range(
        start=_position(document, line - 1, character),
        end=_position(document, line - 1, character + 1),
    )


def _find_incomplete_operators(document) -> list[types.Diagnostic]:
    diagnostics: list[types.Diagnostic] = []
    try:
        _load_default_operators()
        _load_configured_operators()
        operators = list_known_operators()
        tokens = list(tokenize_source(document.source))
        lines = document.lines
    except (SyntaxError, tokenize.TokenError):
        return []
    except Exception:
        logger.exception("Could not inspect operators in %s", document.uri)
        return []

    # Track bracket depth so an operator that continues inside parentheses is
    # not reported as incomplete just because its line ends there.
    depth = 0
    for current in tokens:
        if current.type == token.OP:
            if current.string in ("(", "[", "{"):
                depth += 1
                continue
            if current.string in (")", "]", "}"):
                depth = max(depth - 1, 0)
                continue

        if current.type not in (token.OP, token.NAME):
            continue
        if current.string not in operators:
            continue
        if current.type == token.NAME and keyword.iskeyword(current.string):
            continue
        if current.string.isidentifier() and operators[current.string] not in {
            "binary",
            "ternary",
        }:
            # Single-letter gate names such as ``s`` are ordinary
            # identifiers far more often than incomplete operators.
            continue
        if depth:
            continue

        line_idx = current.start[0] - 1
        if line_idx < 0 or line_idx >= len(lines):
            continue
        rest = lines[line_idx][current.end[1] :].strip()
        if rest and not rest.startswith("#"):
            continue

        start_pos = _position(document, current.start[0] - 1, current.start[1])
        end_pos = _position(document, current.end[0] - 1, current.end[1])
        diagnostics.append(
            types.Diagnostic(
                range=types.Range(start=start_pos, end=end_pos),
                message=(
                    f"Incomplete operator '{current.string}': missing right operand"
                ),
                severity=types.DiagnosticSeverity.Error,
                source="quantar",
            )
        )
    return diagnostics


def _find_incomplete_augmented_assignments(document) -> list[types.Diagnostic]:
    """Explain a trailing augmented assignment operator as a missing RHS."""
    operators = {"+=", "-=", "*=", "@=", "/=", "%=", "&=", "|=", "^=", "<<=", ">>=", "**=", "//="}
    diagnostics: list[types.Diagnostic] = []
    try:
        tokens = tokenize_source(document.source)
        lines = document.lines
        for current in tokens:
            if current.type != token.OP or current.string not in operators:
                continue
            line_index = current.start[0] - 1
            if line_index < 0 or line_index >= len(lines):
                continue
            remainder = lines[line_index][current.end[1]:].strip()
            if remainder and not remainder.startswith("#"):
                continue
            start = _position(document, current.start[0] - 1, current.start[1])
            end = _position(document, current.end[0] - 1, current.end[1])
            diagnostics.append(
                types.Diagnostic(
                    range=types.Range(start=start, end=end),
                    message=f"Incomplete augmented assignment '{current.string}': expected an expression on the right",
                    severity=types.DiagnosticSeverity.Error,
                    source="quantar",
                )
            )
    except (SyntaxError, tokenize.TokenError):
        return []
    return diagnostics

def _collect_symbol_diagnostics(document) -> list[types.Diagnostic]:
    """Report unresolved names without leaking definitions across scopes."""
    try:
        st, _tree, rewrites, _source_map = analyze(
            document.source, document.path or document.uri
        )
    except Exception:
        logger.exception("Symbol analysis failed for %s", document.uri)
        return []

    definitions: dict[int, set[str]] = {}
    defining_kinds = {
        "var",
        "func",
        "async_func",
        "class",
        "import",
        "import_from",
        "arg",
    }
    for symbol in st.found_symbols:
        if symbol.kind not in defining_kinds:
            continue
        target_scope = symbol.scope_id
        if symbol.is_nonlocal:
            for scope_id, scope_type in reversed(symbol.scope_path[:-1]):
                if scope_type == "function":
                    target_scope = scope_id
                    break
        elif symbol.is_global and target_scope != st.root_scope_id:
            target_scope = st.root_scope_id
        definitions.setdefault(target_scope, set()).add(symbol.name)

    builtin_names = set(dir(builtins))

    def is_defined(symbol) -> bool:
        if symbol.name in builtin_names:
            return True
        if symbol.name in definitions.get(symbol.scope_id, set()):
            return True

        if symbol.scope_id == st.root_scope_id or symbol.is_global:
            return symbol.name in definitions.get(st.root_scope_id, set())

        if symbol.is_free or symbol.is_nonlocal or symbol.scope == "comprehension":
            for scope_id, scope_type in reversed(symbol.scope_path[:-1]):
                # Methods do not close over class attributes.
                if scope_type == "class":
                    continue
                if symbol.name in definitions.get(scope_id, set()):
                    return True
        return False

    unresolved = [
        symbol
        for symbol in st.found_symbols
        if symbol.kind == "ref"
        and not symbol.name.startswith("__")
        and not is_defined(symbol)
    ]
    if not unresolved:
        return []

    if not rewrites:
        # AST columns count UTF-8 bytes, while the LSP document API accepts
        # Python character offsets. With no sentinel expansion, locations map
        # directly back to the source and need no token-name guessing.
        diagnostics = []
        for symbol in unresolved:
            start_line, start_byte = symbol.start
            end_line, end_byte = symbol.end
            if start_line < 1 or end_line < 1:
                continue
            start_text = document.lines[start_line - 1]
            end_text = document.lines[end_line - 1]
            start_char = len(start_text.encode("utf-8")[:start_byte].decode("utf-8", "ignore"))
            end_char = len(end_text.encode("utf-8")[:end_byte].decode("utf-8", "ignore"))
            diagnostics.append(
                types.Diagnostic(
                    range=types.Range(
                        start=_position(document, start_line - 1, start_char),
                        end=_position(document, end_line - 1, end_char),
                    ),
                    message=f"Undefined variable '{symbol.name}'",
                    severity=types.DiagnosticSeverity.Warning,
                    source="quantar",
                )
            )
        return diagnostics

    try:
        tokens = [
            current
            for current in tokenize_source(document.source)
            if current.type == token.NAME
        ]
    except (SyntaxError, tokenize.TokenError):
        return []

    diagnostics: list[types.Diagnostic] = []
    used_tokens: set[int] = set()
    for symbol in unresolved:
        candidates = [
            (index, current)
            for index, current in enumerate(tokens)
            if index not in used_tokens and current.string == symbol.name
        ]
        if not candidates:
            continue

        line, column = symbol.start
        index, current = min(
            candidates,
            key=lambda item: (
                abs(item[1].start[0] - line) * 1_000_000
                + abs(item[1].start[1] - column)
            ),
        )
        used_tokens.add(index)
        start = _position(document, current.start[0] - 1, current.start[1])
        end = _position(document, current.end[0] - 1, current.end[1])
        diagnostics.append(
            types.Diagnostic(
                range=types.Range(start=start, end=end),
                message=f"Undefined variable '{symbol.name}'",
                severity=types.DiagnosticSeverity.Warning,
                source="quantar",
            )
        )
    return diagnostics

def _publish_diagnostics(ls: LanguageServer, document) -> None:
    diagnostics: list[types.Diagnostic] = []
    try:
        filename = document.path or document.uri
        parse_ast(document.source, filename)
        diagnostics.extend(_collect_symbol_diagnostics(document))
    except SyntaxError:
        # Python is authoritative for ordinary syntax. This fallback prevents
        # stale generated Quantar grammar from rejecting valid Python such as
        # ``value += 1`` while retaining Quantar's custom-operator parser.
        try:
            ast.parse(document.source, filename=document.path or document.uri)
        except SyntaxError as python_error:
            diagnostics.extend(
                _find_incomplete_operators(document)
                + _find_incomplete_augmented_assignments(document)
            )
            rng = _syntax_error_range(document, python_error)
            is_dup = any(
                d.range.start.line == rng.start.line
                and d.range.start.character
                <= rng.start.character
                <= d.range.end.character
                for d in diagnostics
            )
            if not is_dup:
                diagnostics.append(
                    types.Diagnostic(
                        range=rng,
                        message=python_error.msg or "Invalid Quantar syntax",
                        severity=types.DiagnosticSeverity.Error,
                        source="quantar",
                    )
                )
        else:
            diagnostics.extend(_collect_symbol_diagnostics(document))
    except tokenize.TokenError as error:
        diagnostics.append(
            types.Diagnostic(
                range=_token_error_range(document, error),
                message=error.args[0] if error.args else "Tokenizer error",
                severity=types.DiagnosticSeverity.Error,
                source="quantar",
            )
        )
    except Exception as error:
        op_diags = _find_incomplete_operators(document) + _find_incomplete_augmented_assignments(document)
        diagnostics.extend(op_diags)
        if not diagnostics:
            diagnostics.append(
                types.Diagnostic(
                    range=_full_range(document),
                    message=str(error),
                    severity=types.DiagnosticSeverity.Error,
                    source="quantar",
                )
            )

    for diagnostic in diagnostics:
        line_number = diagnostic.range.start.line + 1
        diagnostic.message = f"Line {line_number}: {diagnostic.message}"

    seen = set()
    unique_diags = []
    for d in diagnostics:
        key = (d.range.start.line, d.range.start.character, d.range.end.line, d.range.end.character, d.message)
        if key not in seen:
            seen.add(key)
            unique_diags.append(d)

    ls.text_document_publish_diagnostics(
        types.PublishDiagnosticsParams(
            uri=document.uri,
            diagnostics=unique_diags,
            version=document.version,
        )
    )


def _completion_items(document) -> list[types.CompletionItem]:
    items: list[types.CompletionItem] = []
    seen: set[str] = set()

    for name, kind in list_known_operators().items():
        seen.add(name)
        items.append(
            types.CompletionItem(
                label=name,
                kind=types.CompletionItemKind.Operator,
                detail=f"Quantar {kind} operator",
                documentation=types.MarkupContent(
                    kind=types.MarkupKind.Markdown,
                    value=operator_help(name) or f"Built-in Quantar `{kind}` operator.",
                ),
                sort_text=f"0_{name}",
            )
        )

    symbol_kinds = {
        "var": types.CompletionItemKind.Variable,
        "arg": types.CompletionItemKind.Variable,
        "func": types.CompletionItemKind.Function,
        "async_func": types.CompletionItemKind.Function,
        "class": types.CompletionItemKind.Class,
        "import": types.CompletionItemKind.Module,
        "import_from": types.CompletionItemKind.Module,
    }
    try:
        st, _tree, _rewrites, _source_map = analyze(
            document.source, document.path or document.uri
        )
    except Exception:
        st = None
    if st is not None:
        for symbol in st.found_symbols:
            item_kind = symbol_kinds.get(symbol.kind)
            if item_kind is None or symbol.name in seen:
                continue
            seen.add(symbol.name)
            items.append(
                types.CompletionItem(
                    label=symbol.name,
                    kind=item_kind,
                    detail=f"Quantar {symbol.kind.replace('_', ' ')}",
                    sort_text=f"1_{symbol.name}",
                )
            )

    for name in dir(builtins):
        if name.startswith("_") or name in seen:
            continue
        seen.add(name)
        items.append(
            types.CompletionItem(
                label=name,
                kind=types.CompletionItemKind.Function,
                detail="Python built-in",
                sort_text=f"3_{name}",
            )
        )

    for name in keyword.kwlist:
        if name in seen:
            continue
        seen.add(name)
        items.append(
            types.CompletionItem(
                label=name,
                kind=types.CompletionItemKind.Keyword,
                sort_text=f"2_{name}",
            )
        )

    return items


@server.feature(
    types.TEXT_DOCUMENT_COMPLETION,
    types.CompletionOptions(trigger_characters=["<", "-", "~"]),
)
def completion(
    ls: LanguageServer, params: types.CompletionParams
) -> types.CompletionList:
    document = ls.workspace.get_text_document(params.text_document.uri)
    return types.CompletionList(is_incomplete=False, items=_completion_items(document))


def _semantic_tokens(document) -> types.SemanticTokens:
    """Highlight registered symbolic operators where they actually occur."""
    symbols = {name for name in list_known_operators() if not name.isidentifier()}
    data: list[int] = []
    previous_line = 0
    previous_character = 0

    try:
        tokens = list(tokenize_source(document.source))
    except (SyntaxError, tokenize.TokenError):
        return types.SemanticTokens(data=[])

    for current in tokens:
        if current.string not in symbols or current.start[0] != current.end[0]:
            continue
        start = _position(document, current.start[0] - 1, current.start[1])
        end = _position(document, current.end[0] - 1, current.end[1])
        length = end.character - start.character
        if length <= 0:
            continue
        delta_line = start.line - previous_line
        delta_start = (
            start.character - previous_character if delta_line == 0 else start.character
        )
        data.extend((delta_line, delta_start, length, 0, 0))
        previous_line = start.line
        previous_character = start.character

    return types.SemanticTokens(data=data)


@server.feature(types.TEXT_DOCUMENT_SEMANTIC_TOKENS_FULL, LEGEND)
def semantic_tokens_full(
    ls: LanguageServer, params: types.SemanticTokensParams
) -> types.SemanticTokens:
    return _semantic_tokens(ls.workspace.get_text_document(params.text_document.uri))


@server.feature(types.TEXT_DOCUMENT_DID_OPEN)
def did_open(ls: LanguageServer, params: types.DidOpenTextDocumentParams) -> None:
    _publish_diagnostics(ls, ls.workspace.get_text_document(params.text_document.uri))


@server.feature(types.TEXT_DOCUMENT_DID_CHANGE)
def did_change(ls: LanguageServer, params: types.DidChangeTextDocumentParams) -> None:
    _publish_diagnostics(ls, ls.workspace.get_text_document(params.text_document.uri))


@server.feature(types.TEXT_DOCUMENT_DID_CLOSE)
def did_close(ls: LanguageServer, params: types.DidCloseTextDocumentParams) -> None:
    ls.text_document_publish_diagnostics(
        types.PublishDiagnosticsParams(
            uri=params.text_document.uri,
            diagnostics=[],
        )
    )


@server.feature(types.TEXT_DOCUMENT_DID_SAVE)
def did_save(ls: LanguageServer, params: types.DidSaveTextDocumentParams) -> None:
    _publish_diagnostics(ls, ls.workspace.get_text_document(params.text_document.uri))


def main() -> None:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    _load_default_operators()
    _load_configured_operators()
    server.start_io()


if __name__ == "__main__":
    main()

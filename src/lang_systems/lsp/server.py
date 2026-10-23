"""Quantar language server.

The server deliberately reuses the real tokenizer and parser so editor
highlighting and diagnostics cannot drift from the language implementation.
"""

from __future__ import annotations

import keyword
import logging
import os
import sys
import token
import tokenize

from lsprotocol import types
from pygls.lsp.server import LanguageServer

from lang_systems.ast_builder import parse_ast
from lang_systems.operators import list_known_operators, load_operators
from lang_systems.quantar_tokens import tokenize_source

TOKEN_TYPES = ("quantarOperator",)
TOKEN_MODIFIERS = (
    "nullary",
    "unary",
    "binary",
    "ternary",
    "callable",
    "matrix",
)
MODIFIER_BITS = {
    name: 1 << index for index, name in enumerate(TOKEN_MODIFIERS)
}
LEGEND = types.SemanticTokensLegend(
    token_types=list(TOKEN_TYPES),
    token_modifiers=list(TOKEN_MODIFIERS),
)

logger = logging.getLogger(__name__)

server = LanguageServer("quantar-lsp", "0.1.0")


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


def _publish_diagnostics(ls: LanguageServer, document) -> None:
    diagnostics: list[types.Diagnostic] = []
    try:
        parse_ast(document.source, document.path or document.uri)
    except SyntaxError as error:
        diagnostics.append(
            types.Diagnostic(
                range=_syntax_error_range(document, error),
                message=error.msg or "Invalid Quantar syntax",
                severity=types.DiagnosticSeverity.Error,
                source="quantar",
            )
        )
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
        diagnostics.append(
            types.Diagnostic(
                range=_full_range(document),
                message=str(error),
                severity=types.DiagnosticSeverity.Error,
                source="quantar",
            )
        )

    ls.text_document_publish_diagnostics(
        types.PublishDiagnosticsParams(
            uri=document.uri,
            diagnostics=diagnostics,
            version=document.version,
        )
    )


@server.feature(types.TEXT_DOCUMENT_DID_OPEN)
def did_open(ls: LanguageServer, params: types.DidOpenTextDocumentParams) -> None:
    document = ls.workspace.get_text_document(params.text_document.uri)
    _publish_diagnostics(ls, document)


@server.feature(types.TEXT_DOCUMENT_DID_CHANGE)
def did_change(ls: LanguageServer, params: types.DidChangeTextDocumentParams) -> None:
    document = ls.workspace.get_text_document(params.text_document.uri)
    _publish_diagnostics(ls, document)


@server.feature(types.TEXT_DOCUMENT_DID_SAVE)
def did_save(ls: LanguageServer, params: types.DidSaveTextDocumentParams) -> None:
    document = ls.workspace.get_text_document(params.text_document.uri)
    _publish_diagnostics(ls, document)


@server.feature(
    types.TEXT_DOCUMENT_SEMANTIC_TOKENS_FULL,
    LEGEND,
)
def semantic_tokens_full(
    ls: LanguageServer, params: types.SemanticTokensParams
) -> types.SemanticTokens:
    document = ls.workspace.get_text_document(params.text_document.uri)
    operators = list_known_operators()
    data: list[int] = []

    try:
        tokens = tokenize_source(document.source)
        previous_line = 0
        previous_start = 0

        for current in tokens:
            if current.type not in (token.OP, token.NAME):
                continue
            if current.string not in operators:
                continue
            if current.type == token.NAME and keyword.iskeyword(current.string):
                continue

            start = _position(document, current.start[0] - 1, current.start[1])
            end = _position(document, current.end[0] - 1, current.end[1])
            if start.line != end.line or end.character <= start.character:
                continue

            delta_line = start.line - previous_line
            delta_start = (
                start.character - previous_start
                if delta_line == 0
                else start.character
            )
            modifier = MODIFIER_BITS.get(operators[current.string], 0)

            data.extend(
                [
                    delta_line,
                    delta_start,
                    end.character - start.character,
                    0,
                    modifier,
                ]
            )
            previous_line = start.line
            previous_start = start.character
    except (SyntaxError, tokenize.TokenError, IndentationError):
        return types.SemanticTokens(data=[])

    return types.SemanticTokens(data=data)


def main() -> None:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    _load_default_operators()
    _load_configured_operators()
    server.start_io()


if __name__ == "__main__":
    main()

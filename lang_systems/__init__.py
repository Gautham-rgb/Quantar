"""Quantar language systems - parsers, ASTs, and language tools.

``lang_systems.parse`` returns pegen's raw parse tree.  ``lang_systems.parse_ast``
returns a Python AST with custom :class:`QuantumOp` nodes.  Both are imported
lazily so that importing this package does not require the generated parser
(which is produced at build time) to already exist.
"""

import importlib

_EXPORTS = {
    "parse": ("lang_systems.quantar", "parse"),
    "parse_ast": ("lang_systems.ast_builder", "parse_ast"),
    "build_ast": ("lang_systems.ast_builder", "build_ast"),
    "lower_ast": ("lang_systems.ast_builder", "lower_ast"),
    "compile_ast": ("lang_systems.ast_builder", "compile_ast"),
    "QuantumOp": ("lang_systems.ast_builder", "QuantumOp"),
    "LowerQuantumOps": ("lang_systems.ast_builder", "LowerQuantumOps"),
    "load_operators": ("lang_systems.operators", "load_operators"),
    "QuantarNodeVisitor": ("lang_systems.visitors", "QuantarNodeVisitor"),
    "OperatorCollector": ("lang_systems.visitors", "OperatorCollector"),
}


def __getattr__(name):
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    value = getattr(importlib.import_module(module_name), attribute)
    globals()[name] = value
    return value


__all__ = list(_EXPORTS)

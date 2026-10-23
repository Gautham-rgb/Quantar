"""Visitors for Quantar's extended AST.

These are ordinary :class:`ast.NodeVisitor` subclasses.  Subclass
:class:`QuantarNodeVisitor` when you want custom operators to be traversed by
default, then override :meth:`visit_QuantumOp` to do your own work.
"""

import ast

from lang_systems.ast_builder import QuantumOp

__all__ = ["QuantarNodeVisitor", "OperatorCollector"]


class QuantarNodeVisitor(ast.NodeVisitor):
    """Base visitor that descends into :class:`QuantumOp` operands."""

    def visit_QuantumOp(self, node: QuantumOp):
        self.generic_visit(node)


class OperatorCollector(QuantarNodeVisitor):
    """Collect ``(operator, kind)`` pairs from a Quantar AST."""

    def __init__(self):
        self.operators: list[tuple[str, str]] = []

    def visit_QuantumOp(self, node: QuantumOp):
        self.operators.append((node.operator, node.kind))
        self.generic_visit(node)

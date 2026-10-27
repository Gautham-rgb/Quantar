from __future__ import annotations

import ast
import symtable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import visitors


@dataclass
class Symbol:
    name: str
    kind: str
    start: tuple[int, int]
    end: tuple[int, int]
    node: ast.AST
    scope: str = "module"
    scope_id: int = 0
    scope_path: tuple[tuple[int, str], ...] = ()
    is_global: bool = False
    is_free: bool = False
    is_nonlocal: bool = False


class SymbolTable(visitors.QuantarNodeVisitor):
    """Collect name occurrences with the lexical scopes Python assigns them."""

    def __init__(self, source_code: str, filename: str = "<quantar>") -> None:
        root_table = symtable.symtable(source_code, filename, "exec")
        self.scope_stack = [root_table]
        self.root_scope_id = id(root_table)
        self.scope_path = [(id(root_table), str(root_table.get_type()))]
        self._used_child_scopes: dict[int, set[int]] = {}
        self._scope_children: dict[int, list[Any]] = {}
        self._virtual_scope_id = 0
        self.found_symbols: list[Symbol] = []

    @property
    def current_scope(self):
        return self.scope_stack[-1]

    def _add_symbol(self, name: str, node: ast.AST, default_kind: str) -> None:
        is_global = is_free = is_nonlocal = False
        try:
            sym = self.current_scope.lookup(name)
            kind = "arg" if sym.is_parameter() else default_kind
            is_global = sym.is_global()
            is_free = sym.is_free()
            is_nonlocal = sym.is_nonlocal()
        except KeyError:
            kind = default_kind

        start = (getattr(node, "lineno", 0), getattr(node, "col_offset", 0))
        end = (
            getattr(node, "end_lineno", start[0]),
            getattr(node, "end_col_offset", start[1] + len(name)),
        )
        scope_id, scope_type = self.scope_path[-1]

        # Python 3.12+ inlines list/set/dict comprehensions in symtable. Keep
        # their target bindings in our synthetic comprehension scope.
        if (
            scope_type == "comprehension"
            and isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Store)
        ):
            is_global = False

        self.found_symbols.append(
            Symbol(
                name=name,
                kind=kind,
                start=start,
                end=end,
                node=node,
                scope=scope_type,
                scope_id=scope_id,
                scope_path=tuple(self.scope_path),
                is_global=is_global,
                is_free=is_free,
                is_nonlocal=is_nonlocal,
            )
        )

    @contextmanager
    def track_scope(
        self, scope_names: tuple[str, ...], lineno: int, *, inline: bool = False
    ):
        parent = self.current_scope
        used = self._used_child_scopes.setdefault(id(parent), set())
        children = self._scope_children.setdefault(id(parent), list(parent.get_children()))
        child = next(
            (
                candidate
                for candidate in children
                if candidate.get_name().strip("<>") in scope_names
                and candidate.get_lineno() == lineno
                and id(candidate) not in used
            ),
            None,
        )
        if child is None and not inline:
            raise LookupError(
                f"No symbol-table scope {scope_names!r} at line {lineno}"
            )

        if child is None:
            self._virtual_scope_id -= 1
            scope_id, scope_type, table = (
                self._virtual_scope_id,
                "comprehension",
                parent,
            )
        else:
            used.add(id(child))
            scope_id, scope_type, table = (
                id(child),
                str(child.get_type()),
                child,
            )

        self.scope_stack.append(table)
        self.scope_path.append((scope_id, scope_type))
        try:
            yield
        finally:
            self.scope_stack.pop()
            self.scope_path.pop()

    @staticmethod
    def _arguments(arguments):
        result = [*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs]
        if arguments.vararg is not None:
            result.append(arguments.vararg)
        if arguments.kwarg is not None:
            result.append(arguments.kwarg)
        return result

    def _visit_function_outer_expressions(self, node) -> None:
        for decorator in node.decorator_list:
            self.visit(decorator)
        for default in node.args.defaults:
            self.visit(default)
        for default in node.args.kw_defaults:
            if default is not None:
                self.visit(default)
        for argument in self._arguments(node.args):
            if argument.annotation is not None:
                self.visit(argument.annotation)
        if node.returns is not None:
            self.visit(node.returns)

    def _visit_function_body(self, node) -> None:
        for argument in self._arguments(node.args):
            self._add_symbol(argument.arg, argument, "arg")
        for statement in node.body:
            self.visit(statement)

    def visit_Name(self, node: ast.Name) -> Any:
        if isinstance(node.ctx, ast.Store):
            kind = "var"
        elif isinstance(node.ctx, ast.Del):
            kind = "del"
        else:
            kind = "ref"
        self._add_symbol(node.id, node, kind)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        self._visit_function_outer_expressions(node)
        self._add_symbol(node.name, node, "func")
        with self.track_scope((node.name,), node.lineno):
            self._visit_function_body(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
        self._visit_function_outer_expressions(node)
        self._add_symbol(node.name, node, "async_func")
        with self.track_scope((node.name,), node.lineno):
            self._visit_function_body(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        for decorator in node.decorator_list:
            self.visit(decorator)
        for base in node.bases:
            self.visit(base)
        for keyword in node.keywords:
            self.visit(keyword.value)
        self._add_symbol(node.name, node, "class")
        with self.track_scope((node.name,), node.lineno):
            for statement in node.body:
                self.visit(statement)

    def visit_Lambda(self, node: ast.Lambda) -> Any:
        for default in node.args.defaults:
            self.visit(default)
        for default in node.args.kw_defaults:
            if default is not None:
                self.visit(default)
        with self.track_scope(("lambda",), node.lineno):
            for argument in self._arguments(node.args):
                self._add_symbol(argument.arg, argument, "arg")
            self.visit(node.body)

    def _visit_comprehension(self, node, scope_names, values) -> None:
        generators = node.generators
        if not generators:
            self.generic_visit(node)
            return

        # The first iterable is evaluated outside the comprehension scope.
        self.visit(generators[0].iter)
        with self.track_scope(scope_names, node.lineno, inline=True):
            for index, generator in enumerate(generators):
                if index:
                    self.visit(generator.iter)
                self.visit(generator.target)
                for condition in generator.ifs:
                    self.visit(condition)
            for value in values:
                self.visit(value)

    def visit_ListComp(self, node: ast.ListComp) -> Any:
        self._visit_comprehension(node, ("listcomp",), (node.elt,))

    def visit_SetComp(self, node: ast.SetComp) -> Any:
        self._visit_comprehension(node, ("setcomp",), (node.elt,))

    def visit_DictComp(self, node: ast.DictComp) -> Any:
        self._visit_comprehension(node, ("dictcomp",), (node.key, node.value))

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> Any:
        self._visit_comprehension(node, ("genexpr",), (node.elt,))

    def visit_QuantumOp(self, node: visitors.QuantumOp) -> Any:
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> Any:
        for alias in node.names:
            name = alias.asname if alias.asname else alias.name.split(".")[0]
            self._add_symbol(name, node, "import")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> Any:
        for alias in node.names:
            name = alias.asname if alias.asname else alias.name
            self._add_symbol(name, node, "import_from")
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> Any:
        if node.type is not None:
            self.visit(node.type)
        if node.name:
            self._add_symbol(node.name, node, "var")
        for statement in node.body:
            self.visit(statement)

    def visit_Match(self, node: ast.Match) -> Any:
        self.visit(node.subject)
        for case in node.cases:
            self.visit(case.pattern)
            for pattern_node in ast.walk(case.pattern):
                name = None
                if isinstance(pattern_node, (ast.MatchAs, ast.MatchStar)):
                    name = pattern_node.name
                elif isinstance(pattern_node, ast.MatchMapping):
                    name = pattern_node.rest
                if name:
                    self._add_symbol(name, pattern_node, "var")
            if case.guard is not None:
                self.visit(case.guard)
            for statement in case.body:
                self.visit(statement)


def analyze(source_code: str, filename: str | Path = "<quantar>"):
    from . import ast_builder

    filename = str(filename)
    source_map = ast_builder._Source(source_code)
    # parse_ast may deliberately fall back to the running Python parser for
    # syntax newer than Quantar's generated grammar. Use the same compatible
    # source for symtable so that fallback documents still get name analysis.
    tree = ast_builder.parse_ast(source_code, filename)
    rewritten = ast_builder.python_compatible_source(source_code)
    st = SymbolTable(rewritten, filename)
    st.visit(tree)

    return st, tree, getattr(tree, "_quantar_rewrites", []), source_map

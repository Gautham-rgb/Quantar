"""Generate ``src/lang_systems/quantar_parser.py``.

Quantar is a *superset* of Python: this builds its parser from the vendored
CPython grammar (``Grammar/python.gram``) plus the small operator overlay in
``src/lang_systems/quantar.gram``.

pegen 0.3.0 is the last PyPI release and predates several things we need, so a
few fixes are applied before generating (each one is marked below). Run directly
with::

    python -m lang_systems.pegen_gram_gen

or let the build do it (see ``hatch_build.py``).
"""

import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
ROOT = PACKAGE_ROOT.parent.parent
BASE_GRAMMAR = ROOT / "Grammar" / "python.gram"
OVERLAY_GRAMMAR = PACKAGE_ROOT / "quantar.gram"
MERGED_GRAMMAR = ROOT / "Grammar" / "quantar_merged.gram"
OUTPUT_PARSER = PACKAGE_ROOT / "quantar_parser.py"

# Pseudo-tokens the overlay grammar references; implemented at runtime in
# lang_systems/quantar.py (registered here so pegen's checker accepts them).
EXTRA_TOKENS = {"QUANTUM_OP", "QUANTUM_NAME"}

# CPython grammar rules whose names collide (case-insensitively) with pegen's
# base token methods (e.g. rule `string` vs the STRING token method). pegen
# 0.3.0 case-folds and generates `self.string()` -> infinite recursion, so the
# rules are renamed.
COLLIDING_RULES = {
    "name",
    "number",
    "string",
    "fstring_start",
    "fstring_middle",
    "fstring_end",
    "op",
    "type_comment",
    "soft_keyword",
}

# Splice that gives custom operators their precedence (between `comparison`
# and `bitwise_or`); see the header of lang_systems/quantar.gram. The comparison
# left operand and every comparison RHS become `quantum_expr`.
COMPARISON_OLD = "comparison[expr_ty]:\n    | a=bitwise_or b=compare_op_bitwise_or_pair+"
COMPARISON_NEW = "comparison[expr_ty]:\n    | a=quantum_expr b=compare_op_bitwise_or_pair+"
COMPARISON_BASE_OLD = "\n    | bitwise_or\n\ncompare_op_bitwise_or_pair["
COMPARISON_BASE_NEW = "\n    | quantum_expr\n\ncompare_op_bitwise_or_pair["
COMPARE_PAIRS_START = "compare_op_bitwise_or_pair[CmpopExprPair*]:"
COMPARE_PAIRS_END = "# Bitwise operators"


def _allow_comparison_rhs_operators(base: str) -> str:
    """Use `quantum_expr` on both sides of comparisons, not just the left."""
    start = base.find(COMPARE_PAIRS_START)
    end = base.find(COMPARE_PAIRS_END, start)
    if start == -1 or end == -1:
        raise RuntimeError(
            "Could not find the comparison operator rules to splice; the "
            "vendored Grammar/python.gram changed."
        )
    block = base[start:end].replace("a=bitwise_or", "a=quantum_expr")
    return base[:start] + block + base[end:]


def _build_grammar_text() -> str:
    base = BASE_GRAMMAR.read_text(encoding="utf-8")
    for old in (COMPARISON_OLD, COMPARISON_BASE_OLD):
        if old not in base:
            raise RuntimeError(
                "Could not find the `comparison` rule to splice; the vendored "
                "Grammar/python.gram changed. Update the COMPARISON_* constants."
            )
    base = base.replace(COMPARISON_OLD, COMPARISON_NEW, 1)
    base = base.replace(COMPARISON_BASE_OLD, COMPARISON_BASE_NEW, 1)
    base = _allow_comparison_rhs_operators(base)
    overlay = OVERLAY_GRAMMAR.read_text(encoding="utf-8")
    merged = base + "\n" + overlay

    MERGED_GRAMMAR.write_text(merged, encoding="utf-8")
    return merged


def build() -> None:
    # The full CPython grammar is deep; generation recurses a lot.
    sys.setrecursionlimit(200_000)

    import token

    from pegen.build import build_parser
    from pegen.grammar import Alt, NameLeaf
    from pegen.python_generator import (
        PythonCallMakerVisitor,
        PythonParserGenerator,
    )

    # pegen 0.3.0 predates t-strings (Python 3.14). Map TSTRING_* tokens to
    # expect() instead of a (nonexistent) self.TSTRING_START() method.
    original_name_leaf = PythonCallMakerVisitor.visit_NameLeaf

    def patched_name_leaf(self, node):
        name = node.value
        if name in ("TSTRING_START", "TSTRING_MIDDLE", "TSTRING_END"):
            return "_" + name.lower(), f"self.expect({name!r})"
        return original_name_leaf(self, node)

    PythonCallMakerVisitor.visit_NameLeaf = patched_name_leaf

    _build_grammar_text()
    grammar, _, _ = build_parser(str(MERGED_GRAMMAR))

    # pegen 0.3.0 ignores --skip-actions (a `# TODO` in the source) and still
    # ast.parse()s every action, so the CPython C actions must be removed. The
    # `@trailer` meta (C `_PyPegen_parse`) and the `[return_type]` annotations
    # would also leak into the generated Python, so drop those too.
    grammar.metas = {"trailer": ""} #type: ignore
    renames = {r: r + "_rule" for r in COLLIDING_RULES if r in grammar.rules}

    def strip(node, seen):
        if id(node) in seen:
            return
        seen.add(id(node))
        if isinstance(node, Alt) and node.action:
            node.action = None
        if isinstance(node, NameLeaf):
            if node.value in renames:
                node.value = renames[node.value]
            return
        for value in vars(node).values():
            if isinstance(value, list):
                for item in value:
                    strip(item, seen)
            elif hasattr(value, "__dict__"):
                strip(value, seen)

    seen = set()
    for rule in grammar.rules.values():
        rule.type = None
        strip(rule.rhs, seen)

    for old, new in renames.items():
        rule = grammar.rules.pop(old)
        rule.name = new
        grammar.rules[new] = rule

    tokens = set(token.tok_name.values()) | EXTRA_TOKENS
    with open(OUTPUT_PARSER, "w", encoding="utf-8") as out:
        generator = PythonParserGenerator(grammar, out, tokens=tokens)
        generator.generate(str(MERGED_GRAMMAR))

    print(f"Generated {OUTPUT_PARSER.relative_to(ROOT)} ({len(grammar.rules)} rules)")


if __name__ == "__main__":
    build()

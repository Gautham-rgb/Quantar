"""Generic operator registry and ``@operator`` decorator.

Completely independent of quantum code -- it can be used for anything.

Operator kinds are named by **how the operator is written**, which is usually
what you care about:

    kind          operands   example
    ----------    --------   -------------------------
    nullary       0          ``pi``
    unary_pre     1          ``-x``, ``~x``
    unary_post    1          ``x!``, ``x†``
    binary        2          ``a <-> b``
    ternary       3          ``a ? b : c``
    callable      any        ``f(a, b, c)``
    matrix        0          a gate factory returning an ``np.ndarray``

``kind`` is optional and defaults to ``"callable"``. Common synonyms are
accepted (``"unary"``/``"prefix"`` -> ``unary_pre``, ``"infix"`` -> ``binary``,
...), so existing code keeps working. As a shortcut you can write
``@operator.binary("CNOT")`` instead of ``@operator(kind="binary", name="CNOT")``.
"""

from functools import wraps
from typing import Any, Callable, Literal

import numpy as np


OperatorKind = Literal[
    "nullary",
    "unary_pre",
    "unary_post",
    "binary",
    "ternary",
    "callable",
    "matrix",
]

#: Every accepted kind, in a convenient order for docs/validation.
OPERATOR_KINDS: tuple[str, ...] = (
    "nullary",
    "unary_pre",
    "unary_post",
    "binary",
    "ternary",
    "callable",
    "matrix",
)

#: Friendly synonyms, so the old names and obvious variations still work.
_KIND_ALIASES: dict[str, str] = {
    "constant": "nullary",
    "value": "nullary",
    "zero": "nullary",
    "unary": "unary_pre",
    "prefix": "unary_pre",
    "pre": "unary_pre",
    "postfix": "unary_post",
    "post": "unary_post",
    "infix": "binary",
    "nary": "callable",
    "any": "callable",
    "func": "callable",
    "gate": "matrix",
    "gate_matrix": "matrix",
}


def normalize_kind(kind: str) -> str:
    """Return the canonical kind name for `kind` (or raise ``ValueError``)."""
    if not isinstance(kind, str):
        raise ValueError(f"Operator kind must be a string, got {type(kind).__name__}.")
    key = _KIND_ALIASES.get(kind.strip().lower(), kind.strip().lower())
    if key not in OPERATOR_KINDS:
        raise ValueError(
            f"Unknown operator kind {kind!r}. Choose one of: {', '.join(OPERATOR_KINDS)}."
        )
    return key


# Global operator registry
_global_operators: dict[str, Callable] = {}
_operator_metadata: dict[str, dict[str, Any]] = {}


def _mark(obj, name: str, kind: str, requires_instance: bool) -> None:
    obj._operator_name = name
    obj._is_operator = True
    obj._requires_instance = requires_instance
    obj._kind = kind


def _register(name: str, kind: str, func: Callable) -> Callable:
    if not isinstance(name, str) or not name:
        raise ValueError("Operator name must be a non-empty string")
    kind = normalize_kind(kind)

    # Handle staticmethod/classmethod
    if isinstance(func, (staticmethod, classmethod)):
        func = func.__func__

    # For matrix kind, call once now to validate and cache the matrix.
    if kind == "matrix":
        try:
            matrix = func()
        except TypeError as error:
            if "missing 1 required positional argument: 'self'" in str(error):
                # Instance method: defer the call until an instance is supplied.
                def instance_wrapper(instance):
                    return func(instance)

                _mark(instance_wrapper, name, kind, True)
                _global_operators[name] = instance_wrapper
                _operator_metadata[name] = {"kind": kind, "requires_instance": True}
                _mark(func, name, kind, True)
                return func
            raise

        if not isinstance(matrix, np.ndarray):
            raise ValueError(f"Operator function '{name}' must return a numpy array")

        _mark(func, name, kind, False)
        func._operator_matrix = matrix  # type: ignore[attr-defined]
        _global_operators[name] = func
        _operator_metadata[name] = {"kind": kind, "requires_instance": False}
        return func

    # unary_pre / unary_post / binary / ternary / nullary / callable
    code = getattr(func, "__code__", None)
    requires_instance = bool(code and "self" in code.co_varnames)
    _mark(func, name, kind, requires_instance)
    _global_operators[name] = func
    _operator_metadata[name] = {"kind": kind, "requires_instance": requires_instance}

    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    _mark(wrapper, name, kind, requires_instance)
    return wrapper


class operator:
    """Register a callable under an arbitrary name.

    The name can be any string: capital letters, symbols, unicode, etc. Works as
    a function, method, or class decorator (or with any Python callable).

    Examples::

        @operator(kind="matrix", name="h")
        def h_MAT(): ...

        @operator.binary("<->")
        def swap(control, target): ...

        @operator.unary_post("†")
        def adjoint(qubit): ...

    ``kind`` may be omitted (default ``"callable"``).
    """

    def __init__(self, name: str, kind: OperatorKind = "callable"):
        self.name = name
        self.kind = kind

    def __call__(self, func: Callable) -> Callable:
        return _register(self.name, self.kind, func)

    # ─── Convenience constructors ───

    @classmethod
    def nullary(cls, name: str) -> "operator":
        return cls(name, "nullary")

    @classmethod
    def unary_pre(cls, name: str) -> "operator":
        return cls(name, "unary_pre")

    @classmethod
    def unary_post(cls, name: str) -> "operator":
        return cls(name, "unary_post")

    @classmethod
    def binary(cls, name: str) -> "operator":
        return cls(name, "binary")

    @classmethod
    def ternary(cls, name: str) -> "operator":
        return cls(name, "ternary")

    @classmethod
    def callable(cls, name: str) -> "operator":  # noqa: A003 - intentional API name
        return cls(name, "callable")

    @classmethod
    def matrix(cls, name: str) -> "operator":
        return cls(name, "matrix")


# ─── Public API ───

def get_global_operator(name: str):
    """Retrieve a registered operator by name.

    Returns ``(operator, kind)``; ``kind`` is one of :data:`OPERATOR_KINDS`, or
    ``None`` when the name is not registered.
    """
    op = _global_operators.get(name)
    if op is None:
        return None, None
    return op, getattr(op, "_kind", "callable")


def get_operator(name: str):
    """Alias for get_global_operator (backwards compatibility)."""
    return get_global_operator(name)


def get_operator_kind(name: str) -> str | None:
    """Get the canonical kind of a registered operator."""
    meta = _operator_metadata.get(name)
    return meta.get("kind") if meta else None


def get_operator_requires_instance(name: str) -> bool:
    """Check if an operator requires an instance (is a method)."""
    meta = _operator_metadata.get(name)
    return meta.get("requires_instance", False) if meta else False


def get_global_operator_matrix(name: str):
    """Retrieve a registered matrix operator by name."""
    op, kind = get_global_operator(name)
    if kind == "matrix" and callable(op):
        try:
            return op()
        except TypeError:
            # Instance method: can't call without an instance.
            return op
    return None


def get_operator_matrix(name: str):
    """Alias for get_global_operator_matrix (backwards compatibility)."""
    return get_global_operator_matrix(name)


def list_operators(kind: str | None = None) -> list[str]:
    """List registered operator names, optionally filtered to one `kind`."""
    if kind is None:
        return list(_global_operators.keys())
    wanted = normalize_kind(kind)
    return [name for name in _global_operators if _operator_metadata.get(name, {}).get("kind") == wanted]


def clear_operators():
    """Clear all registered operators (useful for testing)."""
    _global_operators.clear()
    _operator_metadata.clear()


def apply_operator(name: str, *args, **kwargs) -> Any:
    """Execute a registered operator by name with the given arguments.

    Raises:
        ValueError: If the operator is unknown or needs an instance.
    """
    op, kind = get_operator(name)
    if op is None:
        raise ValueError(f"Unknown operator: '{name}'")

    meta = _operator_metadata.get(name, {})
    if meta.get("kind") == "matrix" and callable(op):
        try:
            return op()
        except TypeError as error:
            if "missing 1 required positional argument: 'self'" in str(error):
                raise ValueError(f"Operator '{name}' is a method and requires an instance") from error
            raise

    return op(*args, **kwargs)


__all__ = [
    "operator",
    "OperatorKind",
    "OPERATOR_KINDS",
    "normalize_kind",
    "get_operator",
    "get_global_operator",
    "get_global_operator_matrix",
    "get_operator_kind",
    "get_operator_requires_instance",
    "get_operator_matrix",
    "list_operators",
    "clear_operators",
    "apply_operator",
]

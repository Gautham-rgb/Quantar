"""
Generic operator registry and @operator decorator.

Completely independent of quantum code - can be used for any purpose.
"""

from typing import Callable, Any, Dict, List, Optional
from functools import wraps
import numpy as np


# Global operator registry
_global_operators: Dict[str, Callable] = {}
_operator_metadata: Dict[str, Dict[str, Any]] = {}


def operator(name: str, kind: str = "callable"):
    """Decorator to register a custom operator with an arbitrary name.

    The name can be any string: capital letters, symbols, unicode, etc.
    Works as a function decorator, method decorator, class decorator,
    or with any Python class.

    Args:
        name: Operator name (any string: "H", "X", "CNOT", "<->", "⟂", etc.)
        kind: Operator kind - "matrix" (returns np.ndarray), 
              "unary" (callable taking one argument),
              "binary" (callable taking two arguments),
              "callable" (generic callable, default)

    Usage:
        @operator("H")
        def hadamard():
            return np.array([[1, 1], [1, -1]]) / np.sqrt(2)

        @operator("DOUBLE", kind="unary")
        def double(x):
            return x * 2

        @operator("CONCAT", kind="binary")
        def concat(a, b):
            return str(a) + str(b)

        @operator("PRINT", kind="callable")
        def print_state(*args, **kwargs):
            print(*args, **kwargs)

        class MyClass:
            @operator("MY_OP", kind="binary")
            def my_op(self, a, b):
                return a + b
    """
    if not isinstance(name, str) or not name:
        raise ValueError("Operator name must be a non-empty string")
    
    valid_kinds = {"matrix", "unary", "binary", "callable"}
    if kind not in valid_kinds:
        raise ValueError(f"Invalid kind: {kind}. Must be one of {valid_kinds}")

    def decorator(func: Callable) -> Callable:
        if not isinstance(name, str) or not name:
            raise ValueError("Operator name must be a non-empty string")
        
        # Handle staticmethod/classmethod
        if isinstance(func, (staticmethod, classmethod)):
            func = func.__func__
        
        # For matrix kind, try to call to get matrix immediately
        if kind == "matrix":
            try:
                matrix = func()
            except TypeError as e:
                if "missing 1 required positional argument: 'self'" in str(e):
                    # Instance method - wrap to be called with instance later
                    def wrapper(instance):
                        return func(instance)
                    wrapper._operator_name = name
                    wrapper._is_operator = True
                    wrapper._requires_instance = True
                    wrapper._kind = "matrix"
                    _global_operators[name] = wrapper
                    _operator_metadata[name] = {"kind": "matrix", "requires_instance": True}
                    func._operator_name = name
                    func._is_operator = True
                    func._requires_instance = True
                    func._kind = "matrix"
                    return func
                raise
            
            if not isinstance(matrix, np.ndarray):
                raise ValueError(f"Operator function '{name}' must return a numpy array")
            
            _global_operators[name] = matrix
            func._operator_name = name
            func._is_operator = True
            func._requires_instance = False
            func._kind = "matrix"
            _operator_metadata[name] = {"kind": "matrix", "requires_instance": False}
            return func
        
        # For unary/binary/callable kinds, store the callable directly
        func._operator_name = name
        func._is_operator = True
        func._requires_instance = "self" in func.__code__.co_varnames
        func._kind = kind
        
        _global_operators[name] = func
        _operator_metadata[name] = {
            "kind": kind,
            "requires_instance": func._requires_instance
        }
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        
        wrapper._operator_name = name
        wrapper._is_operator = True
        wrapper._requires_instance = func._requires_instance
        wrapper._kind = kind
        
        return wrapper
    
    return decorator


# ─── Public API ───

_global_operators: Dict[str, Callable] = {}
_operator_metadata: Dict[str, Dict[str, Any]] = {}


def get_global_operator(name: str):
    """Retrieve a globally registered operator by name.
    
    Returns a tuple of (operator, kind) where kind is one of:
    'matrix', 'unary', 'binary', 'callable', or None if not found.
    """
    op = _global_operators.get(name)
    if op is None:
        return None, None
    kind = getattr(op, '_kind', 'callable')
    return op, kind


def get_operator(name: str):
    """Alias for get_global_operator (backwards compatibility)."""
    return get_global_operator(name)


def get_operator_kind(name: str) -> Optional[str]:
    """Get the kind of a registered operator."""
    meta = _operator_metadata.get(name)
    return meta.get("kind") if meta else None


def get_operator_requires_instance(name: str) -> bool:
    """Check if an operator requires an instance (is a method)."""
    meta = _operator_metadata.get(name)
    return meta.get("requires_instance", False) if meta else False


def get_global_operator_matrix(name: str):
    """Retrieve a globally registered matrix operator by name."""
    op, kind = get_global_operator(name)
    if kind == 'matrix':
        return op
    return None


def get_operator_matrix(name: str):
    """Alias for get_global_operator_matrix (backwards compatibility)."""
    return get_global_operator_matrix(name)


def list_operators() -> List[str]:
    """List all globally registered operator names."""
    return list(_global_operators.keys())


def clear_operators():
    """Clear all registered operators (useful for testing)."""
    _global_operators.clear()
    _operator_metadata.clear()


def apply_operator(name: str, *args, **kwargs) -> Any:
    """Execute a registered operator by name with given arguments.
    
    Args:
        name: Operator name
        *args: Positional arguments to pass to the operator
        **kwargs: Keyword arguments to pass to the operator
    
    Returns:
        The result of calling the operator.
    
    Raises:
        ValueError: If operator not found or arguments mismatch.
    """
    op, kind = get_operator(name)
    if op is None:
        raise ValueError(f"Unknown operator: '{name}'")
    
    # For matrix kind, we might need to call it
    meta = _operator_metadata.get(name, {})
    if meta.get("kind") == "matrix" and callable(op):
        try:
            return op()
        except TypeError as e:
            if "missing 1 required positional argument: 'self'" in str(e):
                raise ValueError(f"Operator '{name}' is a method and requires an instance")
            raise
    
    return op(*args, **kwargs)


# ─── Public API ───

__all__ = [
    "operator",
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
"""Runtime bridge for executing lowered Quantar ASTs.

The language core stays independent of the quantum package.  Lowered
``QuantumOp`` nodes call ``__quantar_apply__(name, *operands)``; this module
provides that function and a ready-made globals dictionary for ``exec``.
"""

from __future__ import annotations

from typing import Any

from lang_systems.operators import (
    apply_operator,
    get_global_operator,
    get_global_operator_matrix,
)


def _load_quantum_operators() -> None:
    """Import the quantum gate registrations on first use."""
    try:
        import quantum.gate_matrix  # noqa: F401
    except ImportError:
        return


def _common_computer(operands: tuple[Any, ...]):
    computer = getattr(operands[0], "computer", None)
    if computer is None:
        raise TypeError("gate operators require qubit operands")
    if any(getattr(operand, "computer", None) is not computer for operand in operands):
        raise ValueError("all qubits in a gate operation must belong to the same computer")
    return computer


def apply_quantar_operator(name: str, *operands: Any) -> Any:
    """Apply a registered operator or gate to its operands."""
    operator, kind = get_global_operator(name)
    if operator is None:
        _load_quantum_operators()
        operator, kind = get_global_operator(name)

    if operator is None:
        raise ValueError(f"Unknown operator: {name!r}")

    if kind == "matrix":
        matrix = get_global_operator_matrix(name)
        if matrix is None:
            raise ValueError(f"Operator {name!r} is not a usable matrix")
        if not operands:
            return matrix
        computer = _common_computer(operands)
        return computer.apply_gate(matrix, target_qubits=list(operands))

    return apply_operator(name, *operands)


def runtime_globals() -> dict[str, Any]:
    """Return globals that make a lowered Quantar AST executable."""
    return {"__quantar_apply__": apply_quantar_operator}


__all__ = ["apply_quantar_operator", "runtime_globals"]

"""Import OpenQASM 3 into a :class:`QuantumComputer`.

Two passes over the parsed program:

1. :class:`RegisterCollector` gathers every ``qubit[n] name;`` declaration (so we
   know the total width and where each named register starts) and every
   ``gate name(a) q { ... }`` definition.
2. :class:`Visitor` walks the gates and applies them, expanding any custom gate
   by replaying its body against the actual operand qubits.

Because the simulator's ``apply_gate`` takes *control* qubits separately, a gate
name that already carries controls (``cx``, ``ccx``, ``cswap``, ...) and the
``ctrl``/``negctrl`` modifiers are both folded into that control list; the
remaining operands are the targets. This mirrors how ``to_openqasm3`` writes
them, so a parse/export round-trip is the identity.
"""

from __future__ import annotations

import operator
from typing import Any, Callable

import numpy as np

import openqasm3 as oqa3
from openqasm3.ast import GateModifierName

from . import gate_matrix as gm
from .qubit_abs import QuantumComputer

ast = oqa3.ast


# ─── Expression evaluation ───
# OpenQASM gate arguments are expressions (e.g. ``pi/2``, ``2*theta``). We only
# need the numeric ones here, but they must be evaluated before building a gate.

_CONSTANTS = {
    "pi": np.pi,
    "π": np.pi,
    "tau": 2 * np.pi,
    "τ": 2 * np.pi,
    "e": np.e,
    "E": np.e,
}

_BINARY_OPERATORS: dict[str, Callable[[Any, Any], Any]] = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
    "//": operator.floordiv,
    "%": operator.mod,
    "**": operator.pow,
    "<<": operator.lshift,
    ">>": operator.rshift,
    "&": operator.and_,
    "|": operator.or_,
    "^": operator.xor,
    "&&": lambda a, b: bool(a) and bool(b),
    "||": lambda a, b: bool(a) or bool(b),
    ".": lambda a, b: complex(a, b),
}

_UNARY_OPERATORS: dict[str, Callable[[Any], Any]] = {
    "-": operator.neg,
    "+": operator.pos,
    "~": operator.invert,
    "!": operator.not_,
}

_MATH_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "asin": np.arcsin,
    "acos": np.arccos,
    "atan": np.arctan,
    "arcsin": np.arcsin,
    "arccos": np.arccos,
    "arctan": np.arctan,
    "sqrt": np.sqrt,
    "exp": np.exp,
    "ln": np.log,
    "log": np.log,
    "abs": np.abs,
    "floor": np.floor,
    "ceiling": np.ceil,
    "pow": pow,
    "mod": operator.mod,
}


def evaluate(node: ast.QASMNode, bindings: dict[str, float] | None = None) -> Any:
    """Evaluate a numeric OpenQASM expression node to a Python integer.

    `bindings` resolves formal parameter names while expanding a custom gate.
    """
    if isinstance(node, ast.IntegerLiteral):
        return node.value
    if isinstance(node, ast.FloatLiteral):
        return node.value
    if isinstance(node, ast.BooleanLiteral):
        return node.value
    if isinstance(node, ast.ImaginaryLiteral):
        return complex(0, node.value)
    if isinstance(node, ast.Identifier):
        if bindings is not None and node.name in bindings:
            return bindings[node.name]
        if node.name in _CONSTANTS:
            return _CONSTANTS[node.name]
        raise NameError(f"Unknown constant '{node.name}' in OpenQASM expression.")
    if isinstance(node, ast.UnaryExpression):
        return _UNARY_OPERATORS[node.op.name](evaluate(node.expression, bindings))
    if isinstance(node, ast.BinaryExpression):
        return _BINARY_OPERATORS[node.op.name](
            evaluate(node.lhs, bindings), evaluate(node.rhs, bindings)
        )
    if isinstance(node, ast.FunctionCall):
        function = _MATH_FUNCTIONS.get(node.name.name)
        if function is None:
            raise NameError(f"Unknown function '{node.name.name}'.")
        return function(*(evaluate(argument, bindings) for argument in node.arguments))
    raise TypeError(f"Cannot evaluate OpenQASM node {type(node).__name__}.")


# ─── Gate resolution ───
# ``u``/``U`` is the OpenQASM 3 builtin arbitrary unitary; the rest of the
# parameterised rotations live in gate_matrix already.

def u_MAT(theta: float, phi: float, lam: float) -> np.ndarray:
    """The OpenQASM 3 ``U(theta, phi, lambda)`` single-qubit unitary."""
    cos = np.cos(theta / 2)
    sin = np.sin(theta / 2)
    return np.array(
        [
            [cos, -np.exp(1j * lam) * sin],
            [np.exp(1j * phi) * sin, np.exp(1j * (phi + lam)) * cos],
        ],
        dtype=complex,
    )


# Names that already carry their own control(s); value is (base gate, #controls).
_CONTROLLED_BASES: dict[str, tuple[str, int]] = {
    "cx": ("x", 1),
    "cy": ("y", 1),
    "cz": ("z", 1),
    "cp": ("p", 1),
    "ccx": ("x", 2),
    "cswap": ("swap", 1),
}

_EXTRA_GATES: dict[str, Callable[..., np.ndarray]] = {
    "rx": gm.rx_MAT,
    "ry": gm.ry_MAT,
    "rz": gm.rz_MAT,
    "u": u_MAT,
    "U": u_MAT,
}


def resolve_gate(name: str) -> tuple[Callable[..., np.ndarray], int]:
    """Return (matrix factory, number of controls implied by the name)."""
    if name in _CONTROLLED_BASES:
        base, controls = _CONTROLLED_BASES[name]
        factory, _ = resolve_gate(base)
        return factory, controls

    for gate_name, factory in gm._BUILTIN_GATE_NAMES:
        if gate_name == name:
            return factory, 0
    for gate_name, factory in gm._BUILTIN_ARG_GATE_NAMES:
        if gate_name == name:
            return factory, 0
    if name in _EXTRA_GATES:
        return _EXTRA_GATES[name], 0

    raise ValueError(f"Unknown OpenQASM gate '{name}'.")


def _matrix_power(matrix: np.ndarray, power: float) -> np.ndarray:
    """Raise a (unitary) matrix to an integer or fractional power."""
    if float(power).is_integer():
        return np.linalg.matrix_power(matrix, int(power))
    # Fractional power via eigendecomposition (fine for unitaries).
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    return eigenvectors @ np.diag(eigenvalues**power) @ np.linalg.inv(eigenvectors)


# ─── Pass 1: registers and gate definitions ───

class RegisterCollector(oqa3.visitor.QASMVisitor):
    """Collect qubit declarations and custom gate definitions."""

    def __init__(self) -> None:
        self.num_qubits = 0
        # register name -> (offset, size)
        self.qubit_reg: dict[str, tuple[int, int]] = {}
        # custom gate name -> QuantumGateDefinition
        self.gate_defs: dict[str, ast.QuantumGateDefinition] = {}

    def visit_QubitDeclaration(self, node: ast.QubitDeclaration) -> None:
        size = 1 if node.size is None else int(evaluate(node.size))
        offset = self.num_qubits
        self.num_qubits += size
        self.qubit_reg[node.qubit.name] = (offset, size)

    def visit_QuantumGateDefinition(self, node: ast.QuantumGateDefinition) -> None:
        self.gate_defs[node.name.name] = node


# ─── Pass 2: gates ───

class Visitor(oqa3.visitor.QASMVisitor):
    """Apply each statement of an OpenQASM program to a QuantumComputer."""

    def __init__(
        self,
        qc: QuantumComputer,
        qubit_reg: dict[str, tuple[int, int]],
        gate_defs: dict[str, ast.QuantumGateDefinition] | None = None,
        bindings: dict[str, float] | None = None,
    ) -> None:
        self.qc = qc
        self.qubit_reg = qubit_reg
        self.gate_defs = gate_defs or {}
        self.bindings = bindings

    def _indices(self, operand: ast.Identifier | ast.IndexedIdentifier) -> list[int]:
        """Resolve a gate operand to global qubit index(es).

        A bare identifier names a whole register (used for ``reset``/``barrier``
        and for formal qubits inside a custom-gate body); an indexed identifier
        names one qubit.
        """
        if isinstance(operand, ast.Identifier):
            name = operand.name
            return list(range(*_span(self.qubit_reg, name)))

        name = operand.name.name
        offset, size = self.qubit_reg[name]
        if not operand.indices:  # bare register reference
            return list(range(offset, offset + size))
        local = int(evaluate(operand.indices[0][0], self.bindings)) #type: ignore
        if not 0 <= local < size:
            raise IndexError(f"Qubit index {local} is out of range for register '{name}'[{size}].")
        return [offset + local]

    def _index(self, operand: ast.Identifier | ast.IndexedIdentifier) -> int:
        """Resolve a gate operand to a single global qubit index."""
        indices = self._indices(operand)
        if len(indices) != 1:
            raise ValueError("A gate operand must name exactly one qubit.")
        return indices[0]

    def visit_QuantumGate(self, node: ast.QuantumGate) -> None:
        name = node.name.name

        if name in self.gate_defs and name not in _EXTRA_GATES:
            self._expand_definition(self.gate_defs[name], node)
            return

        factory, name_controls = resolve_gate(name)

        arguments = [evaluate(argument, self.bindings) for argument in node.arguments]
        matrix = np.asarray(factory(*arguments), dtype=complex)

        # `ctrl`/`negctrl` modifiers, in operand order; `inv`/`pow` act on the matrix.
        modifier_controls: list[bool] = []  # True == negctrl (control on |0>)
        inverted = False
        power: float | None = None
        for modifier in node.modifiers:
            kind = modifier.modifier
            if kind is GateModifierName.ctrl or kind is GateModifierName.negctrl:
                count = 1 if modifier.argument is None else int(evaluate(modifier.argument, self.bindings))
                modifier_controls.extend([kind is GateModifierName.negctrl] * count)
            elif kind is GateModifierName.inv:
                inverted = True
            elif kind is GateModifierName.pow:
                power = evaluate(modifier.argument, self.bindings) #type: ignore

        if inverted:
            matrix = matrix.conj().T
        if power is not None:
            matrix = _matrix_power(matrix, power)

        operands = [self._index(qubit) for qubit in node.qubits]
        # Operands read: [modifier controls..., name's own controls..., targets...].
        num_controls = len(modifier_controls) + name_controls
        controls = operands[:num_controls]
        targets = operands[num_controls:]

        if not targets:
            raise ValueError(f"Gate '{name}' has no target qubit.")

        # Remember the name/params for export, unless `inv`/`pow`
        # changed the matrix (then the matrix must speak for itself).
        record_name = None if (inverted or power is not None) else name
        record_params = arguments if record_name is not None else None

        # negctrl means "control on |0>": conjugate those controls with X.
        negated = [controls[i] for i, is_negated in enumerate(modifier_controls) if is_negated]
        for qubit in negated:
            self.qc.apply_gate(gm.px_MAT(), [qubit])

        self.qc.apply_gate(
            matrix,
            targets, #type: ignore
            control_qubits=controls, #type: ignore
            qasm_name=record_name,
            qasm_params=record_params,
        )

        for qubit in negated:
            self.qc.apply_gate(gm.px_MAT(), [qubit])

    def _expand_definition(
        self, definition: ast.QuantumGateDefinition, call: ast.QuantumGate
    ) -> None:
        """Replay a custom gate's body onto the call's actual operands."""
        formals = [qubit.name for qubit in definition.qubits]
        if len(call.qubits) != len(formals):
            raise ValueError(
                f"Gate '{definition.name.name}' takes {len(formals)} qubit(s), "
                f"got {len(call.qubits)}."
            )

        actual_params = [evaluate(argument, self.bindings) for argument in call.arguments]
        if len(actual_params) != len(definition.arguments):
            raise ValueError(
                f"Gate '{definition.name.name}' takes {len(definition.arguments)} argument(s), "
                f"got {len(actual_params)}."
            )
        bindings = dict(zip((p.name for p in definition.arguments), actual_params))

        operand_indices = [self._index(qubit) for qubit in call.qubits]
        child_reg = {formal: (index, 1) for formal, index in zip(formals, operand_indices)}

        body_visitor = Visitor(self.qc, child_reg, self.gate_defs, bindings)
        for statement in definition.body:
            body_visitor.visit(statement)

    def visit_QuantumGateDefinition(self, node: ast.QuantumGateDefinition) -> None:
        """Gate definitions are expanded at their call sites, not executed here."""

    def visit_QuantumReset(self, node: ast.QuantumReset) -> None:
        qubits = node.qubits if isinstance(node.qubits, list) else [node.qubits]
        for qubit in qubits:
            for index in self._indices(qubit):
                self._reset(index)

    def visit_QuantumBarrier(self, node: ast.QuantumBarrier) -> None:
        """Barriers are a scheduling hint; the simulator has no timing."""

    def visit_QuantumMeasurementStatement(self, node: ast.QuantumMeasurementStatement) -> None:
        if node.measure is not None:
            self.qc[self._index(node.measure.qubit)].measure()

    def _reset(self, index: int) -> None:
        """Set a qubit to |0> (OpenQASM ``reset``): measure it, then flip if it read 1."""
        qubit = self.qc[index]
        outcome = qubit.measure()  # collapses onto the measured branch
        qubit.is_measured = False  # reset revives the qubit for further gates
        if outcome:
            self.qc.apply_gate(gm.px_MAT(), [index], log_history=False)


def _span(qubit_reg: dict[str, tuple[int, int]], name: str) -> tuple[int, int]:
    if name not in qubit_reg:
        raise KeyError(f"Unknown qubit register '{name}'.")
    offset, size = qubit_reg[name]
    return offset, offset + size


# ─── Entry points ───

def load_program(program: ast.Program) -> QuantumComputer:
    """Build a QuantumComputer from an already-parsed OpenQASM 3 program."""
    collector = RegisterCollector()
    collector.visit(program)

    computer = QuantumComputer(collector.num_qubits)
    Visitor(computer, collector.qubit_reg, collector.gate_defs).visit(program)
    return computer


def parse_qasm(source: str) -> QuantumComputer:
    """Parse OpenQASM 3 source and return a QuantumComputer."""
    return load_program(oqa3.parse(source))


load = parse_qasm

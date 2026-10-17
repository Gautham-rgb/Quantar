import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quantum.qubit_abs import QuantumComputer

class Gate:
    """One recorded gate application: a matrix plus the qubits it touches.

    `inverted` marks a gate that was applied backwards (see `QuantumComputer.
    uncompute`), so the OpenQASM 3 export can write it as `inv @ ...`.
    """

    def __init__(self, matrix, controls: list[int], targets: list[int],
                 name: str | None = None, params: list[float] | None = None,
                 inverted: bool = False):
        self.matrix = np.asarray(matrix, dtype=complex)
        self.controls = list(controls)
        self.targets = list(targets)
        self.name = name
        self.params = list(params) if params else []
        self.inverted = inverted

# Verify unitarity (optional check)
        if not np.allclose(self.matrix.conj().T @ self.matrix, np.eye(self.matrix.shape[0]), atol=1e-9):
            deviation = self.matrix.conj().T @ self.matrix
            raise ValueError(f"Gate matrix is not unitary (max deviation = {deviation})")

    def __repr__(self):
        label = self.name if self.name else f"{self.matrix.shape[0]}x{self.matrix.shape[1]} matrix"
        direction = "inv @ " if self.inverted else ""
        qubits = ", ".join([str(i) for i in self.controls + self.targets])
        return f"<{direction}{label} on [{qubits}]>"

    def inverse(self) -> "Gate":
        """The same operation run backwards (the adjoint of a unitary)."""
        return Gate(
            self.matrix.conj().T,
            self.controls,
            self.targets,
            self.name,
            self.params,
            not self.inverted,
        )

    def remap(self, mapping: list[int], extra_controls: list[int] = []) -> "Gate":
        """The same gate with its qubits rewritten through `mapping`.

        This is how a program built on one register is replayed onto another
        (see `CircuitGate`); `extra_controls` are appended, which is how a
        control on the outer gate conditions every gate inside it.
        """
        return Gate(
            self.matrix,
            [mapping[index] for index in self.controls] + list(extra_controls),
            [mapping[index] for index in self.targets],
            self.name,
            self.params,
            self.inverted,
        )

class CompositeGate:
    """A gate built from other operations rather than from one matrix.

    `body(qc, targets, controls)` applies whatever gates it needs; `name` and
    `params` describe it for OpenQASM 3 (its inner `apply_gate` calls are what
    actually end up in `op_hist`).
    """

    def __init__(self, name: str, body, params: list[float] | None = None):
        self.name = name
        self.body = body
        self.params = list(params) if params else []

    def __repr__(self):
        return f"<composite {self.name}>"

    def __call__(self, qc: "QuantumComputer", targets: list[int], controls: list[int]):
        self.body(qc, targets, controls)

class CircuitGate:
    """A whole circuit usable as one gate: a program built on its own register.

    `CircuitGate.build(name, width, builder)` runs `builder(qc, qubits)` on a
    scratch register of `width` qubits and keeps whatever it recorded. Applying
    the result replays that program onto the parent's target qubits, so
    `qc.apply(qft_gate(4), qc[0:4])` works exactly like running the QFT circuit
    by hand -- and `op_hist` still holds the individual gates, so `uncompute`
    and `to_openqasm3` see the real operations rather than an opaque block.
    """

    def __init__(self, name: str, width: int, program: list[Gate],
                 params: list[float] | None = None):
        self.name = name
        self.width = width
        self.program = list(program)
        self.params = list(params) if params else []

    def __repr__(self):
        return f"<circuit {self.name}: {len(self.program)} gate(s) on {self.width} qubit(s)>"

    @classmethod
    def build(cls, name: str, width: int, builder, params: list[float] | None = None) -> "CircuitGate":
        """Record `builder(qc, qubits)` on a scratch register and keep the program."""
        from quantum.qubit_abs import QuantumComputer
        scratch = QuantumComputer(width, name=name)
        builder(scratch, scratch.qubits)
        return cls(name, width, scratch.op_hist, params)

    def inverse(self) -> "CircuitGate":
        """The same circuit backwards: every gate inverted, in reverse order."""
        return CircuitGate(
            self.name,
            self.width,
            [gate.inverse() for gate in reversed(self.program)],
            self.params,
        )

    def __call__(self, qc: "QuantumComputer", targets: list[int], controls: list[int]):
        targets = [qc.index_of(target) for target in targets]
        if len(targets) != self.width:
            raise ValueError(
                f"'{self.name}' acts on {self.width} qubit(s), got {len(targets)} target(s)."
            )
        extra = [qc.index_of(control) for control in controls]
        qc._apply_program([gate.remap(targets, extra) for gate in self.program])


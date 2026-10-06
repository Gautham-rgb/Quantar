"""Export a recorded gate history as OpenQASM 3.

The simulator records structure (which gate, on which qubits, with which
parameters), so this module only has to name each gate and assemble the text:

* a gate OpenQASM 3 knows by name is emitted directly (`h q[0];`);
* controls become control modifiers (`ctrl(2) @ x q[0], q[1], q[2];`);
* gates that already take a control (`cp`) keep it in their operand list;
* any other single-qubit unitary becomes `U(theta, phi, lambda) q[0];`.

That covers Grover's search completely, since its oracle is a multi-controlled
X and its diffuser a multi-controlled Z.
"""

import numpy as np

import quantum.gate_matrix as gm

_EPSILON = 1e-9


def _euler_angles(matrix: np.ndarray) -> tuple[float, float, float]:
    """Return (theta, phi, lambda) so that `U(theta, phi, lambda)` equals `matrix`."""
    top_left, bottom_left = matrix[0, 0], matrix[1, 0]
    theta = 2 * np.arccos(np.clip(abs(top_left), 0.0, 1.0))
    phi = np.angle(bottom_left) - np.angle(top_left)
    lam = np.angle(matrix[0, 1]) - np.angle(top_left)

    if abs(top_left) < _EPSILON:
        phi = 0.0
        lam = (np.angle(matrix[1, 1]) - np.angle(bottom_left)) % (2 * np.pi)

    return float(theta), float(phi % (2 * np.pi)), float(lam % (2 * np.pi))


def _modifiers(name: str, gate) -> str:
    """Gate modifiers for a recorded gate: controls, plus `inv @` if reversed.

    `ctrl @` is skipped for gates whose name already carries a control
    (`cp` already takes a control qubit, so `ctrl @ cp` would double it).
    """
    if gm.name_is_controlled(name, len(gate.controls)):
        controls = ""
    elif len(gate.controls) == 1:
        controls = "ctrl @ "
    else:
        controls = f"ctrl({len(gate.controls)}) @ " if gate.controls else ""

    return controls + ("inv @ " if gate.inverted else "")


def _gate_call(gate) -> str:
    """One OpenQASM 3 statement for a recorded gate, including its modifiers."""
    matrix = np.asarray(gate.matrix, dtype=complex)
    num_controls, num_targets = len(gate.controls), len(gate.targets)

    name = gate.name or gm.named_gate(matrix, num_targets)
    params = list(gate.params)

    if name is None and num_controls == 0 and num_targets == 1:
        name, params = "U", list(_euler_angles(matrix))

    if name is None:
        raise ValueError(
            f"No OpenQASM 3 name for a {num_controls}-control {num_targets}-target gate. "
            "Apply it with qasm_name= (and qasm_params=) so it can be written out."
        )

    if name == "U":
        arguments = "(theta, phi, lambda) "
    else:
        arguments = "".join(f"{value!r}, " for value in params)

    operands = ", ".join(f"q[{index}]" for index in gate.controls + gate.targets)
    return f"{_modifiers(name, gate)}{name} {arguments}{operands};"


def to_openqasm3(num_qubits: int, circuit_name: str, gates: list) -> str:
    """Render gates (recorded by QuantumComputer.op_hist) as an OpenQASM 3 program."""
    body = [_gate_call(gate) for gate in gates]
    header = [
        "OPENQASM 3.0;",
        'include "stdgates.inc";',
        "",
        f"// {circuit_name}: {num_qubits} qubit(s)",
        f"qubit[{num_qubits}] q;",
        "",
    ]
    return "\n".join(header + body) + "\n"

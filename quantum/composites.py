"""Composite gates: operations assembled out of other operations.

These are the building blocks for circuits that are easier to state than to
expand gate by gate — the QFT is `h` plus `crk` pairs, for instance.
"""

import numpy as np

import quantum.gate_matrix as gm
from quantum.qubit_abs import CompositeGate


def crk(k: int, inverse: bool = False) -> CompositeGate:
    """The CR_k gate: controlled phase shift by 2*pi / 2**k.

    `crk(1)` is the plain controlled-Z, `crk(2)` the controlled-S, and so on.
    With `inverse=True` the phase is negated, which is what turns the QFT
    circuit into the inverse QFT.

    Used as: `qc.apply(crk(3), [target], control_qubits=[control])`.
    """
    angle = (2 * np.pi) / (1 << k) * (-1 if inverse else 1)

    def body(qc, targets, controls):
        qc.apply_gate(
            gm.p_MAT(angle),
            target_qubits=targets,
            control_qubits=controls,
            qasm_name="cp",
            qasm_params=[float(angle)],
        )

    return CompositeGate(f"{'inv_' if inverse else ''}crk{k}", body, [float(angle)])

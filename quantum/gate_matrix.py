"""Gate matrices, plus the names OpenQASM 3 gives them.

Every matrix is a unitary gate matrix. The name table at the bottom lets a recorded operation be
written out as a readable OpenQASM 3 `gate` statement instead of a raw matrix.
"""

import numpy as np

h_MAT = np.array([[1, 1],
                  [1, -1]], dtype=complex) / np.sqrt(2)

px_MAT = np.array([[0, 1],
                   [1, 0]], dtype=complex)

py_MAT = np.array([[0, -1j],
                   [1j, 0]], dtype=complex)

pz_MAT = np.array([[1, 0],
                   [0, -1]], dtype=complex)

i_MAT = np.eye(2, dtype=complex)

s_MAT = np.array([[1, 0],
                  [0, 1j]], dtype=complex)

sdg_MAT = np.array([[1, 0],
                    [0, -1j]], dtype=complex)

t_MAT = np.array([[1, 0],
                  [0, np.exp(1j * np.pi / 4)]], dtype=complex)

tdg_MAT = np.array([[1, 0],
                    [0, np.exp(-1j * np.pi / 4)]], dtype=complex)

swap_MAT = np.array([[1, 0, 0, 0],
                     [0, 0, 1, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1]], dtype=complex)

iswap_MAT = np.array([[1, 0,  0,  0],
                      [0, 0, 1j, 0],
                      [0, 1j,  0,  0],
                      [0, 0,  0,  1]], dtype=complex)

swap_sqrt_MAT = np.array([[1, 0, 0, 0],
                          [0, 0.5 + 0.5j, 0.5 - 0.5j, 0],
                          [0, 0.5 - 0.5j, 0.5 + 0.5j, 0],
                          [0, 0, 0, 1]], dtype=complex)


def rx_MAT(angle: float) -> np.ndarray:
    c = np.cos(angle / 2)
    s = np.sin(angle / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def ry_MAT(angle: float) -> np.ndarray:
    c = np.cos(angle / 2)
    s = np.sin(angle / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def rz_MAT(angle: float) -> np.ndarray:
    c_minus = np.cos(angle / 2) - 1j * np.sin(angle / 2)
    c_plus = np.cos(angle / 2) + 1j * np.sin(angle / 2)
    return np.array([[c_minus, 0.0], [0.0, c_plus]], dtype=complex)


def p_MAT(angle: float) -> np.ndarray:
    """Phase gate: leaves |0> alone and phases |1> by `angle`.

    Used as the target of a controlled phase shift (CR_k) by listing a control
    qubit alongside it.
    """
    return np.array([[1, 0], [0, np.exp(1j * angle)]], dtype=complex)


def cp_MAT(angle: float) -> np.ndarray:
    """Controlled phase shift as a standalone 4x4 matrix: CR_k(control, target)."""
    phased = p_MAT(angle)
    return np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, phased[1, 1]]], dtype=complex)


def rk_MAT(k: int) -> np.ndarray:
    e_exp = (np.pi * 1j) / (1 << k - 1)
    return np.array([[1, 0],
                     [0, np.e ** e_exp]])


def qft_MAT(num_qubits: int) -> np.ndarray:
    N = 1 << num_qubits
    omega = np.exp(2j * np.pi / N)

    qft = np.zeros((N, N), dtype=complex)
    for r in range(N):
        for c in range(N):
            qft[r, c] = (omega ** (r * c)) / np.sqrt(N)

    return qft


def iqft_MAT(num_qubits: int) -> np.ndarray:
    return np.conj(qft_MAT(num_qubits)).T


# Uncontrolled gates OpenQASM 3 spells with a builtin name. Anything matched
# here also works under a `ctrl @` modifier, which is how multi-controlled
# X and Z (Grover's oracle and diffuser) are written out.
_BUILTIN_GATE_NAMES = [
    ("h", h_MAT),
    ("x", px_MAT),
    ("y", py_MAT),
    ("z", pz_MAT),
    ("s", s_MAT),
    ("sdg", sdg_MAT),
    ("t", t_MAT),
    ("tdg", tdg_MAT),
    ("id", i_MAT),
    ("swap", swap_MAT),
]


def named_gate(matrix: np.ndarray, num_targets: int) -> str | None:
    """Return the OpenQASM 3 name for this uncontrolled gate, or None."""
    matrix = np.asarray(matrix, dtype=complex)
    for name, builtin in _BUILTIN_GATE_NAMES:
        if builtin.shape == matrix.shape and np.allclose(builtin, matrix):
            return name
    return None


# Gates whose OpenQASM 3 name already carries its own control operand, so a
# `ctrl @` modifier must not be added on top of it.
_CONTROLLED_BUILTIN_ARITY = {"cx": 1, "cy": 1, "cz": 1, "cp": 1, "ccx": 2, "cswap": 2}


def name_is_controlled(name: str | None, num_controls: int) -> bool:
    """True when `name` already includes exactly `num_controls` of its own."""
    return name is not None and _CONTROLLED_BUILTIN_ARITY.get(name, -1) == num_controls

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
    c_plus  = np.cos(angle / 2) + 1j * np.sin(angle / 2)
    return np.array([[c_minus, 0.0], [0.0, c_plus]], dtype=complex)

swap_MAT = np.array([[1, 0, 0, 0],
                      [0, 0, 1, 0],
                      [0, 1, 0, 0],
                      [0, 0, 0, 1]], dtype=complex)

iswap_MAT = np.array([[1, 0,  0, 0],
                      [0, 0, 1j, 0],
                      [0, 1j,  0, 0],
                      [0, 0,  0, 1]], dtype=complex)

swap_sqrt_MAT = np.array([[1, 0, 0, 0],
                          [0, 0.5 + 0.5j, 0.5 - 0.5j, 0],
                          [0, 0.5 - 0.5j, 0.5 + 0.5j, 0],
                          [0, 0, 0, 1]], dtype=complex)
"""Shor's factoring algorithm.

The QFT is not built as one dense 2**n x 2**n matrix (which would need
terabytes for a realistic register). It is decomposed into Hadamards and
controlled phase shifts, `CR_k`, so each gate only touches the qubits it acts
on. For N = 91 that is 21 qubits, and factoring takes a couple of seconds.

`quantum_subroutine` still cheats the modular exponentiation classically: it
writes the state |k>|a**k mod N> directly instead of running the gates that
would produce it. That is the usual simulation shortcut and keeps the register
at 2**t amplitudes.
"""

from fractions import Fraction
from math import gcd

import numpy as np

import quantum.gate_matrix as gm
import quantum.qubit_abs as qa
from quantum.composites import crk


def qft_circuit(qc: qa.QuantumComputer, qubits: list):
    """Quantum Fourier transform built from Hadamards and `crk` operations.

    Matches the dense `gate_matrix.qft_MAT`, with `qubits[0]` the most
    significant bit. n Hadamards plus n(n-1)/2 controlled phase shifts.
    """
    for control in range(len(qubits) - 1, -1, -1):
        qc.apply(gm.h_MAT, [qubits[control]])
        for target in range(control):
            qc.apply(
                crk(control - target + 1),
                [qubits[target]],
                control_qubits=[qubits[control]],
            )


def iqft_circuit(qc: qa.QuantumComputer, qubits: list):
    """Inverse QFT: the QFT operations in reverse, each one inverted."""
    for control in range(len(qubits)):
        for target in range(control - 1, -1, -1):
            qc.apply(
                crk(control - target + 1, inverse=True),
                [qubits[target]],
                control_qubits=[qubits[control]],
            )
        qc.apply(gm.h_MAT, [qubits[control]])


def is_prime_fast(n: int) -> bool:
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    for i in range(5, int(np.sqrt(n)) + 1, 6):
        if n % i == 0 or n % (i + 2) == 0:
            return False
    return True


def is_perfect_power(N: int):
    if N <= 3:
        return False, None
    max_exponent = N.bit_length()

    for y in range(2, max_exponent + 1):
        low = 2
        high = N

        while low <= high:
            mid = (low + high) // 2
            val = mid ** y

            if val > N:
                high = mid - 1
            elif val < N:
                low = mid + 1
            else:
                return True, (mid, N // mid)

    return False, None


def quantum_subroutine(N: int, a: int) -> str:
    """Estimate the period of a mod N and return it as a t-bit string.

    Register 1 holds the superposition over k, register 2 holds a**k mod N.
    """
    m = N.bit_length()
    t = m << 1
    tot_qubits = m + t

    # |k>|a**k mod N> for every k, written straight into the state vector.
    exponents = np.arange(1 << t)
    register1 = exponents << m
    register2 = np.array([pow(a, int(k), N) for k in exponents])
    state = np.zeros((1 << tot_qubits, 1), dtype=complex)
    state[register1 | register2, 0] = 1.0 / np.sqrt(1 << t)

    qc = qa.QuantumComputer(tot_qubits, name="Shor1")
    qc.state_vector = state

    iqft_circuit(qc, qc[0:t])

    probabilities = np.abs(qc.state_vector.flatten()) ** 2
    highest_prob_index = int(np.argmax(probabilities))
    return f"{highest_prob_index:0{tot_qubits}b}"[:t]


def shors_alg(N: int):
    if is_prime_fast(N):
        return None

    is_power, factors = is_perfect_power(N)
    if is_power:
        return factors

    if N % 2 == 0:
        return 2, N // 2

    while True:
        a = np.random.randint(2, N)
        gcd_val = gcd(N, a)

        if gcd_val > 1:
            return gcd_val, N // gcd_val

        bitstr = quantum_subroutine(N, int(a))
        t = len(bitstr)
        y = int(bitstr, 2)

        if y == 0:
            continue

        phase = y / (1 << t)
        r = Fraction(phase).limit_denominator(N).denominator

        if r == 0 or r % 2 != 0:
            continue
        if pow(a, r // 2, N) == N - 1:
            continue

        val = pow(a, r // 2, N)
        factor1 = gcd(val - 1, N)
        factor2 = gcd(val + 1, N)

        if 1 < factor1 < N:
            return factor1, N // factor1

        if 1 < factor2 < N:
            return factor2, N // factor2


if __name__ == "__main__":
    import time

    for target in (91, 15, 21):
        start = time.time()
        print(target, "->", shors_alg(target), f"({time.time() - start:.2f}s)")

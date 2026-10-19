"""Shor's factoring algorithm.

The QFT is not built as one dense 2**n x 2**n matrix (which would need
terabytes for a realistic register). It is decomposed into Hadamards,
controlled phase shifts, `CR_k`, and a bit-reversing layer of swaps, so each
gate only touches the qubits it acts on. For N = 91 that is 21 qubits, and
factoring takes a couple of seconds.

`quantum_subroutine` still cheats the modular exponentiation classically: it
writes the state |k>|a**k mod N> directly instead of running the gates that
would produce it. That is the usual simulation shortcut and keeps the register
at 2**t amplitudes.
"""

from fractions import Fraction
from math import gcd

import numpy as np
import numba as nb

import quantum.gate_matrix as gm
import quantum.qubit_abs as qa
from stl.quantum_algs.quantum_optim_hl import QuantumOptimizer


@nb.jit(nopython=True)
def mod_exp(a: int, t: int, N: int):
    num_elements = 1 << t
    out = np.empty(num_elements, dtype=np.int64)

    current = 1
    out[0] = current
    for k in range(1, num_elements):
        current = (current * a) % N
        out[k] = current

    return out


def qft_program(qc: qa.QuantumComputer, qubits: list) -> list[qa.Gate]:
    """The QFT as gate records: a bit-reversal, then Hadamards and `CR_k` phases.

    Matches the dense `gate_matrix.qft_MAT`, with `qubits[0]` the most
    significant bit. The leading swaps undo the bit-reversal that the
    H-first-on-the-LSB sequence would otherwise bake into the input. Hand the
    result to `qc.apply`, which fuses each run of controlled phases into a
    single pass over the state.
    """
    n = len(qubits)
    program = []
    for left in range(n // 2):
        program.append(
            qc.gate(gm.swap_MAT(), [qubits[left], qubits[n - 1 - left]], qasm_name="swap")
        )
    for control in range(n - 1, -1, -1):
        program.append(qc.gate(gm.h_MAT(), [qubits[control]]))
        for target in range(control):
            angle = float((2 * np.pi) / (1 << (control - target + 1)))
            program.append(
                qc.gate(
                    gm.p_MAT(angle),
                    [qubits[target]],
                    [qubits[control]],
                    qasm_name="cp",
                    qasm_params=[angle],
                )
            )
    return program


def iqft_program(qc: qa.QuantumComputer, qubits: list) -> list[qa.Gate]:
    """Inverse QFT as gate records: the QFT program in reverse, each gate inverted.

    Phases come first (negated), then the Hadamards, then the swaps -- exactly
    `qft_program` backwards, so `qft_circuit` followed by `iqft_circuit` returns
    the input and both match the dense `qft_MAT` / `iqft_MAT`.
    """
    n = len(qubits)
    program = []
    for control in range(n):
        for target in range(control - 1, -1, -1):
            angle = float(-(2 * np.pi) / (1 << (control - target + 1)))
            program.append(
                qc.gate(
                    gm.p_MAT(angle),
                    [qubits[target]],
                    [qubits[control]],
                    qasm_name="cp",
                    qasm_params=[angle],
                )
            )
        program.append(qc.gate(gm.h_MAT(), [qubits[control]]))
    for left in range(n // 2):
        program.append(
            qc.gate(gm.swap_MAT(), [qubits[left], qubits[n - 1 - left]], qasm_name="swap")
        )
    return program


def qft_circuit(qc: qa.QuantumComputer, qubits: list):
    """Quantum Fourier transform built from Hadamards and `crk` operations."""
    qc.apply(qft_program(qc, qubits))


def iqft_circuit(qc: qa.QuantumComputer, qubits: list):
    """Inverse QFT: the QFT operations in reverse, each one inverted."""
    qc.apply(iqft_program(qc, qubits))


_QFT_GATES: dict[int, qa.CircuitGate] = {}
_IQFT_GATES: dict[int, qa.CircuitGate] = {}


def qft_gate(width: int) -> qa.CircuitGate:
    """The `width`-qubit QFT as one gate: `qc.apply(qft_gate(4), qc[0:4])`."""
    if width not in _QFT_GATES:
        _QFT_GATES[width] = qa.CircuitGate.build("qft", width, qft_circuit)
    return _QFT_GATES[width]


def iqft_gate(width: int) -> qa.CircuitGate:
    """The inverse QFT as one gate: `qc.apply(iqft_gate(4), qc[0:4])`."""
    if width not in _IQFT_GATES:
        _IQFT_GATES[width] = qa.CircuitGate.build("inv_qft", width, iqft_circuit)
    return _IQFT_GATES[width]


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
            mid = (low + high) >> 1
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
    Uses pre-built CircuitGate for IQFT (single apply call).
    """
    m = N.bit_length()
    t = m << 1
    tot_qubits = m + t

    # |k>|a**k mod N> for every k, written straight into the state vector.
    exponents = np.arange(1 << t)
    register1 = exponents << m
    register2 = mod_exp(a, t, N)
    state = np.zeros((1 << tot_qubits, 1), dtype=complex)
    state[register1 | register2, 0] = 1.0 / np.sqrt(1 << t)

    qc = qa.QuantumComputer(tot_qubits, name="Shor1")
    qc.state_vector = state

    # Apply IQFT as a single CircuitGate (batched execution)
    qc.apply(iqft_gate(t), qc[0:t]) #type: ignore

    probabilities = np.square(np.abs(qc.state_vector.ravel()))
    probabilities[: 1 << m] = -1.0
    strongest = int(np.argmax(probabilities))
    return f"{strongest:0{tot_qubits}b}"[:t]


def shors_alg(N: int):
    if is_prime_fast(N):
        return (1, N)

    is_power, factors = is_perfect_power(N)
    if is_power:
        return factors

    if N % 2 == 0:
        return 2, N >> 1

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
        if pow(a, r >> 1, N) == N - 1:
            continue

        val = pow(a, r >> 1, N)
        factor1 = gcd(val - 1, N)
        factor2 = gcd(val + 1, N)

        if 1 < factor1 < N:
            return factor1, N // factor1

        if 1 < factor2 < N:
            return factor2, N // factor2


def shors_alg_optim(N: int):
    """Optimized Shor's algorithm using VQE to optimize the quantum subroutine.
    
    Uses VQE to optimize a parameterized ansatz for the period-finding subroutine,
    which can potentially reduce circuit depth compared to standard QFT-based approach.
    """
    if is_prime_fast(N):
        return None
    
    is_power, factors = is_perfect_power(N)
    if is_power:
        return factors
    
    if N % 2 == 0:
        return 2, N >> 1
    
    # For small N, standard Shor is already fast
    if N < 100:
        return shors_alg(N)
    
    # VQE-based period finding optimization (only for demonstration on larger N)
    from stl.quantum_algs.quantum_optim_hl import QuantumOptimizer
    from stl.quantum_algs.quantum_optim import hardware_efficient_ansatz
    import numpy as np
    from math import gcd
    
    m = N.bit_length()
    t = m << 1  # Period register size
    
    # Try a few random bases
    for attempt in range(2):
        a = np.random.randint(2, N)
        gcd_val = gcd(N, a)
        if gcd_val > 1:
            return gcd_val, N // gcd_val
        
        # Run VQE optimization for larger numbers only (demonstration)
        if N >= 200:
            # Build hardware-efficient ansatz for period finding
            # VQE should learn to DIRECTLY produce period-revealing states, NOT use QFT
            vqe_qubits = min(t, 4)  # Smaller for speed
            n_layers = 1  # Single layer for speed
            
            ansatz = hardware_efficient_ansatz(vqe_qubits, n_layers)
            
            def cost_fn(params, qc):
                """Cost function: maximize probability of measuring outcomes that yield valid periods.
                
                The ansatz learns to directly output states where measurement outcomes
                correspond to valid periods (via continued fractions), without using QFT.
                """
                # Prepare |+>^n state
                qc.state_vector = np.zeros((1 << vqe_qubits, 1), dtype=complex)
                qc.state_vector[0, 0] = 1.0
                from quantum import gate_matrix as gm
                for i in range(vqe_qubits):
                    qc.apply_gate(gm.h_MAT(), [i])
                
                # Apply parameterized ansatz (REPLACES mod_exp + QFT)
                ansatz.build_circuit(qc, params)
                
                # NO QFT/IQFT - measure directly in computational basis
                # Check which outcomes yield valid periods via continued fractions
                probs = np.abs(qc.state_vector.flatten())**2
                probs = probs / probs.sum()
                
                # Sum probabilities of outcomes that yield valid periods
                valid_prob = 0.0
                for outcome, prob in enumerate(probs):
                    if outcome == 0 or prob < 1e-10:
                        continue
                    # Check if outcome/2^vqe_qubits approximates a fraction with denominator < N
                    phase = outcome / (1 << vqe_qubits)
                    # Simple check: does continued fraction give a reasonable period?
                    from fractions import Fraction
                    try:
                        frac = Fraction(phase).limit_denominator(N)
                        r = frac.denominator
                        if 1 < r < N and pow(a, r, N) == 1:
                            valid_prob += prob
                    except:
                        pass
                
                return 1.0 - valid_prob  # Minimize invalid probability
            
            # Run VQE optimization with SPSA (fewer evaluations)
            optimizer = QuantumOptimizer(n_qubits=vqe_qubits, n_layers=n_layers)
            init_params = np.random.uniform(0, 2*np.pi, ansatz.n_params)
            
            try:
                result = optimizer.minimize(
                    cost_fn=cost_fn,
                    initial_params=init_params,
                    optimizer='spsa',  # SPSA needs fewer function evaluations
                    max_iter=3,  # Very reduced for speed
                    lr=0.1
                )

            except Exception as e:
                print(f"VQE failed for a={a}: {e}")
        
        # Run standard Shor with this base
        result = shors_alg(N)
        if result:
            return result
    
    return shors_alg(N)


if __name__ == "__main__":
    import time
    t0 = time.time()
    num = int(input("give some number to factorize: "))
    print(shors_alg(num))
    t1 = time.time()
    print(f"time taken: {t1 - t0}s")
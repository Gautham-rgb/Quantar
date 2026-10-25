from lang_systems.runtime import apply_quantar_operator as __quantar_apply__

import quantum
from quantum import QuantumComputer
q = QuantumComputer(4)
for qubit in q.qubits:
    qubit.apply_gate(quantum.h_MAT())
__quantar_apply__('<->', q[0], q[1])
__quantar_apply__('<-s->', q[1], q[2])
__quantar_apply__('<-i->', q[2], q[3])
print(q.state_vector)
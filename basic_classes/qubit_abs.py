import numpy.random as rnd
import numpy as np
import gate_matrix as gm

class Qubit:
    def __init__(self, computer: QuantumComputer, index: int, name: str = "qubit"):
        self.computer = computer
        self.index = index
        self.name = name
        self.is_measured = False
        self._true_val = None # true value of qubit, only given a a value when measured

    def __repr__(self):
        return f"{self.name}({self.index})"

    @classmethod
    def from_list_str(cls, computer: QuantumComputer, list: list[str]):
        qubits = []
        for i in range(len(list)):
            qubit = cls(computer, i, list[i])
            qubits.append(qubit)
        return qubits

    def is_alive(self):
        return not self.is_measured

    def _assert_alive(self):
        if not self.is_alive():
            raise ValueError(f"Qubit {self.name} is locked after measurement (no gates can be applied).")

    def apply_gate(self, gate_matrix: np.ndarray):
        self._assert_alive()
        op = np.array([[1.]], dtype = complex)



    def measure(self):
        self._assert_alive()
        if not self.is_alive():
            return self._true_val
        
        self.is_measured = True
        self._true_val = rnd.choice([0, 1], p=[1 - self.prob_1, self.prob_1])
        self.prob_1 = 1.0 if self._true_val == 1 else 0.0
        return self._true_val




class QuantumComputer:
    def __init__(self, num_qubits: int, name: str = "QC1", qubit_names: list[str] |None = None):
        self.num_qubits = num_qubits
        self.name = name

        if qubit_names == None:
            self.qubits: list[Qubit] = [Qubit(self, index = i, name = f"q{i}") for i in range(num_qubits)]
        else:
            if len(qubit_names) != num_qubits:
                raise ValueError(f"The number of qubit names ({len(qubit_names)}) must match the number of qubits ({self.num_qubits}).")
            
            self.qubits = Qubit.from_list_str(self, qubit_names)

    def __getitem__(self, key):
        if isinstance(key, int):
            if key < 0 or key >= self.num_qubits:
                raise IndexError(f"Qubit index {key} is out of range for a quantum computer with {self.num_qubits} qubits.")
            return self.qubits[key]
        elif isinstance(key, str):
            for qubit in self.qubits:
                if qubit.name == key:
                    return qubit
            raise KeyError(f"No qubit found with name '{key}'.")
        else:
            raise TypeError(f"Key ({key}) must be an integer index or a string name.")
            

    def __repr__(self):
        qubit_lines = "\n  └─ ".join([str(qubit) for qubit in self.qubits])
        return f"System: {self.name} ({self.num_qubits} Qubits)\n  └─ {qubit_lines}"


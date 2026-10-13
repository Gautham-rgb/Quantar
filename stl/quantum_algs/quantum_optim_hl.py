"""
High-level quantum optimization interface.
Define cost functions directly, let Quantar handle the rest.
"""

import numpy as np
from typing import Callable, Optional, List, Literal
from dataclasses import dataclass

import numpy as np
from quantum.qubit_abs import QuantumComputer
from quantum.gate_matrix import rx_MAT, ry_MAT, rz_MAT, p_MAT, px_MAT
from stl.quantum_algs.quantum_optim import (
    hardware_efficient_ansatz, 
    measure_pauli_expectation, 
    Hamiltonian, 
    Ansatz
)


@dataclass
class OptimizeResult:
    """Result of a quantum optimization."""
    optimal_params: np.ndarray
    min_cost: float
    history: List[float]
    n_iterations: int
    success: bool
    message: str


class QuantumOptimizer:
    """
    High-level quantum optimizer.
    
    Usage:
        # Define your cost function
        def cost(params):
            qc = QuantumComputer(3)
            # build circuit with params...
            return expectation_value
        
        # Optimize
        optimizer = QuantumOptimizer(n_qubits=3, n_layers=2)
        result = optimizer.minimize(cost, initial_params)
        print(result.min_cost, result.optimal_params)
    """
    
    def __init__(
        self, n_qubits: int, n_layers: int = 2, 
        shots: int = 1000, hamiltonian: Optional['Hamiltonian'] = None
    ):
        """
        Args:
            n_qubits: Number of qubits
            n_layers: Number of ansatz layers
            shots: Measurement shots for expectation
            hamiltonian: Optional Hamiltonian for VQE
        """
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.shots = shots
        
        self._qc = QuantumComputer(n_qubits)
        
        # Build default hardware-efficient ansatz
        from stl.quantum_algs.quantum_optim import hardware_efficient_ansatz
        self._ansatz = hardware_efficient_ansatz(n_qubits, n_layers, self._qc)
        
        # Optional Hamiltonian for VQE
        self._hamiltonian = None
    
    def set_hamiltonian(self, hamiltonian):
        """Set Hamiltonian for VQE."""
        self._hamiltonian = hamiltonian
    
    def _build_energy_fn(self, cost_fn: Callable[[np.ndarray, 'QuantumComputer'], float]):
        """Build energy function that constructs circuit and returns cost."""
        def energy(params: np.ndarray) -> float:
            qc = QuantumComputer(self.n_qubits)
            self._ansatz.build_circuit(qc, params)
            return cost_fn(params, qc)
        return energy
    
    def minimize(
        self, 
        cost_fn: Callable[[np.ndarray, 'QuantumComputer'], float], 
        initial_params: Optional[np.ndarray] = None, 
        optimizer: Literal["gradient", "spsa", "cobyla"] = "gradient", 
        max_iter: int = 100, 
        lr: float = 0.1, 
        callback: Optional[Callable[[int, float, np.ndarray], None]] = None,
        verbose: bool = True
    ) -> 'OptimizeResult':
        """
        Minimize a cost function using quantum circuit.
        
        Args:
            cost_fn: Function(params, qc) -> cost. 
                     Receives params and a fresh QuantumComputer.
                     Should build circuit and return expectation.
            initial_params: Initial parameters (random if None)
            optimizer: 'gradient', 'spsa', or 'cobyla'
            max_iter: Maximum iterations
            lr: Learning rate
            callback: Called as callback(iteration, cost, params)
        
        Returns:
            OptimizeResult with optimal_params, min_cost, history
        """
        if initial_params is None:
            initial_params = np.random.uniform(0, 2*np.pi, self._ansatz.n_params)
        
        params = initial_params.copy()
        history = []
        
        def energy(params: np.ndarray) -> float:
            """Build circuit with params and return cost."""
            qc = QuantumComputer(self.n_qubits)
            self._ansatz.build_circuit(qc, params)
            return cost_fn(params, qc)
        
        history.append(energy(params))
        if callback:
            callback(0, history[-1], params.copy())
        
        if optimizer == "gradient":
            for i in range(max_iter):
                # Compute gradient via parameter shift rule
                grad = np.zeros_like(params)
                shift = np.pi / 2
                for j in range(len(params)):
                    params_plus = params.copy()
                    params_plus[j] += np.pi / 2
                    params_minus = params.copy()
                    params_minus[j] -= np.pi / 2
                    
                    # Parameter shift rule: gradient = 0.5 * (E(θ+π/2) - E(θ-π/2))
                    e_plus = energy(params_plus)
                    e_minus = energy(params_minus)
                    grad[j] = 0.5 * (energy(params_plus) - energy(params_minus))
                
                params -= lr * grad
                
                cost = energy(params)
                history.append(cost)
                
                if callback:
                    callback(i + 1, cost, params.copy())
                
                if i % 10 == 0 and verbose:
                    print(f"  Iter {i}: cost = {cost:.6f}")
        
        elif optimizer == "spsa":
            # Simultaneous Perturbation Stochastic Approximation
            # Only needs 2 function evaluations per iteration regardless of parameter count
            a = lr
            c = 0.1
            A = max_iter * 0.1
            alpha = 0.602
            gamma = 0.101
            
            for i in range(max_iter):
                # Compute step sizes
                ak = a / (i + 1 + A) ** alpha
                ck = c / (i + 1) ** gamma
                
                # Random perturbation vector
                delta = np.random.choice([-1, 1], size=len(params))
                
                params_plus = params + ck * delta
                params_minus = params - ck * delta
                
                # Evaluate cost at perturbed points
                e_plus = energy(params_plus)
                e_minus = energy(params_minus)
                
                # Gradient approximation
                grad = (e_plus - e_minus) / (2 * ck * delta)
                
                # Update parameters
                params -= ak * grad
                
                cost = energy(params)
                history.append(cost)
                
                if callback:
                    callback(i + 1, cost, params.copy())
                
                if i % 5 == 0 and verbose:
                    print(f"  Iter {i}: cost = {cost:.6f}")
        
        elif optimizer == "cobyla":
            # Use SciPy's COBYLA if available
            try:
                from scipy.optimize import minimize as sp_minimize
                result = sp_minimize(energy, params, method='COBYLA', 
                                   options={'maxiter': max_iter, 'rhobeg': lr})
                params = result.x
                history.append(energy(params))
            except ImportError:
                # Fallback to gradient
                print("SciPy not available, falling back to gradient")
                for i in range(max_iter):
                    grad = np.zeros_like(params)
                    for j in range(len(params)):
                        params_plus = params.copy()
                        params_plus[j] += np.pi / 2
                        params_minus = params.copy()
                        params_minus[j] -= np.pi / 2
                        e_plus = energy(params_plus)
                        e_minus = energy(params_minus)
                        grad[j] = 0.5 * (e_plus - e_minus)
                    params -= lr * grad
                    cost = energy(params)
                    history.append(cost)
                    if callback:
                        callback(i + 1, cost, params.copy())
        
        return OptimizeResult(
            optimal_params=params,
            min_cost=history[-1],
            history=history,
            n_iterations=len(history) - 1,
            success=True,
            message="Optimization completed"
        )


def minimize_cost(
    cost_fn: Callable[[np.ndarray, 'QuantumComputer'], float],
    n_qubits: int,
    n_layers: int = 2,
    initial_params: Optional[np.ndarray] = None,
    optimizer: Literal["cobyla", "spsa", "gradient"] = "gradient",
    max_iter: int = 100,
    lr: float = 0.1) -> 'OptimizeResult':
    """
    One-liner to minimize a cost function.
    
    Args:
        cost_fn: Function(params, qc) -> cost
        n_qubits: Number of qubits
        n_layers: Ansatz layers
        initial_params: Starting parameters
        optimizer: 'gradient' | 'spsa' | 'cobyla'
        max_iter: Max iterations
        lr: Learning rate
    
    Returns:
        OptimizeResult
    """
    opt = QuantumOptimizer(n_qubits, n_layers)
    return opt.minimize(cost_fn, initial_params, optimizer, max_iter, lr)


class Penalty:
    """
    Define a penalty term for constrained optimization.
    
    Usage:
        penalty = Penalty("constraint", lambda qc: measure(qc), target=0.0, weight=10.0)
        cost = lambda params, qc: objective(params, qc) + penalty.evaluate(params, qc)
    """
    
    def __init__(
        self,
        name: str,
        measure_fn: Callable[[QuantumComputer], float],
        target: float = 0.0,
        weight: float = 1.0,
        penalty_type: Literal["quadratic", "linear", "log_barrier"] = "quadratic",  # 'quadratic', 'linear', 'log_barrier'
    ):
        """
        Args:
            name: Penalty name
            measure_fn: Function(qc) -> measured value
            target: Target value (constraint: measure == target)
            weight: Penalty weight
            penalty_type: 'quadratic' | 'linear' | 'log_barrier'
        """
        self.name = name
        self.measure_fn = measure_fn
        self.target = target
        self.weight = weight
        self.penalty_type = penalty_type
    
    def evaluate(self, qc: 'QuantumComputer') -> float:
        """Evaluate penalty for current state."""
        value = self.measure_fn(qc)
        diff = value - self.target
        
        if self.penalty_type == "quadratic":
            return self.weight * (diff ** 2)
        elif self.penalty_type == "linear":
            return self.weight * abs(diff)
        elif self.penalty_type == "log_barrier":
            # For inequality constraints: target is upper bound
            if diff >= 0:
                return 1e6  # Large penalty for violation
            return -self.weight * np.log(-diff)
        else:
            raise ValueError(f"Unknown penalty type: {self.penalty_type}")
    
    def __call__(self, qc: 'QuantumComputer') -> float:
        return self.evaluate(qc)


def penalized_cost(
    objective_fn: Callable[[np.ndarray, 'QuantumComputer'], float],
    penalties: List['Penalty'],
) -> Callable[[np.ndarray, 'QuantumComputer'], float]:
    """
    Combine objective with penalties.
    
    Usage:
        cost = penalized_cost(objective, [penalty1, penalty2])
        result = minimize_cost(cost, n_qubits=3)
    """
    def cost_fn(params, qc):
        cost = objective_fn(params, qc)
        for penalty in penalties:
            cost += penalty(qc)
        return cost
    return cost_fn


if __name__ == "__main__":
    # Example: Minimize <Z0 Z1> + penalty for |00> population
    import numpy as np
    from quantum.qubit_abs import QuantumComputer
    from quantum import gate_matrix as gm
    
    # Define a simple objective: minimize <Z0 Z1>
    def objective(params, qc):
        qc.state_vector = np.zeros((4, 1), dtype=complex)
        qc.state_vector[0, 0] = 1.0
        # Build ansatz manually for demo
        qc.apply('H', [0])
        qc.apply('RY', [0], params=[params[0]])  # Need RY gate
        qc.apply('CNOT', [0, 1])
        return qc.measure_pauli('ZZ', [0, 1])
    
    # Add penalty: discourage |00> state (penalty if <00|psi> > 0.1)
    penalty = Penalty(
        "discourage_00",
        lambda qc: np.abs(qc.state_vector[0, 0])**2,
        target=0.1,
        weight=10.0,
    )
    
    cost = penalized_cost(objective, [penalty])
    
    optimizer = QuantumOptimizer(n_qubits=2, n_layers=2)
    result = optimizer.minimize(
        cost_fn=cost,
        optimizer="gradient",
        max_iter=50,
        lr=0.1,
    )
    
    print(f"Min cost: {result.min_cost}")
    print(f"Params: {result.optimal_params}")
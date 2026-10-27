"""Physics algorithms: automatic differentiation, numerical integration,
and classical dynamics."""

import inspect
import numpy as np
from typing import Callable, Literal


class Dual:
    """Dual number for forward-mode automatic differentiation.

    A dual number has a real part and a dual part (derivative).
    Arithmetic operations propagate derivatives via the chain rule.

    Example:
        x = Dual(3.0, [1.0, 0.0])  # x = 3, dx/dx = 1
        y = x * x + 2 * x
        print(y.real, y.dual)  # (15.0, [8.0])
    """

    def __init__(self, real, dual=None):
        self.real = float(real)
        self.dual = np.array(dual, dtype=float) if dual is not None else np.array([0.0])

    def __neg__(self):
        return Dual(-self.real, -self.dual)

    def __pos__(self):
        return self

    def __abs__(self):
        return Dual(abs(self.real), self.dual if self.real >= 0 else -self.dual)

    def __add__(self, other):
        if not isinstance(other, Dual):
            return Dual(self.real + other, self.dual)
        return Dual(self.real + other.real, self.dual + other.dual)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __mul__(self, other):
        if not isinstance(other, Dual):
            return Dual(self.real * other, self.dual * other)
        return Dual(self.real * other.real, self.dual * other.real + self.real * other.dual)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if not isinstance(other, Dual):
            return Dual(self.real / other, self.dual / other)
        denom = other.real ** 2
        return Dual(self.real / other.real, (self.dual * other.real - self.real * other.dual) / denom)

    def __rtruediv__(self, other):
        other = Dual(other, np.zeros_like(self.dual))
        return other.__truediv__(self)

    def __pow__(self, power):
        if isinstance(power, Dual):
            real_val = self.real ** power.real
            dual_val = real_val * (power.dual * np.log(self.real) + power.real * self.dual / self.real)
            return Dual(real_val, dual_val)
        return Dual(self.real ** power, power * (self.real ** (power - 1)) * self.dual)

    def __rpow__(self, base):
        real_val = base ** self.real
        return Dual(real_val, real_val * np.log(base) * self.dual)

    _UFUNC_MAP = {
        np.exp: (lambda r: np.exp(r), lambda r: np.exp(r)),
        np.exp2: (lambda r: 2**r, lambda r: (2**r) * np.log(2)),
        np.expm1: (lambda r: np.expm1(r), lambda r: np.exp(r)),
        np.log: (lambda r: np.log(r), lambda r: 1.0 / r),
        np.log10: (lambda r: np.log10(r), lambda r: 1.0 / (r * np.log(10))),
        np.log2: (lambda r: np.log2(r), lambda r: 1.0 / (r * np.log(2))),
        np.log1p: (lambda r: np.log1p(r), lambda r: 1.0 / (r + 1.0)),
        np.sin: (lambda r: np.sin(r), lambda r: np.cos(r)),
        np.cos: (lambda r: np.cos(r), lambda r: -np.sin(r)),
        np.tan: (lambda r: np.tan(r), lambda r: 1.0 / (np.cos(r) ** 2)),
        np.arcsin: (lambda r: np.asin(r), lambda r: 1.0 / np.sqrt(1.0 - r**2)),
        np.arccos: (lambda r: np.acos(r), lambda r: -1.0 / np.sqrt(1.0 - r**2)),
        np.arctan: (lambda r: np.atan(r), lambda r: 1.0 / (1.0 + r**2)),
        np.sinh: (lambda r: np.sinh(r), lambda r: np.cosh(r)),
        np.cosh: (lambda r: np.cosh(r), lambda r: np.sinh(r)),
        np.tanh: (lambda r: np.tanh(r), lambda r: 1.0 / (np.cosh(r) ** 2)),
        np.arcsinh: (lambda r: np.asinh(r), lambda r: 1.0 / np.sqrt(r**2 + 1.0)),
        np.arccosh: (lambda r: np.acosh(r), lambda r: 1.0 / np.sqrt(r**2 - 1.0)),
        np.arctanh: (lambda r: np.atanh(r), lambda r: 1.0 / (1.0 - r**2)),
        np.sqrt: (lambda r: np.sqrt(r), lambda r: 0.5 / np.sqrt(r)),
        np.cbrt: (lambda r: r**(1/3) if r>=0 else -(-r)**(1/3), lambda r: (1/3) * (r**(-2/3))),
        np.square: (lambda r: r**2, lambda r: 2.0 * r),
        np.reciprocal: (lambda r: 1.0 / r, lambda r: -1.0 / (r**2)),
    }

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method != '__call__':
            return NotImplemented
        inp = inputs[0]
        if not isinstance(inp, Dual):
            return NotImplemented
        if ufunc in self._UFUNC_MAP:
            func, deriv = self._UFUNC_MAP[ufunc]
            return Dual(func(inp.real), inp.dual * deriv(inp.real))
        if ufunc in [np.floor, np.ceil, np.trunc, np.rint]:
            return Dual(ufunc(inp.real), 0.0)
        raise ValueError(f"Operation '{ufunc.__name__}' not defined for differentiation.")

    def __repr__(self):
        return f"Dual(real={self.real}, dual={self.dual.tolist()})"

    @classmethod
    def gradient(cls, func: Callable, point: list):
        """Compute gradient of func at point using forward-mode AD.

        Args:
            func: Function taking list of Dual numbers, returning Dual or list of Dual.
            point: List of real values at which to compute gradient.

        Returns:
            Result of func evaluated with dual numbers containing gradient.
        """
        n = len(point)
        inputs = []
        for i in range(n):
            seed = np.zeros(n)
            seed[i] = 1.0
            inputs.append(cls(point[i], seed))
        return func(inputs)

    @classmethod
    def integrate_nd(cls, func: Callable, bounds: list, deg: int = 20):
        """N-dimensional numerical integration using Gauss-Legendre quadrature.

        Args:
            func: Function taking list of coordinates, returning scalar.
            bounds: List of (lower, upper) bounds for each dimension.
            deg: Quadrature degree (number of nodes per dimension).

        Returns:
            Approximate integral value.
        """
        n = len(bounds)
        x_nodes, weights = np.polynomial.legendre.leggauss(deg)
        grid_axes = [0.5 * ((b - a) * x_nodes + (b + a)) for a, b in bounds]
        mesh = np.meshgrid(*grid_axes, indexing='ij')
        flat_mesh = [axis.flatten() for axis in mesh]
        weight_mesh = np.meshgrid(*[weights for _ in range(n)], indexing='ij')
        flat_weights = np.prod([w.flatten() for w in weight_mesh], axis=0)
        total_sum = 0.0
        scale_factor = np.prod([0.5 * (b - a) for a, b in bounds])
        for idx in range(len(flat_mesh[0])):
            pt = [axis[idx] for axis in flat_mesh]
            total_sum += flat_weights[idx] * func(pt)
        return scale_factor * total_sum


def dual_einsum(subscripts: str, *operands):
    """Einstein summation with automatic differentiation support.

    Args:
        subscripts: Einstein summation subscripts string.
        *operands: Operands (arrays or scalars, may contain Dual numbers).

    Returns:
        Result of einsum with derivatives propagated.
    """
    real_operands = []
    grad_dim = 0
    for op in operands:
        if isinstance(op, np.ndarray):
            real_operands.append(np.vectorize(lambda x: x.real if isinstance(x, Dual) else float(x))(op))
            for item in op.flat:
                if isinstance(item, Dual):
                    grad_dim = max(grad_dim, item.dual.shape[0])
        else:
            real_operands.append(op.real if isinstance(op, Dual) else float(op))
            if isinstance(op, Dual):
                grad_dim = max(grad_dim, op.dual.shape[0])
    real_result = np.einsum(subscripts, *real_operands)
    if grad_dim == 0:
        grad_dim = 1
    dual_shape = real_result.shape + (grad_dim,)
    dual_result = np.zeros(dual_shape)
    for i, op in enumerate(operands):
        modified_operands = list(real_operands)
        for g in range(grad_dim):
            if isinstance(op, np.ndarray):
                modified_operands[i] = np.vectorize(lambda x: x.dual[g] if isinstance(x, Dual) and g < len(x.dual) else 0.0)(op)
            else:
                modified_operands[i] = op.dual[g] if isinstance(op, Dual) and g < len(op.dual) else 0.0
            dual_result[..., g] += np.einsum(subscripts, *modified_operands)
    if real_result.ndim == 0:
        return Dual(real_result, dual_result)
    out_array = np.empty(real_result.shape, dtype=object)
    it = np.nditer(real_result, flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        out_array[idx] = Dual(real_result[idx], dual_result[idx])
        it.iternext()
    return out_array


def dual_kron(a, b):
    """Kronecker product with automatic differentiation support.

    Args:
        a: First operand (array or Dual).
        b: Second operand (array or Dual).

    Returns:
        Kronecker product with derivatives propagated.
    """
    a_flat = a.flatten() if isinstance(a, np.ndarray) else np.array([a])
    b_flat = b.flatten() if isinstance(b, np.ndarray) else np.array([b])
    out_shape = tuple(s1 * s2 for s1, s2 in zip(a.shape, b.shape)) if isinstance(a, np.ndarray) and isinstance(b, np.ndarray) else (len(a_flat) * len(b_flat),)
    out = np.empty(out_shape, dtype=object)
    for idx_a, val_a in np.ndenumerate(a):
        for idx_b, val_b in np.ndenumerate(b):
            target_idx = tuple(i * s + j for i, j, s in zip(idx_a, idx_b, b.shape))
            out[target_idx] = val_a * val_b
    return out


def dual_norm(vector_array):
    """Euclidean norm of array with Dual elements.

    Args:
        vector_array: Array of Dual numbers.

    Returns:
        Dual number representing the norm with derivatives.
    """
    squared_sum = sum(x ** 2 for x in vector_array.flat)
    return np.sqrt(squared_sum)


class SpaceMetric:
    """Coordinate system metric for gradient transformation.

    Provides scale factors for different coordinate systems to transform
    raw gradients (partial derivatives) into physical gradients.
    """

    def __init__(self, system: Literal["cartesian", "spherical", "cylindrical"] = "cartesian"):
        """Initialize metric for given coordinate system.

        Args:
            system: One of "cartesian", "spherical", "cylindrical".
        """
        self.system = system

    def scale_factors(self, coords):
        """Compute scale factors for given coordinates.

        Args:
            coords: Coordinate values (length depends on system).

        Returns:
            List of scale factors for each coordinate.
        """
        if self.system == "cartesian":
            return [1.0 for _ in coords]
        elif self.system == "spherical":
            r, theta = coords[0], coords[1]
            return [1.0, r, r * np.sin(theta)]
        elif self.system == "cylindrical":
            rho = coords[0]
            return [1.0, rho, 1.0]
        raise ValueError("Unknown metric system.")

    def transform_gradient(self, raw_gradient, coords):
        """Transform raw partial derivatives to physical gradient components.

        Args:
            raw_gradient: Partial derivatives wrt coordinates.
            coords: Coordinate values.

        Returns:
            Physical gradient components.
        """
        h = self.scale_factors(coords)
        return [raw_gradient[i] / h[i] for i in range(len(h))]


# --- Classical dynamics ------------------------------------------------------

def _broadcast_mass(masses, shape):
    """Broadcast a scalar, per-particle, or per-component mass to ``shape``."""
    values = np.asarray(masses, dtype=float)
    if values.shape == shape:
        return values
    if values.ndim == 0:
        return np.full(shape, float(values))
    if values.shape == shape[:-1]:
        return np.broadcast_to(values[..., None], shape)
    return np.broadcast_to(values, shape)


def kinetic_energy(velocities, masses=1.0) -> float:
    """Kinetic energy ``0.5 * sum(m * v**2)`` for one configuration.

    ``masses`` may be a scalar, per-particle ``(n,)``, or per-component array.
    """
    velocities = np.asarray(velocities, dtype=float)
    mass = _broadcast_mass(masses, velocities.shape)
    return float(0.5 * np.sum(mass * velocities * velocities))


def _required_parameters(function):
    target = getattr(function, "py_func", function)
    try:
        parameters = inspect.signature(target).parameters.values()
    except (TypeError, ValueError):
        return None

    positional_kinds = (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    )
    return [
        parameter
        for parameter in parameters
        if parameter.kind in positional_kinds and parameter.default is parameter.empty
    ]


def _call_with_arity(function, position, velocity, time):
    """Call a dynamics function with the arity it declares.

    Supports ``f(x)``, ``f(x, v)`` and ``f(x, v, t)``.  numba dispatchers are
    inspected through their ``py_func``.
    """
    required = _required_parameters(function)
    if required is not None:
        if len(required) <= 1:
            return function(position)
        if len(required) == 2:
            return function(position, velocity)
        return function(position, velocity, time)

    try:
        return function(position, velocity, time)
    except TypeError:
        try:
            return function(position, velocity)
        except TypeError:
            return function(position)


class Trajectory:
    """Sampled positions, velocities and times from an integrator."""

    def __init__(self, positions, velocities, times):
        self.positions = np.asarray(positions, dtype=float)
        self.velocities = np.asarray(velocities, dtype=float)
        self.times = np.asarray(times, dtype=float)

    def __len__(self) -> int:
        return int(self.times.size)

    def __getitem__(self, index):
        return self.times[index], self.positions[index], self.velocities[index]

    def __repr__(self) -> str:
        return (
            f"Trajectory(samples={len(self)}, t=[{self.times[0]:g}, {self.times[-1]:g}], "
            f"shape={self.positions.shape[1:]})"
        )

    def kinetic_energy(self, masses=1.0) -> np.ndarray:
        """Kinetic energy at every sampled time."""
        return np.asarray(
            [kinetic_energy(velocity, masses) for velocity in self.velocities],
            dtype=float,
        )

    def energy(self, potential: Callable, masses=1.0) -> np.ndarray:
        """Total energy at every sampled time, for ``potential(position)``."""
        potential_values = np.asarray(
            [potential(position) for position in self.positions], dtype=float
        )
        return self.kinetic_energy(masses) + potential_values


def velocity_verlet(
    positions,
    velocities,
    dt: float,
    steps: int,
    acceleration: Callable | None = None,
    *,
    force: Callable | None = None,
    masses=1.0,
    t0: float = 0.0,
    save_every: int = 1,) -> Trajectory:

    """Integrate dynamics with the velocity-Verlet scheme.

    The update is the standard one::

        x(t+dt) = x + v dt + 0.5 a(t) dt^2
        v(t+dt) = v + 0.5 (a(t) + a(t+dt)) dt

    Pass either ``acceleration`` (``a(x)``, ``a(x, v)`` or ``a(x, v, t)``) or
    ``force`` (same signatures, divided by ``masses``).  ``masses`` applies only
    to ``force=``.  Accelerations may be scalar or per-particle arrays; they are
    broadcast to the shape of ``positions``.

    Returns a :class:`Trajectory`.  ``save_every`` stores every n-th step (the
    initial and final states are always stored).

    Example
    -------
    >>> traj = velocity_verlet([1.0, 0.0], [0.0, 1.0], 0.01, 628, lambda x: -x)
    >>> np.allclose(traj.positions[-1], [1.0, 0.0], atol=5e-3)
    True
    """
    if (acceleration is None) == (force is None):
        raise ValueError("Pass exactly one of 'acceleration' or 'force'.")
    if dt <= 0:
        raise ValueError("dt must be positive.")
    if steps < 0:
        raise ValueError("steps must be non-negative.")
    if save_every < 1:
        raise ValueError("save_every must be at least 1.")

    x = np.array(positions, dtype=float, copy=True)
    v = np.array(velocities, dtype=float, copy=True)
    if x.shape != v.shape:
        raise ValueError(f"positions {x.shape} and velocities {v.shape} must match.")
    if x.size == 0:
        raise ValueError("positions must not be empty.")

    def acceleration_at(position, velocity, time):

        if force is not None:
            mass = _broadcast_mass(masses, x.shape)
            force_value = _call_with_arity(force, position, velocity, time)
            return np.asarray(force_value, dtype=float) / mass

        else:
            return _call_with_arity(acceleration, position, velocity, time)

    def evaluate_acceleration(position, velocity, time):
        value = np.asarray(acceleration_at(position, velocity, time), dtype=float)
        if value.shape == x.shape:
            return value
        return np.broadcast_to(value, x.shape).astype(float, copy=True)

    current_time = float(t0)
    accel = evaluate_acceleration(x, v, current_time)

    times = [current_time]
    positions_out = [x.copy()]
    velocities_out = [v.copy()]

    for step in range(1, steps + 1):
        x = x + v * dt + 0.5 * accel * dt * dt
        current_time = t0 + step * dt
        new_accel = evaluate_acceleration(x, v, current_time)
        v = v + 0.5 * (accel + new_accel) * dt
        accel = new_accel

        if step % save_every == 0 or step == steps:
            times.append(current_time)
            positions_out.append(x.copy())
            velocities_out.append(v.copy())

    return Trajectory(
        np.asarray(positions_out),
        np.asarray(velocities_out),
        np.asarray(times),
    )


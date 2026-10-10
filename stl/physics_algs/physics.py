"""Physics algorithms: automatic differentiation and numerical integration."""

import numpy as np
from typing import Callable


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

    def __init__(self, system="cartesian"):
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
            r, theta, phi = coords[0], coords[1], coords[2]
            return [1.0, r, r * np.sin(theta)]
        elif self.system == "cylindrical":
            rho, phi, z = coords[0], coords[1], coords[2]
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
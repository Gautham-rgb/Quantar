"""Physical units built from base units.

A unit is a product of powers of base units.  This module provides:

* :class:`Unit` -- a dimension, a scale, and an optional affine offset;
* :class:`Quantity` -- a value together with a unit;
* :class:`UnitRegistry` -- named units plus an expression parser;
* :data:`SI` -- a ready-made registry with the SI base and common derived units.

Unit expressions use ``*``, ``/``, ``**`` (or ``^``), parentheses, numbers and
unit names, for example ``"kg*m/s^2"`` or ``"km/h"``.

Definitions can be written in Quantar source with ordinary calls (Quantar is a
Python superset, so this is valid source)::

    from stl.physics_algs.units import unit, units, Q

    m, s, kg = units("m", "s", "kg")
    N = unit("N", "kg*m/s^2")
    speed = Q(100, "km/h").to("m/s")

They can also be loaded from a compact definition block::

    unit m
    unit s
    unit kg
    unit N = kg*m/s^2

with :func:`load_units` or :meth:`UnitRegistry.load`.
"""

from __future__ import annotations

import ast
import math
import re
from fractions import Fraction
from typing import Iterable

import numpy as np


class UnitError(ValueError):
    """Base class for unit and dimension errors."""


class DimensionError(UnitError):
    """Raised when incompatible dimensions are combined."""


class UnknownUnitError(UnitError, KeyError):
    """Raised when a unit name is not in the registry."""


def _exponent(value) -> Fraction:
    """Convert an exponent to an exact fraction (0.5 -> 1/2)."""
    if isinstance(value, Fraction):
        return value
    if isinstance(value, bool):
        raise TypeError("boolean exponents are not allowed")
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, float):
        return Fraction(value).limit_denominator(1_000_000)
    raise TypeError(f"unsupported exponent type: {type(value).__name__}")


def _validate_symbol(symbol: str) -> None:
    """Reject symbols that cannot safely name a unit."""
    if not isinstance(symbol, str) or not symbol:
        raise UnitError("unit symbol must be a non-empty string")
    if any(character.isspace() for character in symbol) or any(
        operator in symbol for operator in "*/^()=#"
    ):
        raise UnitError(f"invalid unit symbol: {symbol!r}")


def _merge_dimensions(left, right, multiplier):
    dimension = dict(left)
    for symbol, exponent in right.items():
        value = dimension.get(symbol, Fraction(0)) + multiplier * exponent
        if value:
            dimension[symbol] = value
        elif symbol in dimension:
            del dimension[symbol]
    return dimension


def _format_exponent(exponent: Fraction) -> str:
    if exponent.denominator == 1:
        return str(exponent.numerator)
    return f"({exponent.numerator}/{exponent.denominator})"


def _symbol_with_exponent(symbol: str, exponent: Fraction) -> str:
    if exponent == 1:
        return symbol
    return f"{symbol}^{_format_exponent(exponent)}"


def _require_identifier(symbol: str, kind: str) -> None:
    if not isinstance(symbol, str) or not symbol.isidentifier():
        raise UnitError(f"{kind} name must be an identifier, got {symbol!r}")


def _is_one(value) -> bool:
    return np.isscalar(value) and not isinstance(value, bool) and value == 1


_QUANTITY_PATTERN = re.compile(
    r"^\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*(.*?)\s*$"
)


class Unit:
    """A physical unit: a dimension, a scale to base units, and an offset.

    ``dimension`` maps base-unit symbols to exponents, e.g.
    ``{"kg": 1, "m": 1, "s": -2}`` for newtons.  ``scale`` converts to the base
    units (``km`` has scale 1000), and ``offset`` supports affine scales such as
    degrees Celsius.
    """

    __slots__ = ("symbol", "dimension", "scale", "offset", "name")

    def __init__(self, symbol="", dimension=None, scale=1.0, offset=0.0, name=None):
        self.symbol = symbol or ""
        self.dimension = {
            key: _exponent(value)
            for key, value in (dimension or {}).items()
            if value != 0
        }
        self.scale = float(scale)
        self.offset = float(offset)
        self.name = name or self.symbol or "dimensionless"

    # --- arithmetic -------------------------------------------------------

    def _combine(self, other, *, divide=False):
        if not isinstance(other, Unit):
            return NotImplemented
        if self.offset or other.offset:
            raise UnitError("affine units (such as degC) cannot be multiplied or divided")

        if divide:
            dimension = _merge_dimensions(self.dimension, other.dimension, -1)
            scale = self.scale / other.scale
        else:
            dimension = _merge_dimensions(self.dimension, other.dimension, 1)
            scale = self.scale * other.scale
        return Unit("", dimension, scale)

    def __mul__(self, other):
        if isinstance(other, Unit):
            return self._combine(other)
        if isinstance(other, Quantity):
            return NotImplemented
        return Quantity._build(other, self)

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, Unit):
            return self._combine(other, divide=True)
        if isinstance(other, Quantity):
            return NotImplemented
        return Quantity._build(1.0 / other, self)

    def __rtruediv__(self, other):
        if isinstance(other, Quantity):
            return NotImplemented
        inverse = self ** -1
        if _is_one(other):
            return inverse
        return Quantity._build(other, inverse)

    def __pow__(self, power):
        if self.offset:
            raise UnitError("affine units cannot be raised to a power")
        exponent = _exponent(power)
        dimension = {}
        for symbol, current in self.dimension.items():
            updated = current * exponent
            if updated:
                dimension[symbol] = updated
        return Unit("", dimension, self.scale ** float(exponent))

    def scaled(self, factor, symbol: str = "", name: str | None = None) -> "Unit":
        """Return a copy whose scale is multiplied by ``factor``.

        ``cm = SI["m"].scaled(0.01, "cm")``.
        """
        if self.offset:
            raise UnitError("affine units cannot be rescaled")
        return Unit(symbol, self.dimension, self.scale * factor, 0.0, name or symbol)

    def named(self, symbol: str, name: str | None = None) -> "Unit":
        """Return a copy carrying ``symbol`` (useful before registering it)."""
        _validate_symbol(symbol)
        return Unit(symbol, self.dimension, self.scale, self.offset, name or symbol)

    # --- numpy interop ----------------------------------------------------

    __array_priority__ = 1000

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method != "__call__" or kwargs.get("out") is not None:
            return NotImplemented

        if ufunc is np.multiply:
            return _multiply_operands(*inputs)
        if ufunc in (np.divide, np.true_divide):
            return _divide_operands(*inputs)
        if ufunc is np.power:
            base, exponent = inputs
            return base ** exponent if isinstance(base, Unit) else NotImplemented
        return NotImplemented

    # --- conversion -------------------------------------------------------

    def compatible(self, other) -> bool:
        """Whether two units have the same dimension."""
        return isinstance(other, Unit) and self.dimension == other.dimension

    def to_base(self, value):
        """Convert a value in this unit to base units."""
        return value * self.scale + self.offset

    def from_base(self, value):
        """Convert a value in base units to this unit."""
        return (value - self.offset) / self.scale

    def convert(self, value, target):
        """Convert a value from this unit to ``target``."""
        if not self.compatible(target):
            raise DimensionError(
                f"cannot convert {self} to {target}: incompatible dimensions"
            )
        return target.from_base(self.to_base(value))

    # --- identity and display --------------------------------------------

    def _signature(self):
        return tuple(sorted(self.dimension.items())), self.scale, self.offset

    def __eq__(self, other):
        return isinstance(other, Unit) and self._signature() == other._signature()

    def __hash__(self):
        return hash(self._signature())

    def __str__(self):
        if self.symbol:
            return self.symbol
        if not self.dimension:
            return "1" if self.scale == 1.0 else f"{self.scale:g}"

        numerator, denominator = self._display_parts()
        text = "*".join(numerator) if numerator else "1"
        if denominator:
            text += "/" + "*".join(denominator)
        if self.scale == 1.0:
            return text
        return f"{self.scale:g}*{text}" if text != "1" else f"{self.scale:g}"

    def _display_parts(self):
        numerator = []
        denominator = []
        for symbol, exponent in self.dimension.items():
            if exponent > 0:
                numerator.append(_symbol_with_exponent(symbol, exponent))
            else:
                denominator.append(_symbol_with_exponent(symbol, -exponent))
        return numerator, denominator

    def __repr__(self):
        return f"Unit({str(self)!r})"


DIMENSIONLESS = Unit("1", {}, 1.0, 0.0, "dimensionless")


class Quantity:
    """A value with a unit, with dimensional checking on arithmetic."""

    __slots__ = ("value", "unit")
    __array_priority__ = 1000

    def __init__(self, value, unit: Unit):
        if not isinstance(unit, Unit):
            raise TypeError("Quantity needs a Unit; use Q(value, 'name') instead")
        self.value = value
        self.unit = unit

    @property
    def magnitude(self):
        """The numeric value (alias for ``.value``)."""
        return self.value

    @property
    def base_value(self):
        """The value expressed in base units."""
        return self.unit.to_base(self.value)

    def to(self, target) -> "Quantity":
        """Return this quantity converted to ``target`` (a Unit or name)."""
        if isinstance(target, str):
            target = SI[target]
        if not isinstance(target, Unit):
            raise TypeError("target must be a Unit or a unit name")
        return Quantity(self.unit.convert(self.value, target), target)

    # --- helpers ----------------------------------------------------------

    @staticmethod
    def _build(value, unit: Unit) -> "Quantity":
        """Build a quantity, folding an unnamed scale into the value."""
        should_fold = not unit.symbol and unit.offset == 0.0 and unit.scale != 1.0
        if not should_fold:
            return Quantity(value, unit)

        base_unit = Unit("", unit.dimension, 1.0, 0.0)
        return Quantity(value * unit.scale, base_unit)

    def _coerce(self, other):
        """Convert `other` into this quantity's unit for +/- and comparisons."""
        if isinstance(other, Quantity):
            if not self.unit.compatible(other.unit):
                raise DimensionError(
                    f"incompatible dimensions: {self.unit} and {other.unit}"
                )
            return other.unit.convert(other.value, self.unit)
        if self.unit.dimension:
            raise DimensionError(
                f"cannot combine a dimensionless value with {self.unit}"
            )
        return self.unit.from_base(other)

    # --- arithmetic -------------------------------------------------------

    def __add__(self, other):
        return Quantity(self.value + self._coerce(other), self.unit)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return Quantity(self.value - self._coerce(other), self.unit)

    def __rsub__(self, other):
        return Quantity(self._coerce(other) - self.value, self.unit)

    def __mul__(self, other):
        if isinstance(other, Quantity):
            return self._build(self.value * other.value, self.unit * other.unit)
        if isinstance(other, Unit):
            return self._build(self.value, self.unit * other)
        return Quantity(self.value * other, self.unit)

    def __rmul__(self, other):
        if isinstance(other, Unit):
            return self._build(self.value, other * self.unit)
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, Quantity):
            return self._build(self.value / other.value, self.unit / other.unit)
        if isinstance(other, Unit):
            return self._build(self.value, self.unit / other)
        return Quantity(self.value / other, self.unit)

    def __rtruediv__(self, other):
        if isinstance(other, Unit):
            return self._build(1.0 / self.value, other / self.unit)
        return self._build(other / self.value, 1.0 / self.unit)

    def __pow__(self, power):
        if isinstance(power, Quantity):
            if power.unit.dimension:
                raise DimensionError("exponent must be dimensionless")
            power = power.value
        return self._build(self.value ** power, self.unit ** power)

    def __rpow__(self, base):
        return Quantity(base, DIMENSIONLESS) ** self

    def __neg__(self):
        return Quantity(-self.value, self.unit)

    def __pos__(self):
        return self

    def __abs__(self):
        return Quantity(abs(self.value), self.unit)

    # --- comparison -------------------------------------------------------

    def __eq__(self, other):
        try:
            return bool(self.value == self._coerce(other))
        except DimensionError:
            return False

    def __lt__(self, other):
        return bool(self.value < self._coerce(other))

    def __le__(self, other):
        return bool(self.value <= self._coerce(other))

    def __gt__(self, other):
        return bool(self.value > self._coerce(other))

    def __ge__(self, other):
        return bool(self.value >= self._coerce(other))

    def __float__(self):
        if self.unit.dimension:
            raise TypeError(f"cannot convert {self!r} to float; use .value or .to(...)")
        return float(self.unit.to_base(self.value))

    # --- numpy interop ----------------------------------------------------

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method != "__call__" or kwargs.get("out") is not None:
            return NotImplemented

        if ufunc is np.multiply:
            return _multiply_operands(*inputs)
        if ufunc in (np.divide, np.true_divide):
            return _divide_operands(*inputs)
        if ufunc in (np.add, np.subtract):
            left, right = inputs
            if isinstance(left, Unit) or isinstance(right, Unit):
                return NotImplemented
            if ufunc is np.add:
                return _as_quantity(left) + _as_quantity(right)
            return _as_quantity(left) - _as_quantity(right)
        if ufunc is np.power:
            return _as_quantity(inputs[0]) ** inputs[1]
        if ufunc is np.negative:
            return -self
        if ufunc is np.positive:
            return self
        return NotImplemented

    # --- display ----------------------------------------------------------

    @property
    def _is_plain_number(self) -> bool:
        return not self.unit.dimension and self.unit.scale == 1.0

    def __repr__(self):
        text = repr(self.value)
        return text if self._is_plain_number else f"{text} {self.unit}"

    def __str__(self):
        text = str(self.value)
        return text if self._is_plain_number else f"{text} {self.unit}"


def _as_quantity(value) -> Quantity:
    """Treat a plain number or array as a dimensionless quantity."""
    if isinstance(value, Quantity):
        return value
    return Quantity(value, DIMENSIONLESS)


def _multiply_operands(left, right):
    if isinstance(left, Unit):
        if isinstance(right, Unit):
            return left * right
        if isinstance(right, Quantity):
            return right * left
        return Quantity._build(right, left)

    if isinstance(left, Quantity):
        if isinstance(right, Unit):
            return left * right
        if isinstance(right, Quantity):
            return left * right
        return Quantity(left.value * right, left.unit)

    if isinstance(right, Unit):
        return Quantity._build(left, right)
    if isinstance(right, Quantity):
        return Quantity(left * right.value, right.unit)
    return NotImplemented


def _divide_operands(left, right):
    if isinstance(left, Unit):
        if isinstance(right, Unit):
            return left / right
        if isinstance(right, Quantity):
            return Quantity._build(1.0 / right.value, left / right.unit)
        return Quantity._build(1.0 / right, left)

    if isinstance(left, Quantity):
        if isinstance(right, Unit):
            return left / right
        if isinstance(right, Quantity):
            return left / right
        return Quantity(left.value / right, left.unit)

    if isinstance(right, Unit):
        return Quantity._build(left, right ** -1)
    if isinstance(right, Quantity):
        return Quantity(left / right.value, 1.0 / right.unit)
    return NotImplemented


class UnitRegistry:
    """A named collection of units and an expression parser."""

    def __init__(self, name: str = "units"):
        self.name = name
        self.units: dict[str, Unit] = {}
        self.aliases: dict[str, Unit] = {}
        self.constants = {"pi": math.pi, "tau": math.tau, "e": math.e}
        self.register(DIMENSIONLESS)

    # --- definition -------------------------------------------------------

    def define_base(self, symbol: str, name: str | None = None) -> Unit:
        """Define (or return) a base unit."""
        if symbol in self.units:
            return self.units[symbol]

        _require_identifier(symbol, "base unit")
        unit = Unit(symbol, {symbol: 1}, 1.0, 0.0, name or symbol)
        self.units[symbol] = unit
        return unit

    def base(self, *symbols: str) -> tuple[Unit, ...]:
        """Define and return several base units at once."""
        return tuple(self.define_base(symbol) for symbol in symbols)

    def register(self, unit: Unit, aliases: Iterable[str] = ()) -> Unit:
        """Add an already-built unit, optionally with extra names."""
        _validate_symbol(unit.symbol)
        self.units[unit.symbol] = unit
        for alias in aliases:
            if alias:
                _validate_symbol(alias)
                self.aliases[alias] = unit
        return unit

    def define(
        self,
        symbol: str,
        expression: str,
        name: str | None = None,
        aliases: Iterable[str] = (),
    ) -> Unit:
        """Define ``symbol`` from an expression such as ``"kg*m/s^2"``."""
        _require_identifier(symbol, "derived unit")
        parsed = self.parse(expression)
        unit = Unit(symbol, parsed.dimension, parsed.scale, parsed.offset, name or symbol)
        return self.register(unit, aliases=aliases)

    # --- lookup and parsing ----------------------------------------------

    def __getitem__(self, key: str) -> Unit:
        if key in self.units:
            return self.units[key]
        if key in self.aliases:
            return self.aliases[key]
        return self.parse(key)

    def __contains__(self, key: str) -> bool:
        return key in self.units or key in self.aliases

    def parse(self, expression: str) -> Unit:
        """Parse a unit expression into a :class:`Unit`."""
        if not isinstance(expression, str):
            raise TypeError("unit expression must be a string")
        text = expression.strip()
        if not text:
            return DIMENSIONLESS
        text = text.replace("^", "**")
        try:
            node = ast.parse(text, mode="eval").body
        except SyntaxError as error:
            raise UnitError(f"invalid unit expression {expression!r}") from error
        return self._evaluate(node, expression)

    def _lookup(self, name: str, expression: str) -> Unit:
        if name in self.units:
            return self.units[name]
        if name in self.aliases:
            return self.aliases[name]
        if name in self.constants:
            return Unit("", {}, self.constants[name])
        raise UnknownUnitError(
            f"unknown unit or constant {name!r} in {expression!r}"
        )

    def _evaluate(self, node, expression: str) -> Unit:
        if isinstance(node, ast.Name):
            return self._lookup(node.id, expression)

        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, (int, float))
            and not isinstance(node.value, bool)
        ):
            return Unit("", {}, float(node.value))

        if isinstance(node, ast.BinOp):
            return self._evaluate_binary(node, expression)
        if isinstance(node, ast.UnaryOp):
            return self._evaluate_unary(node, expression)
        raise UnitError(f"unsupported unit expression: {expression!r}")

    def _evaluate_binary(self, node, expression: str) -> Unit:
        left = self._evaluate(node.left, expression)
        right = self._evaluate(node.right, expression)

        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Pow):
            if right.dimension:
                raise UnitError(f"exponent must be dimensionless in {expression!r}")
            return left ** right.scale
        raise UnitError(f"unsupported operator in {expression!r}")

    def _evaluate_unary(self, node, expression: str) -> Unit:
        operand = self._evaluate(node.operand, expression)

        if isinstance(node.op, ast.UAdd):
            return operand
        if isinstance(node.op, ast.USub):
            if operand.dimension:
                raise UnitError(f"cannot negate a unit in {expression!r}")
            return Unit("", {}, -operand.scale)
        raise UnitError(f"unsupported unary operator in {expression!r}")

    # --- quantities -------------------------------------------------------

    def quantity(self, value, unit) -> Quantity:
        """Build a quantity; ``unit`` may be a Unit or a unit name."""
        if isinstance(unit, str):
            unit = self[unit]
        return Quantity(value, unit)

    def parse_quantity(self, text: str) -> Quantity:
        """Parse a quantity string such as ``"9.81 m/s^2"`` or ``"100 km/h"``."""
        if not isinstance(text, str):
            raise TypeError("quantity text must be a string")
        match = _QUANTITY_PATTERN.match(text)
        if match is None:
            raise UnitError(f"invalid quantity: {text!r}")
        value = float(match.group(1))
        unit_text = match.group(2) or "1"
        return self.quantity(value, unit_text)

    def __call__(self, text: str) -> Quantity:
        """Alias for :meth:`parse_quantity`."""
        return self.parse_quantity(text)

    # --- definition blocks ------------------------------------------------

    def load(self, text: str) -> "UnitRegistry":
        """Load ``unit name`` / ``unit name = expression`` lines.

        Base units are defined before derived ones, so base definitions may
        appear anywhere in the block.  Derived units may reference units defined
        earlier (or any base unit).  Blank lines and ``#`` comments are ignored.
        """
        base_definitions: list[str] = []
        derived_definitions: list[tuple[int, str, str]] = []

        for line_number, raw in enumerate(text.splitlines(), 1):
            line = raw.partition("#")[0].strip()
            if not line:
                continue

            parts = line.split(None, 1)
            keyword = parts[0]
            body = parts[1].strip() if len(parts) == 2 else ""
            if keyword not in ("unit", "base"):
                raise UnitError(
                    f"line {line_number}: expected 'unit' or 'base', got {keyword!r}"
                )
            if not body:
                raise UnitError(f"line {line_number}: missing unit name")

            if "=" in body:
                symbol, _, expression = body.partition("=")
                derived_definitions.append(
                    (line_number, symbol.strip(), expression.strip())
                )
            else:
                base_definitions.append(body)

        for symbol in base_definitions:
            self.define_base(symbol)
        for line_number, symbol, expression in derived_definitions:
            try:
                self.define(symbol, expression)
            except UnitError as error:
                raise UnitError(f"line {line_number}: {error}") from error
        return self

    @classmethod
    def from_text(cls, text: str, name: str = "units") -> "UnitRegistry":
        """Create a registry from a definition block."""
        return cls(name).load(text)


class UnitSystem(UnitRegistry):
    """A registry that creates base units and exposes them as attributes.

    Example
    -------
    >>> u = UnitSystem("m", "s", "kg")
    >>> N = u.define("N", "kg*m/s^2")
    >>> distance = 10 * u.m
    >>> speed = distance / (2 * u.s)
    >>> speed
    5.0 m/s
    """

    def __init__(self, *base_symbols: str, name: str = "units"):
        super().__init__(name)
        self.base(*base_symbols)

    def __getattr__(self, name: str) -> Unit:
        if name.startswith("_"):
            raise AttributeError(name)

        units = self.__dict__.get("units", {})
        aliases = self.__dict__.get("aliases", {})
        if name in units:
            return units[name]
        if name in aliases:
            return aliases[name]
        raise AttributeError(f"{type(self).__name__} has no unit {name!r}")


def base_units(*symbols: str, name: str = "units") -> UnitSystem:
    """Create a :class:`UnitSystem` from base-unit names.

    ``u = base_units("m", "s", "kg")`` is shorthand for
    ``UnitSystem("m", "s", "kg")``.
    """
    return UnitSystem(*symbols, name=name)


# --- Ready-made SI registry --------------------------------------------------

SI = UnitRegistry("SI")

_SI_BASE_UNITS = (
    ("m", "metre"),
    ("kg", "kilogram"),
    ("s", "second"),
    ("A", "ampere"),
    ("K", "kelvin"),
    ("mol", "mole"),
    ("cd", "candela"),
)

_SI_DERIVED_UNITS = (
    ("Hz", "1/s"),
    ("N", "kg*m/s^2"),
    ("Pa", "N/m^2"),
    ("J", "N*m"),
    ("W", "J/s"),
    ("C", "A*s"),
    ("V", "W/A"),
    ("F", "C/V"),
    ("ohm", "V/A"),
    ("S", "1/ohm"),
    ("Wb", "V*s"),
    ("T", "Wb/m^2"),
    ("H", "Wb/A"),
    ("lm", "cd*sr"),
    ("lx", "lm/m^2"),
    ("Bq", "1/s"),
    ("Gy", "J/kg"),
    ("Sv", "J/kg"),
    ("kat", "mol/s"),
)

_SI_COMMON_UNITS = (
    ("L", "1e-3*m^3"),
    ("mL", "1e-3*L"),
    ("min", "60*s"),
    ("h", "3600*s"),
    ("day", "24*h"),
    ("g", "1e-3*kg"),
    ("mg", "1e-3*g"),
    ("ug", "1e-6*g"),
    ("km", "1e3*m"),
    ("cm", "1e-2*m"),
    ("mm", "1e-3*m"),
    ("um", "1e-6*m"),
    ("nm", "1e-9*m"),
    ("ms", "1e-3*s"),
    ("us", "1e-6*s"),
    ("ns", "1e-9*s"),
    ("kHz", "1e3*Hz"),
    ("MHz", "1e6*Hz"),
    ("GHz", "1e9*Hz"),
    ("kN", "1e3*N"),
    ("MN", "1e6*N"),
    ("kJ", "1e3*J"),
    ("MJ", "1e6*J"),
    ("kW", "1e3*W"),
    ("MW", "1e6*W"),
    ("kPa", "1e3*Pa"),
    ("MPa", "1e6*Pa"),
    ("GPa", "1e9*Pa"),
    ("bar", "1e5*Pa"),
    ("atm", "101325*Pa"),
    ("eV", "1.602176634e-19*J"),
    ("deg", "pi/180*rad"),
)

for symbol, name in _SI_BASE_UNITS:
    SI.define_base(symbol, name)

SI.register(Unit("rad", {}, 1.0, 0.0, "radian"))
SI.register(Unit("sr", {}, 1.0, 0.0, "steradian"))

for symbol, expression in _SI_DERIVED_UNITS:
    SI.define(symbol, expression)
for symbol, expression in _SI_COMMON_UNITS:
    SI.define(symbol, expression)

_SI_AFFINE_UNITS = (
    ("degC", 1.0, 273.15, "degree Celsius", ("Celsius",)),
    ("degF", 5.0 / 9.0, 255.3722222222222, "degree Fahrenheit", ("Fahrenheit",)),
)

for symbol, scale, offset, name, aliases in _SI_AFFINE_UNITS:
    SI.register(
        Unit(symbol, SI["K"].dimension, scale, offset, name),
        aliases=aliases,
    )


# --- Convenient module-level API --------------------------------------------

def unit(
    symbol: str,
    expression: str | None = None,
    *,
    registry: UnitRegistry = SI,
) -> Unit:
    """Get or define a unit.

    ``unit("m")`` returns the existing metre (or defines a base unit if new);
    ``unit("N", "kg*m/s^2")`` defines a derived unit.
    """
    if expression is None:
        if symbol in registry:
            return registry[symbol]
        return registry.define_base(symbol)
    return registry.define(symbol, expression)


def units(*symbols: str, registry: UnitRegistry = SI) -> tuple[Unit, ...]:
    """Return several units at once: ``m, s, kg = units("m", "s", "kg")``."""
    return tuple(registry[symbol] for symbol in symbols)


def Q(value, unit=None, *, registry: UnitRegistry = SI) -> Quantity:
    """Build a quantity: ``Q(9.81, "m/s^2")`` or ``Q("9.81 m/s^2")``."""
    if unit is None:
        if isinstance(value, str):
            return registry.parse_quantity(value)
        raise TypeError("Q needs a unit")
    if isinstance(unit, str):
        unit = registry[unit]
    return Quantity(value, unit)


def load_units(
    text: str,
    *,
    name: str = "units",
    base: UnitRegistry | None = None,
) -> UnitRegistry:
    """Build a registry from a definition block.

    Example::

        ureg = load_units('''
            unit m
            unit s
            unit kg
            unit N = kg*m/s^2
        ''')
        force = ureg.quantity(10, "N")
    """
    registry = UnitRegistry(name)
    if base is not None:
        registry.units.update(base.units)
        registry.aliases.update(base.aliases)
        registry.constants.update(base.constants)
    return registry.load(text)


__all__ = [
    "Unit",
    "Quantity",
    "UnitRegistry",
    "UnitSystem",
    "UnitError",
    "DimensionError",
    "UnknownUnitError",
    "DIMENSIONLESS",
    "SI",
    "unit",
    "units",
    "base_units",
    "Q",
    "load_units",
]

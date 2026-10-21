from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias, cast

import numpy as np
import numpy.typing as npt
import sympy
from sympy import Expr, Symbol
from sympy.parsing.sympy_parser import parse_expr

from src.models import FloatArray


Coordinate: TypeAlias = float | npt.NDArray[np.float64]
ExactSolution: TypeAlias = Callable[[Coordinate, Coordinate], object]

X = Symbol("x", real=True)
Y = Symbol("y", real=True)
Z = Symbol("z")


def _require_expression(value: object, *, name: str) -> Expr:
    if not isinstance(value, Expr):
        raise TypeError(f"{name} is not a SymPy expression")
    return value


def is_harmonic(expression: Expr) -> bool:
    """Return whether ``expression`` has a symbolically zero 2D Laplacian."""

    unsupported_symbols = expression.free_symbols - {X, Y}
    if unsupported_symbols:
        names = ", ".join(sorted(str(symbol) for symbol in unsupported_symbols))
        raise ValueError(f"Expression contains unsupported symbols: {names}")

    derivative_x = _require_expression(
        cast(object, sympy.diff(expression, X, 2)), name="x derivative"
    )
    derivative_y = _require_expression(
        cast(object, sympy.diff(expression, Y, 2)), name="y derivative"
    )
    laplacian = _require_expression(
        cast(object, sympy.simplify(derivative_x + derivative_y)), name="Laplacian"
    )
    return laplacian.equals(0) is True


def real_part_from_complex_function(function_text: str) -> Expr:
    """Parse ``f(z)`` and return its harmonic real part ``u(x, y)``.

    The input is interpreted as a SymPy expression in the single complex
    variable ``z``. A ``ValueError`` is raised if its real part is not
    harmonic.
    """

    if not function_text.strip():
        raise ValueError("The complex function must not be empty")

    parsed = _require_expression(
        cast(object, parse_expr(function_text, local_dict={"z": Z})),
        name="parsed function",
    )
    unsupported_symbols = parsed.free_symbols - {Z}
    if unsupported_symbols:
        names = ", ".join(sorted(str(symbol) for symbol in unsupported_symbols))
        raise ValueError(f"f(z) may only contain z; found: {names}")

    imaginary_unit = _require_expression(cast(object, sympy.I), name="imaginary unit")
    substituted = _require_expression(
        cast(object, parsed.subs(Z, X + imaginary_unit * Y)), name="f(x + iy)"
    )
    expanded = _require_expression(
        cast(object, sympy.expand_complex(substituted)), name="expanded f"
    )
    real_part = _require_expression(expanded.as_real_imag()[0], name="real part")
    real_part = _require_expression(
        cast(object, sympy.simplify(real_part)), name="simplified real part"
    )

    if not is_harmonic(real_part):
        raise ValueError("The real part of the supplied function is not harmonic")

    return real_part


def make_exact_solution(expression: Expr) -> ExactSolution:
    """Convert a symbolic ``u(x, y)`` expression to a NumPy-aware callable."""

    unsupported_symbols = expression.free_symbols - {X, Y}
    if unsupported_symbols:
        names = ", ".join(sorted(str(symbol) for symbol in unsupported_symbols))
        raise ValueError(f"Expression contains unsupported symbols: {names}")

    return cast(
        ExactSolution,
        cast(object, sympy.lambdify((X, Y), expression, modules="numpy")),
    )


def evaluate_exact_solution(
    exact_solution: ExactSolution,
    x: Coordinate,
    y: Coordinate,
) -> FloatArray:
    """Evaluate and broadcast an exact solution, rejecting invalid values."""

    x_values, y_values = np.broadcast_arrays(
        np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    )
    try:
        raw_values = np.asarray(exact_solution(x_values, y_values), dtype=np.complex128)
    except (TypeError, ValueError) as error:
        raise ValueError("Exact solution did not return numeric values") from error

    if not np.allclose(raw_values.imag, 0.0, atol=1e-12, rtol=0.0):
        raise ValueError("Exact solution returned complex values")

    try:
        values = np.array(
            np.broadcast_to(raw_values.real, x_values.shape),
            dtype=np.float64,
            copy=True,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("Exact solution did not return real numeric values") from error

    if not np.all(np.isfinite(values)):
        raise ValueError("Exact solution is not finite throughout the grid")

    return values

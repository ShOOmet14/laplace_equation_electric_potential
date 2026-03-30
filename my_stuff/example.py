from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, TypeAlias

import sympy
from sympy import Expr, I, Symbol
from sympy.parsing.sympy_parser import parse_expr


RealNumber: TypeAlias = float | int


@dataclass(frozen=True)
class BoundaryConditions:
    left: Expr
    right: Expr
    bottom: Expr
    top: Expr


def require_expr(value: object, *, name: str) -> Expr:
    """
    Runtime check + static narrowing helper.

    SymPy operations often come back as broad types in type checkers.
    This function makes sure the value is really an Expr.
    """
    if not isinstance(value, Expr):
        raise TypeError(f"{name} is not a SymPy Expr, got {type(value).__name__}")
    return value


def build_real_part(function_text: str, x: Symbol, y: Symbol, z: Symbol) -> Expr:
    parsed: Expr = require_expr(parse_expr(function_text), name="parsed")

    substituted: Expr = require_expr(parsed.subs(z, x + I * y), name="substituted")

    # SymPy docs: complex=True splits into real/imaginary structure
    expanded: Expr = require_expr(
        sympy.expand(substituted, complex=True),
        name="expanded",
    )

    real_part: Expr = require_expr(expanded.as_real_imag()[0], name="real_part")

    simplified_real_part: Expr = require_expr(
        sympy.simplify(real_part),
        name="simplified_real_part",
    )

    print(f"Parsed function as expression: {parsed}")
    print(f"After substitution f(x,y) = {substituted}")
    print(f"After expanding: f(x,y) = {expanded}")
    print(f"Real part of Re(f(x,y)) = u(x,y) = {real_part}")
    print(f"After simplifying u(x,y) = {simplified_real_part}")

    return simplified_real_part


def build_boundary_conditions(u_xy: Expr, x: Symbol, y: Symbol) -> BoundaryConditions:
    left: Expr = require_expr(u_xy.subs(x, 0), name="left")
    right: Expr = require_expr(u_xy.subs(x, 1), name="right")
    bottom: Expr = require_expr(u_xy.subs(y, 0), name="bottom")
    top: Expr = require_expr(u_xy.subs(y, 1), name="top")

    left = require_expr(sympy.simplify(left), name="left_simplified")
    right = require_expr(sympy.simplify(right), name="right_simplified")
    bottom = require_expr(sympy.simplify(bottom), name="bottom_simplified")
    top = require_expr(sympy.simplify(top), name="top_simplified")

    return BoundaryConditions(
        left=left,
        right=right,
        bottom=bottom,
        top=top,
    )


def build_numeric_function(
    u_xy: Expr, x: Symbol, y: Symbol
) -> Callable[[RealNumber, RealNumber], RealNumber]:
    return sympy.lambdify((x, y), u_xy, "math")


def main() -> None:
    x: Symbol = Symbol("x", real=True)
    y: Symbol = Symbol("y", real=True)
    z: Symbol = Symbol("z")

    user_input: str = input("Podaj funkcję zmiennej zespolonej f(z) = ")

    try:
        u_xy: Expr = build_real_part(user_input, x, y, z)
        bc: BoundaryConditions = build_boundary_conditions(u_xy, x, y)

        print(
            "-" * 20,
            f"\nBoundary conditions for this real part are:"
            f"\nLeft = {bc.left}"
            f"\nBottom = {bc.bottom}"
            f"\nRight = {bc.right}"
            f"\nTop = {bc.top}",
        )

        u_function: Callable[[RealNumber, RealNumber], RealNumber] = (
            build_numeric_function(u_xy, x, y)
        )
        value_at_half: RealNumber = u_function(0.5, 0.5)

        print(
            f"Wartość części rzeczywistej dla x = 0.5 oraz y = 0.5 to: {value_at_half}"
        )

    except Exception as error:
        print(f"Błąd: {error}")


if __name__ == "__main__":
    main()

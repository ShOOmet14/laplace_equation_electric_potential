from typing import cast
from collections.abc import Callable

import sympy
from sympy import I, Symbol, Expr


def require_expr(value: object, name: str) -> Expr:
    """
    This function makes sure all SymPy exprensions are really of the type Expr, narrows down SymPy types

    Args:
    - variable value to check expression type
    - name of the variable that is being checked

    Output:
    - variable value after checking if it has correct type
    """

    if not isinstance(value, Expr):
        raise TypeError(f"{name} is not a SymPy Expr, got {type(value).__name__}")

    return value


def main():
    x: Symbol = Symbol("x", real=True)
    y: Symbol = Symbol("y", real=True)
    z: Symbol = Symbol("z")

    user_input: str = input("Provide complex function formula f(z) = ")

    function: Expr = require_expr(sympy.parse_expr(user_input), "function")
    print(f"Parsed function as expression: {function}")

    function_substitution: Expr = require_expr(
        function.subs(z, x + I * y), "function_substitution"
    )
    print(f"After substitution f(x,y) = {function_substitution}")

    # For complicated functions I need to expand them first into real part and complex part as specified in SymPy docs
    expanded = require_expr(
        cast(object, sympy.expand(function_substitution, complex=True)),
        "expanded",
    )
    print(f"After expanding: f(x,y) = {expanded}")

    # as_real_imag gives a tuple of (Re, Im) and I only need Re
    parts = expanded.as_real_imag()
    if parts is None:
        raise TypeError("as_real_imag() returned None")

    real_part: Expr = require_expr(parts[0], "real_part")
    print(f"Real part of Re(f(x,y)) = u(x,y) = {real_part}")

    simplified_real_part: Expr = require_expr(
        cast(object, sympy.simplify(real_part)), "simplified_real_part"
    )
    print(f"After simplifying u(x,y) = {simplified_real_part}")

    # Boundary conditions
    left: Expr = require_expr(
        cast(object, sympy.simplify(simplified_real_part.subs(x, 0))), "left"
    )
    right: Expr = require_expr(
        cast(object, sympy.simplify(simplified_real_part.subs(x, 1))), "right"
    )
    bottom: Expr = require_expr(
        cast(object, sympy.simplify(simplified_real_part.subs(y, 0))), "bottom"
    )
    top: Expr = require_expr(
        cast(object, sympy.simplify(simplified_real_part.subs(y, 1))), "top"
    )

    print(
        "-" * 20,
        f"\nBoundary conditions after simplification for this real part are:"
        f"\nLeft = {left}"
        f"\nBottom = {bottom}"
        f"\nRight = {right}"
        f"\nTop = {top}",
    )

    # Changing expressions to callable functions using sympy lambdify, it transfroms sympy expressions into lambda functions
    u_function: Callable[[float, float], float] = sympy.lambdify(
        (x, y), simplified_real_part, "math"
    )
    print(
        "Wartości części rzeczywsitej dla x = 0.5 oraz y = 0.5 to: "
        f"{u_function(0.5, 0.5)}"
    )

    # Checking if the function f(z) is holomorphic
    derivative_x_x: Expr = require_expr(
        sympy.diff(simplified_real_part, x, 2), "derivative_x_x"
    )
    print(f"Second order derivative for x: {derivative_x_x}")

    derivative_y_y: Expr = require_expr(
        sympy.diff(simplified_real_part, y, 2), "derivative_y_y"
    )
    print(f"Second order derivative for y: {derivative_y_y}")

    laplacian: Expr = require_expr(
        cast(object, sympy.Add(derivative_x_x, derivative_y_y)), "laplacian"
    )
    print(f"Laplacian = {laplacian}")

    if laplacian == 0:
        print("This real part is harmonic and can be used as an exact solution.")
    else:
        print(
            "This real part is not harmonic, so it cannot be used as an exact Laplace solution."
        )


if __name__ == "__main__":
    main()

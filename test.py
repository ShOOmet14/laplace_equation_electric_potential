import sympy
from sympy import I, Symbol, Expr, Basic


def check_if_expr(value: object, name: str) -> Expr:
    if not isinstance(value, Expr):
        raise TypeError(f"{name} is not Expr, got {type(value).__name__}")
    return value


def main():
    x: Symbol = Symbol("x", real=True)
    y: Symbol = Symbol("y", real=True)
    z: Symbol = Symbol("z")

    user_input: str = input("Provide complex function formula f(z) = ")

    function: Expr = check_if_expr(sympy.parse_expr(user_input), "function")
    print(f"Parsed function as expression: {function}")

    function_substitution: Basic = function.subs(z, x + I * y)
    print(f"After substitution f(x,y) = {function_substitution}")

    # For complicated functions I need to expand them first into real part and complex part as specified in SymPy docs
    expanded: Expr = check_if_expr(
        sympy.expand(function_substitution, complex=True), "expanded"
    )
    print(type(expanded))
    # print(f"After expanding: f(x,y) = {expanded}")

    # as_real_imag gives a tuple of (Re, Im) and I only need Re
    function_parts = expanded.as_real_imag()
    if function_parts is None:
        raise TypeError("as_real_imag() returned None")

    real_part: Expr = cast(Expr, function_parts[0])
    print(type(real_part))

    # print(f"Real part of Re(f(x,y)) = u(x,y) = {real_part}")

    simplified_real_part = cast(Expr, sympy.simplify(real_part))
    print(type(simplified_real_part))
    # print(f"After simplifying u(x,y) = {simplified_real_part}")

    # Boundary conditions
    left = cast(Expr, sympy.simplify(simplified_real_part.subs(x, 0)))
    print(type(left))
    right = cast(Expr, sympy.simplify(simplified_real_part.subs(x, 1)))
    print(type(right))
    bottom = cast(Expr, sympy.simplify(simplified_real_part.subs(y, 0)))
    print(type(bottom))
    top = cast(Expr, sympy.simplify(simplified_real_part.subs(y, 1)))
    print(type(top))

    # print(
    #     "-" * 20,
    #     f"\nBoundary conditions for this real part are:"
    #     f"\nLeft = {left}"
    #     f"\nBottom = {bottom}"
    #     f"\nRight = {right}"
    #     f"\nTop = {top}",
    # )

    # print(
    #     f"Simplified boundary conditions for this real part are:"
    #     f"\nLeft = {left}"
    #     f"\nBottom = {bottom}"
    #     f"\nRight = {right}"
    #     f"\nTop = {top}",
    # )

    # Changing expressions to callable functions using sympy lambdify, it transfroms sympy expressions into lambda functions
    u_function = sympy.lambdify((x, y), simplified_real_part, "math")
    # print(
    #     "Wartości części rzeczywsitej dla x = 0.5 oraz y = 0.5 to: "
    #     f"{u_function(0.5, 0.5)}"
    # )


if __name__ == "__main__":
    main()

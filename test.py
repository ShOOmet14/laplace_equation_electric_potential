def main():

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

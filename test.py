import sympy
from sympy import I


def main():
    x, y = sympy.symbols("x y", real=True)
    z = sympy.symbols("z")

    user_input: str = input("Podaj funkcję zmiennej zespolonej f(z) = ")

    function = sympy.parse_expr(user_input)
    function_substitution = function.subs(z, x + I * y)

    # as_real_imag gives a touple of (Re, Im) and I only need Re
    real_part = function_substitution.as_real_imag()[0]

    print(real_part)


if __name__ == "__main__":
    main()

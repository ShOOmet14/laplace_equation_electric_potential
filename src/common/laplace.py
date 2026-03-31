from collections.abc import Collection, Callable
from typing import cast

import os
import subprocess
import numpy as np
import numpy.typing as npt
import sympy
from sympy import I, Symbol, Expr

from src.common.config import DOMAIN_LENGTH
from src.common.models import LaplaceResultTypes
from src.common.i18n import Translator

# Global variable for storing function as lamba function not expression
function_lambda: Callable[
    [float | npt.NDArray[np.float64], float | npt.NDArray[np.float64]],
    float | npt.NDArray[np.float64],
]


def clear_console() -> None:
    """
    Helper function that clears console on different operating systems
    """

    try:
        if os.name == "nt":
            subprocess.run(["cmd", "/c", "cls"], check=True)
        else:
            subprocess.run(["clear"], check=True)
    except Exception:
        # Fallback for terminals that support ANSI escape codes
        print("\033[2J\033[H", end="")


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


def get_user_function(tr: Translator) -> None:
    """
    Function that takes an input from the user, then checks if the complex function is harmonic or not (solved Laplace equation) and then runs function create_lambda_function() that stores a numpy lambda function after converting it from an expression

    Args:
    - None

    Output:
    - None
    """

    x: Symbol = Symbol("x", real=True)
    y: Symbol = Symbol("y", real=True)
    z: Symbol = Symbol("z")

    while True:
        clear_console()
        user_input: str = input(tr.t("input.function_prompt"))

        parsed_function: Expr = require_expr(
            sympy.parse_expr(user_input), "parsed_function"
        )

        function_substitution: Expr = require_expr(
            parsed_function.subs(z, x + I * y), "function_substitution"
        )

        # For complicated functions I need to expand them first into real part and complex part as specified in SymPy docs
        expanded = require_expr(
            cast(object, sympy.expand(function_substitution, complex=True)),
            "expanded",
        )

        # as_real_imag gives a tuple of (Re, Im) and I only need Re
        parts = expanded.as_real_imag()
        if parts is None:
            raise TypeError("as_real_imag() returned None")

        real_part: Expr = require_expr(parts[0], "real_part")

        simplified_real_part: Expr = require_expr(
            cast(object, sympy.simplify(real_part)), "simplified_real_part"
        )

        # Checking if the function f(z) is holomorphic
        derivative_x_x: Expr = require_expr(
            sympy.diff(simplified_real_part, x, 2), "derivative_x_x"
        )

        derivative_y_y: Expr = require_expr(
            sympy.diff(simplified_real_part, y, 2), "derivative_y_y"
        )

        laplacian: Expr = require_expr(
            cast(object, sympy.Add(derivative_x_x, derivative_y_y)), "laplacian"
        )

        if sympy.simplify(laplacian).equals(0):
            print(tr.t("messages.harmonic_ok"))
            break

        print(tr.t("messages.harmonic_bad"))
        input(tr.t("input.try_again"))

    create_lambda_function(simplified_real_part)


def create_lambda_function(
    real_part: Expr,
) -> None:
    """
    Function that turns a given symbolic expression into callable lambda function

    Args:
    - Real part of given function f(z), provided by the user in get_user_function()

    Output:
    - None
    """

    x: Symbol = Symbol("x", real=True)
    y: Symbol = Symbol("y", real=True)

    global function_lambda
    function_lambda = sympy.lambdify((x, y), real_part, "numpy")


def exact_solution(
    x: float | npt.NDArray[np.float64],
    y: float | npt.NDArray[np.float64],
) -> float | npt.NDArray[np.float64]:
    """
    Return the exact solution u(x, y).

    This function represents the real part of the complex-valued function f(z) introduced by the user.
    """
    global function_lambda

    return function_lambda(x, y)


def solve_laplace(
    grid_size: int,
    tolerance: float,
    max_iterations: int,
    iterations_to_save: Collection[int] | None = None,
    tr: Translator | None = None,
    print_summary: bool = False,
) -> LaplaceResultTypes:
    """
    Solve the Laplace equation on the square [0, 1] x [0, 1] using the Gauss-Seidel iterative method.

    Args:
        grid_size: Number of grid points in each direction.
        tolerance: Stopping criterion for the iteration.
        max_iterations: Maximum number of iterations.
        iterations_to_save: Iteration numbers for which solution snapshots are saved.

    Returns:
        A dictionary containing the grid, numerical solution, iteration count, snapshots, and error analysis data.
    """

    snapshot_iterations = set(iterations_to_save or ())

    x = np.linspace(0.0, DOMAIN_LENGTH, grid_size)
    y = np.linspace(0.0, DOMAIN_LENGTH, grid_size)
    grid_step = DOMAIN_LENGTH / (grid_size - 1)

    # Convention:
    # u[i, j] represents u[x_i, y_j], where i and j are the corresponding
    # indices for the X and Y axes.

    numerical_solution = np.zeros((grid_size, grid_size), dtype=float)

    numerical_solution[:, 0] = exact_solution(x, 0.0)
    numerical_solution[:, -1] = exact_solution(x, DOMAIN_LENGTH)
    numerical_solution[0, :] = exact_solution(0.0, y)
    numerical_solution[-1, :] = exact_solution(DOMAIN_LENGTH, y)

    iteration_snapshots: dict[int, npt.NDArray[np.float64]] = {}
    iteration_count = 0

    converged = False

    for iteration_count in range(1, max_iterations + 1):
        max_change = 0.0

        for i in range(1, grid_size - 1):
            for j in range(1, grid_size - 1):
                old_value = numerical_solution[i, j]

                new_value = 0.25 * (
                    numerical_solution[i + 1, j]
                    + numerical_solution[i - 1, j]
                    + numerical_solution[i, j + 1]
                    + numerical_solution[i, j - 1]
                )

                numerical_solution[i, j] = new_value

                change = abs(new_value - old_value)
                if change > max_change:
                    max_change = change

        if iteration_count in snapshot_iterations:
            iteration_snapshots[iteration_count] = numerical_solution.copy()

        if max_change < tolerance:
            converged = True
            break

    # Comparison of the numerical and exact solutions.
    x_grid, y_grid = np.meshgrid(x, y, indexing="ij")
    exact_solution_grid: npt.NDArray[np.float64] = np.asarray(
        exact_solution(x_grid, y_grid),
        dtype=np.float64,
    )

    error = np.abs(numerical_solution - exact_solution_grid)
    max_error = float(np.max(error))

    result: LaplaceResultTypes = {
        "x": x,
        "y": y,
        "numerical_solution": numerical_solution,
        "grid_step": grid_step,
        "iterations": iteration_count,
        "converged": converged,
        "iteration_snapshots": iteration_snapshots,
        "exact_solution": exact_solution_grid,
        "error": error,
        "max_error": max_error,
        "grid_size": grid_size,
    }

    if print_summary:
        if tr is None:
            raise ValueError("Translator must be provided when print_summary=True")

        if result["converged"]:
            print(tr.t("messages.converged", iterations=result["iterations"]))
        else:
            print(tr.t("messages.not_converged"))

        print(f"\n{tr.t('messages.plot_result_title')}")
        print(tr.t("messages.grid_size", grid_size=result["grid_size"]))
        print(tr.t("messages.grid_step", grid_step=result["grid_step"]))
        print(tr.t("messages.iterations_count", iterations=result["iterations"]))
        print(tr.t("messages.max_error_line", max_error=result["max_error"]))

    return result

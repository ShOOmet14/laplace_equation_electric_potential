from collections.abc import Collection

import numpy as np
import numpy.typing as npt

from src.common.config import DOMAIN_LENGTH
from src.common.models import LaplaceResultTypes


def exact_solution(
    x: float | npt.NDArray[np.float64],
    y: float | npt.NDArray[np.float64],
) -> float | npt.NDArray[np.float64]:
    """
    Return the exact solution u(x, y).

    This function represents the real part of the complex-valued function
    f(z) introduced by the user.
    """
    return np.exp(x) * np.cos(y)


def solve_laplace(
    grid_size: int,
    tolerance: float,
    max_iterations: int,
    iterations_to_save: Collection[int] | None = None,
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

    # Boundary conditions from the exact solution for the square [0, A] x [0, A],
    # where A is the side length.
    # In order: y -> 0, y -> side length, x -> 0, x -> side length.
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

    return {
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

from __future__ import annotations

from collections.abc import Collection

import numpy as np

from src.models import FloatArray, LaplaceResult
from src.symbolic import ExactSolution, evaluate_exact_solution


DOMAIN_LENGTH = 1.0


def solve_laplace(
    exact_solution: ExactSolution,
    *,
    grid_size: int,
    tolerance: float,
    max_iterations: int,
    snapshot_iterations: Collection[int] = (),
) -> LaplaceResult:
    """Solve Laplace's equation on ``[0, 1] x [0, 1]`` with Gauss-Seidel.

    Dirichlet boundary values are sampled from ``exact_solution``. Iteration
    stops when the largest pointwise update in a complete sweep is no greater
    than ``tolerance``. The five-point stencil is applied to all interior
    points.
    """

    if isinstance(grid_size, bool) or grid_size < 3:
        raise ValueError("grid_size must be an integer greater than or equal to 3")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be a positive finite number")
    if (
        isinstance(max_iterations, bool)
        or max_iterations < 1
    ):
        raise ValueError("max_iterations must be a positive integer")

    requested_snapshots = set(snapshot_iterations)
    if any(
        isinstance(iteration, bool)
        or iteration < 1
        for iteration in requested_snapshots
    ):
        raise ValueError("snapshot iterations must be positive integers")

    x = np.linspace(0.0, DOMAIN_LENGTH, grid_size, dtype=np.float64)
    y = np.linspace(0.0, DOMAIN_LENGTH, grid_size, dtype=np.float64)
    grid_spacing = DOMAIN_LENGTH / (grid_size - 1)
    x_grid, y_grid = np.meshgrid(x, y, indexing="ij")
    exact_values = evaluate_exact_solution(exact_solution, x_grid, y_grid)

    numerical_values = np.zeros_like(exact_values)
    numerical_values[0, :] = exact_values[0, :]
    numerical_values[-1, :] = exact_values[-1, :]
    numerical_values[:, 0] = exact_values[:, 0]
    numerical_values[:, -1] = exact_values[:, -1]

    snapshots: dict[int, FloatArray] = {}
    converged = False
    max_update = float("inf")
    iteration = 0

    for iteration in range(1, max_iterations + 1):
        max_update = 0.0

        for i in range(1, grid_size - 1):
            for j in range(1, grid_size - 1):
                old_value = numerical_values[i, j]
                new_value = 0.25 * (
                    numerical_values[i - 1, j]
                    + numerical_values[i + 1, j]
                    + numerical_values[i, j - 1]
                    + numerical_values[i, j + 1]
                )
                numerical_values[i, j] = new_value
                max_update = max(max_update, abs(new_value - old_value))

        if iteration in requested_snapshots:
            snapshots[iteration] = numerical_values.copy()

        if max_update <= tolerance:
            converged = True
            break

    absolute_error = np.abs(numerical_values - exact_values)
    interior_error = absolute_error[1:-1, 1:-1]

    return LaplaceResult(
        x=x,
        y=y,
        numerical_solution=numerical_values,
        exact_solution=exact_values,
        absolute_error=absolute_error,
        grid_spacing=grid_spacing,
        iterations=iteration,
        converged=converged,
        max_update=float(max_update),
        l_inf_error=float(np.max(interior_error)),
        rms_error=float(np.sqrt(np.mean(interior_error**2))),
        snapshots=snapshots,
    )

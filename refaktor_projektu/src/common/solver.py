from collections.abc import Collection

import numpy as np
import numpy.typing as npt

from src.common.models import GridStudyRecord, LaplaceResult


def exact_solution(
    x: float | npt.NDArray[np.float64],
    y: float | npt.NDArray[np.float64],
) -> float | npt.NDArray[np.float64]:
    """Zwraca rozwiązanie dokładne u(x, y)."""
    return np.exp(x) * np.cos(y)


def solve_laplace(
    *,
    grid_size: int,
    tolerance: float,
    max_iterations: int,
    side_length: float,
    iterations_to_save: Collection[int] | None = None,
) -> LaplaceResult:
    """Rozwiązuje równanie Laplace'a metodą Gaussa-Seidla."""

    iterations_set = set(iterations_to_save or ())

    x = np.linspace(0.0, side_length, grid_size)
    y = np.linspace(0.0, side_length, grid_size)
    grid_step = side_length / (grid_size - 1)

    u = np.zeros((grid_size, grid_size), dtype=float)

    u[:, 0] = exact_solution(x, 0.0)
    u[:, -1] = exact_solution(x, side_length)
    u[0, :] = exact_solution(0.0, y)
    u[-1, :] = exact_solution(side_length, y)

    snapshots: dict[int, npt.NDArray[np.float64]] = {}
    converged = False
    iteration = 0

    for iteration in range(1, max_iterations + 1):
        max_change = 0.0

        for i in range(1, grid_size - 1):
            for j in range(1, grid_size - 1):
                old_value = u[i, j]
                new_value = 0.25 * (
                    u[i + 1, j] + u[i - 1, j] + u[i, j + 1] + u[i, j - 1]
                )
                u[i, j] = new_value

                change = abs(new_value - old_value)
                if change > max_change:
                    max_change = change

        if iteration in iterations_set:
            snapshots[iteration] = u.copy()

        if max_change < tolerance:
            converged = True
            break

    x_mesh, y_mesh = np.meshgrid(x, y, indexing="ij")
    exact_grid: npt.NDArray[np.float64] = np.asarray(
        exact_solution(x_mesh, y_mesh),
        dtype=np.float64,
    )
    error = np.abs(u - exact_grid)
    max_error = float(np.max(error))

    return LaplaceResult(
        x=x,
        y=y,
        u=u,
        grid_step=grid_step,
        iterations=iteration,
        snapshots=snapshots,
        exact_u=exact_grid,
        error=error,
        max_error=max_error,
        grid_size=grid_size,
        converged=converged,
    )



def run_grid_study(
    *,
    grid_sizes: tuple[int, ...],
    tolerances: dict[int, float],
    max_iterations: int,
    side_length: float,
) -> list[GridStudyRecord]:
    """Uruchamia obliczenia dla kilku rozmiarów siatki."""

    results: list[GridStudyRecord] = []

    for grid_size in grid_sizes:
        result = solve_laplace(
            grid_size=grid_size,
            tolerance=tolerances[grid_size],
            max_iterations=max_iterations,
            side_length=side_length,
        )
        results.append(
            GridStudyRecord(
                grid_size=result.grid_size,
                grid_step=result.grid_step,
                iterations=result.iterations,
                max_error=result.max_error,
                converged=result.converged,
            )
        )

    return results

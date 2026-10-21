from __future__ import annotations

from collections.abc import Sequence
from math import isfinite, log

from src.models import LaplaceResult
from src.solver import solve_laplace
from src.symbolic import ExactSolution


def estimate_convergence_order(
    coarse_error: float,
    fine_error: float,
    coarse_spacing: float,
    fine_spacing: float,
) -> float:
    """Estimate ``p`` in the asymptotic relation ``error = C h**p``."""

    values = (coarse_error, fine_error, coarse_spacing, fine_spacing)
    if any(not isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("errors and grid spacings must be positive and finite")
    if fine_spacing >= coarse_spacing:
        raise ValueError("fine_spacing must be smaller than coarse_spacing")

    return log(coarse_error / fine_error) / log(coarse_spacing / fine_spacing)


def convergence_orders(results: Sequence[LaplaceResult]) -> list[float | None]:
    """Return pairwise observed orders, or ``None`` when an error is zero."""

    return [
        (
            estimate_convergence_order(
                coarse.l_inf_error,
                fine.l_inf_error,
                coarse.grid_spacing,
                fine.grid_spacing,
            )
            if coarse.l_inf_error > 0.0 and fine.l_inf_error > 0.0
            else None
        )
        for coarse, fine in zip(results, results[1:])
    ]


def run_grid_study(
    exact_solution: ExactSolution,
    *,
    grid_sizes: Sequence[int],
    tolerance: float,
    max_iterations: int,
) -> list[LaplaceResult]:
    """Solve on a strictly increasing sequence of grids for error analysis."""

    if len(grid_sizes) < 2:
        raise ValueError("grid study requires at least two grid sizes")
    if any(fine <= coarse for coarse, fine in zip(grid_sizes, grid_sizes[1:])):
        raise ValueError("grid sizes must be strictly increasing")

    return [
        solve_laplace(
            exact_solution,
            grid_size=grid_size,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )
        for grid_size in grid_sizes
    ]

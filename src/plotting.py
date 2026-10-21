from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from src.models import LaplaceResult


def _save_figure(fig: Figure, path: Path, *, show: bool) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return path


def plot_solution_comparison(
    result: LaplaceResult, output_dir: Path, *, show: bool = False
) -> Path:
    """Plot exact and numerical solutions with the same colour scale."""

    x_grid, y_grid = np.meshgrid(result.x, result.y, indexing="ij")
    value_min = min(result.exact_solution.min(), result.numerical_solution.min())
    value_max = max(result.exact_solution.max(), result.numerical_solution.max())

    fig = plt.figure(figsize=(13, 5.5))
    for position, values, title in (
        (1, result.exact_solution, "Exact harmonic solution"),
        (2, result.numerical_solution, "Gauss-Seidel solution"),
    ):
        axis = fig.add_subplot(1, 2, position, projection="3d")
        surface = axis.plot_surface(
            x_grid,
            y_grid,
            values,
            cmap="viridis",
            vmin=value_min,
            vmax=value_max,
            edgecolor="none",
        )
        axis.set(xlabel="x", ylabel="y", zlabel="u(x, y)", title=title)
        fig.colorbar(surface, ax=axis, shrink=0.65, pad=0.12)

    fig.suptitle(f"Laplace solution on a {result.grid_size} x {result.grid_size} grid")
    return _save_figure(fig, output_dir / "solution_comparison.png", show=show)


def plot_error_map(
    result: LaplaceResult, output_dir: Path, *, show: bool = False
) -> Path:
    """Plot the signed pointwise error ``u_h - u_exact``."""

    signed_error = result.numerical_solution - result.exact_solution
    limit = float(np.max(np.abs(signed_error)))
    fig, axis = plt.subplots(figsize=(7.5, 6))
    image = axis.imshow(
        signed_error.T,
        origin="lower",
        extent=(result.x[0], result.x[-1], result.y[0], result.y[-1]),
        cmap="coolwarm",
        vmin=-limit,
        vmax=limit,
        aspect="equal",
    )
    axis.set(xlabel="x", ylabel="y", title="Signed discretisation error: $u_h-u_{exact}$")
    fig.colorbar(image, ax=axis, label="error")
    return _save_figure(fig, output_dir / "error_map.png", show=show)


def plot_cross_sections(
    result: LaplaceResult,
    output_dir: Path,
    *,
    y_levels: Sequence[float] = (0.25, 0.50, 0.75),
    show: bool = False,
) -> Path:
    """Compare exact and numerical solutions along selected horizontal lines."""

    fig, axis = plt.subplots(figsize=(9, 5.5))
    for y_level in y_levels:
        index = int(np.argmin(np.abs(result.y - y_level)))
        (exact_line,) = axis.plot(
            result.x,
            result.exact_solution[:, index],
            linewidth=2,
            label=f"exact, y={result.y[index]:.2f}",
        )
        axis.plot(
            result.x,
            result.numerical_solution[:, index],
            "o",
            color=exact_line.get_color(),
            markersize=3,
            markevery=max(1, result.grid_size // 16),
            label=f"numerical, y={result.y[index]:.2f}",
        )

    axis.set(xlabel="x", ylabel="u(x, y)", title="Exact and numerical cross-sections")
    axis.grid(alpha=0.3)
    axis.legend(ncol=2, fontsize=9)
    return _save_figure(fig, output_dir / "cross_sections.png", show=show)


def plot_convergence(
    results: Sequence[LaplaceResult], output_dir: Path, *, show: bool = False
) -> Path:
    """Plot maximum interior error against grid spacing on log-log axes."""

    if len(results) < 2:
        raise ValueError("at least two grid results are required")

    spacing = np.array([result.grid_spacing for result in results])
    errors = np.array([result.l_inf_error for result in results])
    second_order_reference = errors[-1] * (spacing / spacing[-1]) ** 2

    fig, axis = plt.subplots(figsize=(7.5, 5.5))
    axis.loglog(spacing, errors, "o-", linewidth=2, label="$L_\\infty$ error")
    axis.loglog(spacing, second_order_reference, "--", label="$O(h^2)$ reference")
    axis.invert_xaxis()
    axis.set(
        xlabel="grid spacing h",
        ylabel="maximum interior absolute error",
        title="Grid-refinement study",
    )
    axis.grid(which="both", alpha=0.3)
    axis.legend()
    return _save_figure(fig, output_dir / "convergence.png", show=show)

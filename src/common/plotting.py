from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from src.common.config import Y_LEVELS_FOR_PLOTS, ITERATIONS_TO_SAVE
from src.common.models import LaplaceResultTypes


def get_y_level_indices(
    y: npt.NDArray[np.float64],
    y_levels: tuple[float, ...] = Y_LEVELS_FOR_PLOTS,
) -> list[int]:
    """
    Return indices of the values in `y` that are closest to the given y-levels.

    Args:
    - y: Array of y-coordinate values.
    - y_levels: Target y-values for which the closest indices should be found.

    Returns:
    - A list of indices corresponding to entries in `y` that are nearest to the specified y-levels.
    """

    indices: list[int] = []

    for y_level in y_levels:
        index = int(np.argmin(np.abs(y - y_level)))
        indices.append(index)

    return indices


def plot_cross_sections(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    solution_values: npt.NDArray[np.float64],
    output_dir: Path,
    labels: dict[str, str],
) -> None:
    """
    Plot function cross-sections for selected y-levels.

    The plot is saved to the output directory and displayed on screen.

    Args:
    - x: Array of x-coordinate values.
    - y: Array of y-coordinate values.
    - solution_values: 2D array containing function values on the grid.
    - output_dir: Directory where the output image will be saved.
    - labels: Dictionary with localized plot labels and output file names.

    Returns:
    - None.
    """

    width_inch: int = 11
    height_inch: int = 7
    plt.figure(figsize=(width_inch, height_inch))

    y_indices = get_y_level_indices(y)

    for j in y_indices:
        plt.plot(
            x,
            solution_values[:, j],
            linewidth=2.5,
            label=f"y = {y[j]:.2f}",
        )

    plt.xlabel(labels["x_label"])
    plt.ylabel(labels["y_label"])
    plt.title(labels["cross_sections_title"])
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / labels["cross_sections_file"], dpi=200)
    plt.show()
    plt.close()


def plot_cross_sections_for_iterations(
    result: LaplaceResultTypes,
    output_dir: Path,
    labels: dict[str, str],
) -> None:
    """
    Plot cross-sections of the approximate solution for selected iterations.

    For each iteration stored in the solver snapshots, the function compares
    the approximate solution with the exact solution and saves a separate plot.

    Args:
    - result: Solver result containing grid coordinates, the exact solution,
            and snapshots of intermediate iterations.
    - output_dir: Directory where the output images will be saved.
    - labels: Dictionary with localized plot labels and output file names.

    Returns:
    - None.
    """

    x = result["x"]
    y = result["y"]
    exact_values = result["exact_solution"]
    snapshots = result["iteration_snapshots"]

    y_indices = get_y_level_indices(y)

    for iteration in ITERATIONS_TO_SAVE:
        if iteration not in snapshots:
            continue

        approximate_values = snapshots[iteration]
        width_inch: int = 11
        height_inch: int = 7
        plt.figure(figsize=(width_inch, height_inch))

        for j in y_indices:
            (exact_line,) = plt.plot(
                x,
                exact_values[:, j],
                linewidth=2.0,
                label=f"{labels['exact_solution']}, y = {y[j]:.2f}",
            )

            color = exact_line.get_color()

            plt.plot(
                x,
                approximate_values[:, j],
                "o--",
                color=color,
                linewidth=1.2,
                markersize=3,
                markevery=2,
                label=f"{labels['numerical_solution']}, y = {y[j]:.2f}",
            )

        plt.xlabel(labels["x_label"])
        plt.ylabel(labels["y_label"])
        plt.title(labels["iteration_cross_sections_title"].format(iteration=iteration))
        plt.grid(True)
        plt.legend(ncol=2, fontsize=9)
        plt.tight_layout()
        plt.savefig(
            output_dir
            / labels["iteration_cross_sections_file"].format(iteration=iteration),
            dpi=200,
        )
        plt.show()
        plt.close()


def plot_final_cross_sections(
    result: LaplaceResultTypes,
    output_dir: Path,
    labels: dict[str, str],
) -> None:
    """
    Plot cross-sections of the final numerical solution.

    The function compares the converged numerical solution with the exact
    solution for selected y-levels, then saves and displays the plot.

    Args:
    - result: Solver result containing grid coordinates, the exact solution, and the final numerical solution.
    - output_dir: Directory where the output image will be saved.
    - labels: Dictionary with localized plot labels and output file names.

    Returns:
    - None.
    """

    x = result["x"]
    y = result["y"]
    exact_values = result["exact_solution"]
    numerical_values = result["numerical_solution"]

    width_inch: int = 11
    height_inch: int = 7
    plt.figure(figsize=(width_inch, height_inch))

    y_indices = get_y_level_indices(y)

    for j in y_indices:
        (exact_line,) = plt.plot(
            x,
            exact_values[:, j],
            linewidth=2.0,
            label=f"{labels['exact_solution']}, y = {y[j]:.2f}",
        )

        color = exact_line.get_color()

        plt.plot(
            x,
            numerical_values[:, j],
            "o--",
            color=color,
            linewidth=1.2,
            markersize=3,
            markevery=2,
            label=f"{labels['numerical_solution']}, y = {y[j]:.2f}",
        )

    plt.xlabel(labels["x_label"])
    plt.ylabel(labels["y_label"])
    plt.title(labels["final_cross_sections_title"])
    plt.grid(True)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(output_dir / labels["final_cross_sections_file"], dpi=200)
    plt.show()
    plt.close()


def plot_3d_solution(
    result: LaplaceResultTypes,
    output_dir: Path,
    labels: dict[str, str],
) -> None:
    """
    Plot the final numerical solution as a 3D surface.

    The generated plot is saved to the output directory and displayed
    on screen.

    Args:
    - result: Solver result containing grid coordinates and the final numerical solution.
    - output_dir: Directory where the output image will be saved.
    - labels: Dictionary with localized plot labels and output file names.

    Returns:
    - None.
    """

    x = result["x"]
    y = result["y"]
    numerical_values = result["numerical_solution"]

    x_grid, y_grid = np.meshgrid(x, y, indexing="ij")

    width_inch: int = 11
    height_inch: int = 7
    fig = plt.figure(figsize=(width_inch, height_inch))
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    surface = ax.plot_surface(
        x_grid,
        y_grid,
        numerical_values,
        cmap="viridis",
        edgecolor="none",
    )

    ax.set_xlabel(labels["x_label"])
    ax.set_ylabel(labels["y_axis_label"])
    ax.set_zlabel(labels["z_label"])
    ax.set_title(labels["solution_3d_title"])

    fig.colorbar(surface, ax=ax, shrink=0.75, pad=0.1)
    plt.tight_layout()
    plt.savefig(output_dir / labels["solution_3d_file"], dpi=200)
    plt.show()
    plt.close()


def plot_3d_exact_solution(
    result: LaplaceResultTypes,
    output_dir: Path,
    labels: dict[str, str],
) -> None:
    """
    Plot the exact analytical solution as a 3D surface.
    """

    x = result["x"]
    y = result["y"]
    exact_values = result["exact_solution"]

    x_grid, y_grid = np.meshgrid(x, y, indexing="ij")

    width_inch: int = 11
    height_inch: int = 7
    fig = plt.figure(figsize=(width_inch, height_inch))
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    surface = ax.plot_surface(
        x_grid,
        y_grid,
        exact_values,
        cmap="viridis",
        edgecolor="none",
    )

    ax.set_xlabel(labels["x_label"])
    ax.set_ylabel(labels["y_axis_label"])
    ax.set_zlabel(labels["z_label"])
    ax.set_title(labels["exact_solution_3d_title"])

    fig.colorbar(surface, ax=ax, shrink=0.75, pad=0.1)
    plt.tight_layout()
    plt.savefig(output_dir / labels["exact_solution_3d_file"], dpi=200)
    plt.show()
    plt.close()


def plot_3d_error(
    result: LaplaceResultTypes,
    output_dir: Path,
    labels: dict[str, str],
) -> None:
    """
    Plot the absolute error |u_num - u_exact| as a 3D surface.
    """

    x = result["x"]
    y = result["y"]
    error_values = result["error"]

    x_grid, y_grid = np.meshgrid(x, y, indexing="ij")

    width_inch: int = 11
    height_inch: int = 7
    fig = plt.figure(figsize=(width_inch, height_inch))
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    surface = ax.plot_surface(
        x_grid,
        y_grid,
        error_values,
        cmap="inferno",
        edgecolor="none",
    )

    ax.set_xlabel(labels["x_label"])
    ax.set_ylabel(labels["y_axis_label"])
    ax.set_zlabel(labels["error_z_label"])
    ax.set_title(labels["error_3d_title"])

    fig.colorbar(surface, ax=ax, shrink=0.75, pad=0.1)
    plt.tight_layout()
    plt.savefig(output_dir / labels["error_3d_file"], dpi=200)
    plt.show()
    plt.close()


def plot_error_map(
    result: LaplaceResultTypes,
    output_dir: Path,
    labels: dict[str, str],
) -> None:
    """
    Plot the signed difference u_num - u_exact as a matrix heatmap.
    """

    x = result["x"]
    y = result["y"]
    signed_error = result["numerical_solution"] - result["exact_solution"]

    width_inch: int = 8
    height_inch: int = 7
    plt.figure(figsize=(width_inch, height_inch))

    image = plt.imshow(
        signed_error.T,
        origin="lower",
        extent=(float(x[0]), float(x[-1]), float(y[0]), float(y[-1])),
        cmap="coolwarm",
        aspect="equal",
    )

    plt.xlabel(labels["x_label"])
    plt.ylabel(labels["y_axis_label"])
    plt.title(labels["error_map_title"])
    plt.colorbar(image, label=labels["error_map_colorbar_label"])
    plt.tight_layout()
    plt.savefig(output_dir / labels["error_map_file"], dpi=200)
    plt.show()
    plt.close()

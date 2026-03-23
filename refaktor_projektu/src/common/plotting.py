from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from src.common.models import LaplaceResult


def plot_sections(
    *,
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    grid_values: npt.NDArray[np.float64],
    y_levels: tuple[float, ...],
    images_dir: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    legend_label_template: str,
    output_filename: str,
) -> None:
    """Rysuje przekroje funkcji dla zadanych poziomów y."""

    plt.figure(figsize=(11, 7))

    for y_level in y_levels:
        j = int(np.argmin(np.abs(y - y_level)))
        plt.plot(
            x,
            grid_values[:, j],
            linewidth=2.5,
            label=legend_label_template.format(y_value=y[j]),
        )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(images_dir / output_filename, dpi=200)
    plt.show()



def plot_sections_for_iterations(
    *,
    result: LaplaceResult,
    y_levels: tuple[float, ...],
    iterations_to_plot: tuple[int, ...],
    images_dir: Path,
    title_template: str,
    xlabel: str,
    ylabel: str,
    exact_label_template: str,
    approx_label_template: str,
    output_filename_template: str,
) -> None:
    """Tworzy wykresy przekrojów dla zapisanych iteracji."""

    x = result.x
    y = result.y
    exact_u = result.exact_u
    snapshots = result.snapshots

    y_indexes = [int(np.argmin(np.abs(y - y_level))) for y_level in y_levels]

    for iteration in iterations_to_plot:
        if iteration not in snapshots:
            continue

        approx_u = snapshots[iteration]
        plt.figure(figsize=(11, 7))

        for j in y_indexes:
            (exact_line,) = plt.plot(
                x,
                exact_u[:, j],
                linewidth=2.0,
                label=exact_label_template.format(y_value=y[j]),
            )

            plt.plot(
                x,
                approx_u[:, j],
                "o--",
                color=exact_line.get_color(),
                linewidth=1.2,
                markersize=3,
                markevery=2,
                label=approx_label_template.format(y_value=y[j]),
            )

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title_template.format(iteration=iteration))
        plt.grid(True)
        plt.legend(ncol=2, fontsize=9)
        plt.tight_layout()
        plt.savefig(
            images_dir / output_filename_template.format(iteration=iteration),
            dpi=200,
        )
        plt.show()



def plot_final_sections(
    *,
    result: LaplaceResult,
    y_levels: tuple[float, ...],
    images_dir: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    exact_label_template: str,
    approx_label_template: str,
    output_filename: str,
) -> None:
    """Rysuje przekroje końcowego rozwiązania numerycznego."""

    plt.figure(figsize=(11, 7))

    for y_level in y_levels:
        j = int(np.argmin(np.abs(result.y - y_level)))

        (exact_line,) = plt.plot(
            result.x,
            result.exact_u[:, j],
            linewidth=2.0,
            label=exact_label_template.format(y_value=result.y[j]),
        )

        plt.plot(
            result.x,
            result.u[:, j],
            "o--",
            color=exact_line.get_color(),
            linewidth=1.2,
            markersize=3,
            markevery=2,
            label=approx_label_template.format(y_value=result.y[j]),
        )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(images_dir / output_filename, dpi=200)
    plt.show()



def plot_solution_3d(
    *,
    result: LaplaceResult,
    images_dir: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    zlabel: str,
    output_filename: str,
) -> None:
    """Rysuje wykres 3D końcowego rozwiązania numerycznego."""

    x_mesh, y_mesh = np.meshgrid(result.x, result.y, indexing="ij")

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    surface = ax.plot_surface(
        x_mesh,
        y_mesh,
        result.u,
        cmap="viridis",
        edgecolor="none",
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    fig.colorbar(surface, ax=ax, shrink=0.75, pad=0.1)

    plt.tight_layout()
    plt.savefig(images_dir / output_filename, dpi=200)
    plt.show()

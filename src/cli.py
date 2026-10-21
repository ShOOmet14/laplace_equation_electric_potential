from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from src.analysis import convergence_orders, run_grid_study
from src.models import LaplaceResult
from src.plotting import (
    plot_convergence,
    plot_cross_sections,
    plot_error_map,
    plot_solution_comparison,
)
from src.solver import solve_laplace
from src.symbolic import make_exact_solution, real_part_from_complex_function


DEFAULT_FUNCTION = "z**4"
DEFAULT_GRID_SIZE = 65
DEFAULT_GRID_STUDY = (9, 17, 33, 65)
DEFAULT_TOLERANCE = 1e-10
DEFAULT_MAX_ITERATIONS = 50_000
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Solve the 2D Laplace equation on the unit square and validate the "
            "five-point finite-difference solution against Re(f(z))."
        )
    )
    parser.add_argument(
        "--function",
        default=DEFAULT_FUNCTION,
        help="SymPy expression f(z); its real part is used (default: %(default)s)",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=DEFAULT_GRID_SIZE,
        help="points per coordinate direction for plots (default: %(default)s)",
    )
    parser.add_argument(
        "--grid-sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_GRID_STUDY),
        help="grid sizes for the refinement study (default: 9 17 33 65)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE,
        help="maximum-update stopping tolerance (default: %(default)g)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=DEFAULT_MAX_ITERATIONS,
        help="maximum Gauss-Seidel sweeps per grid (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "images",
        help="directory for generated figures (default: images)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="display figures interactively after saving them",
    )
    parser.add_argument(
        "--skip-grid-study",
        action="store_true",
        help="solve only the plotting grid and skip convergence analysis",
    )
    return parser


def _print_grid_study(results: Sequence[LaplaceResult]) -> None:
    orders = convergence_orders(results)
    print("\nGrid-refinement study")
    print("  points          h   iterations     L_inf error       RMS error    order")
    print("  ------  ---------  -----------  --------------  --------------  -------")
    for index, result in enumerate(results):
        observed_order = None if index == 0 else orders[index - 1]
        order = "-" if index == 0 else (
            "n/a" if observed_order is None else f"{observed_order:.4f}"
        )
        print(
            f"  {result.grid_size:6d}  {result.grid_spacing:9.6f}  "
            f"{result.iterations:11d}  {result.l_inf_error:14.6e}  "
            f"{result.rms_error:14.6e}  {order:>7}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)

    try:
        exact_expression = real_part_from_complex_function(arguments.function)
        exact_solution = make_exact_solution(exact_expression)
    except (TypeError, ValueError, SyntaxError) as error:
        parser.error(str(error))

    print(f"Exact harmonic solution: u(x, y) = {exact_expression}")

    study_results: list[LaplaceResult] = []
    if not arguments.skip_grid_study:
        try:
            study_results = run_grid_study(
                exact_solution,
                grid_sizes=arguments.grid_sizes,
                tolerance=arguments.tolerance,
                max_iterations=arguments.max_iterations,
            )
            _print_grid_study(study_results)
        except ValueError as error:
            parser.error(str(error))

        if not all(result.converged for result in study_results):
            print("At least one grid did not converge; increase --max-iterations.")
            return 2

    result = next(
        (
            study_result
            for study_result in study_results
            if study_result.grid_size == arguments.grid_size
        ),
        None,
    )
    if result is None:
        result = solve_laplace(
            exact_solution,
            grid_size=arguments.grid_size,
            tolerance=arguments.tolerance,
            max_iterations=arguments.max_iterations,
        )

    if not result.converged:
        print("Plotting-grid solve did not converge; increase --max-iterations.")
        return 2

    print(
        f"\nPlotting grid: {result.grid_size} x {result.grid_size}, "
        f"iterations={result.iterations}, L_inf error={result.l_inf_error:.6e}, "
        f"RMS error={result.rms_error:.6e}"
    )

    output_dir = arguments.output_dir.resolve()
    generated = [
        plot_solution_comparison(result, output_dir, show=arguments.show),
        plot_error_map(result, output_dir, show=arguments.show),
        plot_cross_sections(result, output_dir, show=arguments.show),
    ]
    if study_results and all(result.l_inf_error > 0.0 for result in study_results):
        generated.append(
            plot_convergence(study_results, output_dir, show=arguments.show)
        )
    elif study_results:
        print("Convergence plot omitted because at least one grid has zero error.")

    print("Generated figures:")
    for path in generated:
        print(f"  {path}")
    return 0

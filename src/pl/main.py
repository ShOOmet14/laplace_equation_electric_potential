from math import log

from src.common.analysis import run_grid_study
from src.common.config import (
    ITERATIONS_TO_SAVE,
    PLOT_GRID_SIZE,
    PLOT_MAX_ITERATIONS,
    PLOT_TOLERANCE,
)
from src.common.models import GridStudyRecordTypes, LaplaceResultTypes
from src.common.paths import get_images_dir
from src.common.plotting import (
    plot_3d_error,
    plot_error_map,
    plot_3d_exact_solution,
    plot_3d_solution,
    plot_cross_sections,
    plot_cross_sections_for_iterations,
    plot_final_cross_sections,
)
from src.common.laplace import solve_laplace
from src.pl.labels import LABELS


def print_grid_study_results(results: list[GridStudyRecordTypes]) -> None:
    """
    Print the grid study results in Polish.
    """
    print("\nBADANIE SIATKI I DOKŁADNOŚCI")
    print(f"{'Rozmiar':>8} | {'Krok siatki':>12} | {'Iteracje':>10} | {'Błąd max':>15}")
    print("-" * 57)

    for record in results:
        print(
            f"{record['grid_size']:>8} | "
            f"{record['grid_step']:>12.5f} | "
            f"{record['iterations']:>10} | "
            f"{record['max_error']:>15.8e}"
        )


def print_estimated_orders(results: list[GridStudyRecordTypes]) -> None:
    """
    Print the estimated order of accuracy in Polish.
    """
    if len(results) < 2:
        return

    print("\nPrzybliżony rząd dokładności p:")

    for index in range(len(results) - 1):
        grid_step_1 = results[index]["grid_step"]
        grid_step_2 = results[index + 1]["grid_step"]
        max_error_1 = results[index]["max_error"]
        max_error_2 = results[index + 1]["max_error"]

        order = log(max_error_1 / max_error_2) / log(grid_step_1 / grid_step_2)

        print(
            f"N = {results[index]['grid_size']:>3} -> "
            f"N = {results[index + 1]['grid_size']:>3}: p = {order:.4f}"
        )


def print_first_grid_below_threshold(
    results: list[GridStudyRecordTypes],
    error_threshold: float = 1e-4,
) -> None:
    """
    Print the first grid size for which the error is below the threshold.
    """
    first_good_result = next(
        (record for record in results if record["max_error"] < error_threshold),
        None,
    )

    if first_good_result is not None:
        print(
            f"\nDokładność lepsza od {error_threshold:.0e} osiągnięta dla: "
            f"Rozmiar siatki = {first_good_result['grid_size']}, "
            f"Błąd max = {first_good_result['max_error']:.8e}"
        )
    else:
        print(f"\nNie osiągnięto poziomu {error_threshold:.0e} dla zadanych siatek.")


def print_convergence_status(result: LaplaceResultTypes) -> None:
    """
    Print convergence information in Polish.
    """
    if result["converged"]:
        print(f"Zbieżność osiągnięta po {result['iterations']} iteracjach")
    else:
        print("Nie osiągnięto zbieżności w maksymalnej liczbie iteracji")


def print_plot_result_summary(result: LaplaceResultTypes) -> None:
    """
    Print summary information for the plotting result in Polish.
    """
    print("\nWYNIK DLA WYKRESÓW")
    print(f"Rozmiar siatki = {result['grid_size']}")
    print(f"Krok siatki = {result['grid_step']}")
    print(f"Liczba iteracji = {result['iterations']}")
    print(f"Błąd max = {result['max_error']:.8e}")


def main() -> None:
    """
    Run the Polish version of the program.
    """
    output_dir = get_images_dir("pl")

    grid_study_results = run_grid_study()
    print_grid_study_results(grid_study_results)
    print_estimated_orders(grid_study_results)
    print_first_grid_below_threshold(grid_study_results, error_threshold=1e-4)

    result = solve_laplace(
        grid_size=PLOT_GRID_SIZE,
        tolerance=PLOT_TOLERANCE,
        max_iterations=PLOT_MAX_ITERATIONS,
        iterations_to_save=ITERATIONS_TO_SAVE,
    )

    print_convergence_status(result)
    print_plot_result_summary(result)

    plot_cross_sections(
        x=result["x"],
        y=result["y"],
        solution_values=result["exact_solution"],
        output_dir=output_dir,
        labels=LABELS,
    )

    plot_cross_sections_for_iterations(
        result=result,
        output_dir=output_dir,
        labels=LABELS,
    )

    plot_final_cross_sections(
        result=result,
        output_dir=output_dir,
        labels=LABELS,
    )

    plot_3d_solution(
        result=result,
        output_dir=output_dir,
        labels=LABELS,
    )

    plot_3d_exact_solution(
        result=result,
        output_dir=output_dir,
        labels=LABELS,
    )

    plot_3d_error(
        result=result,
        output_dir=output_dir,
        labels=LABELS,
    )

    plot_error_map(
        result=result,
        output_dir=output_dir,
        labels=LABELS,
    )


if __name__ == "__main__":
    main()

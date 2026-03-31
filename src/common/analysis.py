from math import log

from src.common.config import GRID_SIZES, MAX_ITERATIONS, TOLERANCES
from src.common.i18n import Translator
from src.common.laplace import solve_laplace
from src.common.models import GridStudyRecordTypes


def run_grid_study(
    tr: Translator,
    error_threshold: float = 1e-4,
) -> list[GridStudyRecordTypes]:
    """
    Run computations for multiple grid sizes and print the grid-study summary.

    Returns:
    - A list of dictionaries containing results for successive grid sizes.
    """

    results: list[GridStudyRecordTypes] = []

    print(f"\n{tr.t('messages.grid_study_title')}")
    print(
        f"{tr.t('messages.grid_header_grid'):>8} | "
        f"{tr.t('messages.grid_header_step'):>12} | "
        f"{tr.t('messages.grid_header_iterations'):>10} | "
        f"{tr.t('messages.grid_header_max_error'):>15}"
    )
    print("-" * 57)

    for grid_size in GRID_SIZES:
        result = solve_laplace(
            grid_size=grid_size,
            tolerance=TOLERANCES[grid_size],
            max_iterations=MAX_ITERATIONS,
        )

        record: GridStudyRecordTypes = {
            "grid_size": result["grid_size"],
            "grid_step": result["grid_step"],
            "iterations": result["iterations"],
            "max_error": result["max_error"],
        }

        results.append(record)

        print(
            f"{record['grid_size']:>8} | "
            f"{record['grid_step']:>12.5f} | "
            f"{record['iterations']:>10} | "
            f"{record['max_error']:>15.8e}"
        )

    if len(results) >= 2:
        print(f"\n{tr.t('messages.estimated_order_title')}")

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

    first_good_result = next(
        (record for record in results if record["max_error"] < error_threshold),
        None,
    )

    if first_good_result is not None:
        print(
            tr.t(
                "messages.accuracy_reached",
                threshold=error_threshold,
                grid_size=first_good_result["grid_size"],
                max_error=first_good_result["max_error"],
            )
        )
    else:
        print(tr.t("messages.accuracy_not_reached", threshold=error_threshold))

    return results

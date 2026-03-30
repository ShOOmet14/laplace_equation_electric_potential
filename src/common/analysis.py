from src.common.config import GRID_SIZES, TOLERANCES, MAX_ITERATIONS
from src.common.models import GridStudyRecordTypes
from src.common.laplace import solve_laplace


def run_grid_study() -> list[GridStudyRecordTypes]:
    """
    Run computations for multiple grid sizes and store the results in a list.

    Returns:
    - A list of dictionaries containing results for successive grid sizes.
    """

    results: list[GridStudyRecordTypes] = []

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

    return results

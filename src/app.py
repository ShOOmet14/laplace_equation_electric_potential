from src.common.analysis import run_grid_study
from src.common.config import (
    ITERATIONS_TO_SAVE,
    LANGUAGE,
    PLOT_GRID_SIZE,
    PLOT_MAX_ITERATIONS,
    PLOT_TOLERANCE,
)
from src.common.i18n import Translator
from src.common.laplace import get_user_function, solve_laplace
from src.common.paths import get_images_dir
from src.common.plotting import (
    plot_3d_error,
    plot_3d_exact_solution,
    plot_3d_solution,
    plot_cross_sections,
    plot_cross_sections_for_iterations,
    plot_error_map,
    plot_final_cross_sections,
)


def main() -> None:
    tr = Translator(LANGUAGE)
    plot_labels = tr.section("plots")

    get_user_function(tr)
    output_dir = get_images_dir(LANGUAGE)

    run_grid_study(tr, error_threshold=1e-4)

    result = solve_laplace(
        grid_size=PLOT_GRID_SIZE,
        tolerance=PLOT_TOLERANCE,
        max_iterations=PLOT_MAX_ITERATIONS,
        iterations_to_save=ITERATIONS_TO_SAVE,
        tr=tr,
        print_summary=True,
    )

    plot_cross_sections(
        x=result["x"],
        y=result["y"],
        solution_values=result["exact_solution"],
        output_dir=output_dir,
        labels=plot_labels,
    )

    plot_cross_sections_for_iterations(
        result=result, output_dir=output_dir, labels=plot_labels
    )
    plot_final_cross_sections(result=result, output_dir=output_dir, labels=plot_labels)
    plot_3d_solution(result=result, output_dir=output_dir, labels=plot_labels)
    plot_3d_exact_solution(result=result, output_dir=output_dir, labels=plot_labels)
    plot_3d_error(result=result, output_dir=output_dir, labels=plot_labels)
    plot_error_map(result=result, output_dir=output_dir, labels=plot_labels)


if __name__ == "__main__":
    main()

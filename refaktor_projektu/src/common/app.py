import numpy as np

from src.common.config import (
    ERROR_THRESHOLD,
    GRID_SIZES,
    ITERATIONS_TO_SAVE,
    MAX_ITERATIONS,
    PLOT_GRID_SIZE,
    PLOT_MAX_ITERATIONS,
    PLOT_TOLERANCE,
    SIDE_LENGTH,
    TOLERANCES,
    Y_LEVELS,
    get_images_dir,
)
from src.common.plotting import (
    plot_final_sections,
    plot_sections,
    plot_sections_for_iterations,
    plot_solution_3d,
)
from src.common.solver import run_grid_study, solve_laplace


class LabelsProtocol:
    GRID_STUDY_TITLE: str
    GRID_TABLE_HEADER: str
    GRID_ROW_TEMPLATE: str
    ORDER_TITLE: str
    ORDER_ROW_TEMPLATE: str
    ACCURACY_OK_TEMPLATE: str
    ACCURACY_FAIL_MESSAGE: str
    CONVERGENCE_OK_TEMPLATE: str
    CONVERGENCE_FAIL_MESSAGE: str
    PLOT_RESULT_TITLE: str
    RESULT_GRID_SIZE_TEMPLATE: str
    RESULT_GRID_STEP_TEMPLATE: str
    RESULT_ITERATIONS_TEMPLATE: str
    RESULT_ERROR_TEMPLATE: str
    X_LABEL: str
    Y_LABEL: str
    Z_LABEL: str
    SECTION_LABEL_TEMPLATE: str
    EXACT_LABEL_TEMPLATE: str
    APPROX_LABEL_TEMPLATE: str
    SECTIONS_TITLE: str
    ITERATION_TITLE_TEMPLATE: str
    FINAL_TITLE: str
    SOLUTION_3D_TITLE: str
    SECTIONS_FILENAME: str
    ITERATION_FILENAME_TEMPLATE: str
    FINAL_FILENAME: str
    SOLUTION_3D_FILENAME: str



def run_language_app(*, language_code: str, labels: LabelsProtocol) -> None:
    images_dir = get_images_dir(language_code)

    study_results = run_grid_study(
        grid_sizes=GRID_SIZES,
        tolerances=TOLERANCES,
        max_iterations=MAX_ITERATIONS,
        side_length=SIDE_LENGTH,
    )

    print(f"\n{labels.GRID_STUDY_TITLE}")
    print(labels.GRID_TABLE_HEADER)
    for record in study_results:
        print(
            labels.GRID_ROW_TEMPLATE.format(
                grid_size=record.grid_size,
                grid_step=record.grid_step,
                iterations=record.iterations,
                max_error=record.max_error,
            )
        )

    print(f"\n{labels.ORDER_TITLE}")
    for left_record, right_record in zip(study_results, study_results[1:]):
        order = np.log(left_record.max_error / right_record.max_error) / np.log(
            left_record.grid_step / right_record.grid_step
        )
        print(
            labels.ORDER_ROW_TEMPLATE.format(
                left_grid=left_record.grid_size,
                right_grid=right_record.grid_size,
                order=order,
            )
        )

    first_good = next(
        (record for record in study_results if record.max_error < ERROR_THRESHOLD),
        None,
    )
    if first_good is None:
        print(f"\n{labels.ACCURACY_FAIL_MESSAGE}")
    else:
        print(
            "\n"
            + labels.ACCURACY_OK_TEMPLATE.format(
                grid_size=first_good.grid_size,
                max_error=first_good.max_error,
            )
        )

    plot_result = solve_laplace(
        grid_size=PLOT_GRID_SIZE,
        tolerance=PLOT_TOLERANCE,
        max_iterations=PLOT_MAX_ITERATIONS,
        side_length=SIDE_LENGTH,
        iterations_to_save=ITERATIONS_TO_SAVE,
    )

    if plot_result.converged:
        print(
            labels.CONVERGENCE_OK_TEMPLATE.format(
                iterations=plot_result.iterations,
            )
        )
    else:
        print(labels.CONVERGENCE_FAIL_MESSAGE)

    print(f"\n{labels.PLOT_RESULT_TITLE}")
    print(labels.RESULT_GRID_SIZE_TEMPLATE.format(grid_size=plot_result.grid_size))
    print(labels.RESULT_GRID_STEP_TEMPLATE.format(grid_step=plot_result.grid_step))
    print(labels.RESULT_ITERATIONS_TEMPLATE.format(iterations=plot_result.iterations))
    print(labels.RESULT_ERROR_TEMPLATE.format(max_error=plot_result.max_error))

    plot_sections(
        x=plot_result.x,
        y=plot_result.y,
        grid_values=plot_result.exact_u,
        y_levels=Y_LEVELS,
        images_dir=images_dir,
        title=labels.SECTIONS_TITLE,
        xlabel=labels.X_LABEL,
        ylabel=labels.Y_LABEL,
        legend_label_template=labels.SECTION_LABEL_TEMPLATE,
        output_filename=labels.SECTIONS_FILENAME,
    )
    plot_sections_for_iterations(
        result=plot_result,
        y_levels=Y_LEVELS,
        iterations_to_plot=ITERATIONS_TO_SAVE,
        images_dir=images_dir,
        title_template=labels.ITERATION_TITLE_TEMPLATE,
        xlabel=labels.X_LABEL,
        ylabel=labels.Y_LABEL,
        exact_label_template=labels.EXACT_LABEL_TEMPLATE,
        approx_label_template=labels.APPROX_LABEL_TEMPLATE,
        output_filename_template=labels.ITERATION_FILENAME_TEMPLATE,
    )
    plot_final_sections(
        result=plot_result,
        y_levels=Y_LEVELS,
        images_dir=images_dir,
        title=labels.FINAL_TITLE,
        xlabel=labels.X_LABEL,
        ylabel=labels.Y_LABEL,
        exact_label_template=labels.EXACT_LABEL_TEMPLATE,
        approx_label_template=labels.APPROX_LABEL_TEMPLATE,
        output_filename=labels.FINAL_FILENAME,
    )
    plot_solution_3d(
        result=plot_result,
        images_dir=images_dir,
        title=labels.SOLUTION_3D_TITLE,
        xlabel=labels.X_LABEL,
        ylabel=labels.Y_LABEL,
        zlabel=labels.Z_LABEL,
        output_filename=labels.SOLUTION_3D_FILENAME,
    )

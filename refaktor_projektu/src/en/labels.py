GRID_STUDY_TITLE = "GRID AND ACCURACY STUDY"
GRID_TABLE_HEADER = f"{'Size':>6} {'Grid step':>12} {'Iterations':>12} {'Max error':>15}"
GRID_ROW_TEMPLATE = "{grid_size:>6}{grid_step:>12.5f}{iterations:>12}{max_error:>15.8e}"

ORDER_TITLE = "Estimated order of accuracy p:"
ORDER_ROW_TEMPLATE = "N = {left_grid:>3} -> N = {right_grid:>3}: p = {order:.4f}"

ACCURACY_OK_TEMPLATE = (
    "Accuracy better than 1e-4 reached for: "
    "Grid size = {grid_size}, Max error = {max_error:.8e}"
)
ACCURACY_FAIL_MESSAGE = "The 1e-4 threshold was not reached for the tested grids."

CONVERGENCE_OK_TEMPLATE = "Convergence reached after {iterations} iterations"
CONVERGENCE_FAIL_MESSAGE = "Convergence was not reached within the maximum number of iterations"

PLOT_RESULT_TITLE = "RESULT USED FOR PLOTS"
RESULT_GRID_SIZE_TEMPLATE = "Grid size = {grid_size}"
RESULT_GRID_STEP_TEMPLATE = "Grid step = {grid_step}"
RESULT_ITERATIONS_TEMPLATE = "Number of iterations = {iterations}"
RESULT_ERROR_TEMPLATE = "Max error = {max_error:.8e}"

X_LABEL = "x"
Y_LABEL = "u(x, y)"
Z_LABEL = "u(x, y)"

SECTION_LABEL_TEMPLATE = "y = {y_value:.2f}"
EXACT_LABEL_TEMPLATE = "exact, y = {y_value:.2f}"
APPROX_LABEL_TEMPLATE = "relaxation, y = {y_value:.2f}"

SECTIONS_TITLE = "Function sections at selected y levels"
ITERATION_TITLE_TEMPLATE = "Solution sections after {iteration} iterations"
FINAL_TITLE = "Solution sections after convergence"
SOLUTION_3D_TITLE = "Numerical solution of Laplace equation - 3D plot"

SECTIONS_FILENAME = "sections.png"
ITERATION_FILENAME_TEMPLATE = "sections_iteration_{iteration}.png"
FINAL_FILENAME = "sections_final.png"
SOLUTION_3D_FILENAME = "solution_3d.png"

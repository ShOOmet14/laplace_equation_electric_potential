GRID_STUDY_TITLE = "BADANIE SIATKI I DOKŁADNOŚCI"
GRID_TABLE_HEADER = f"{'Rozmiar':>6} {'Krok siatki':>12} {'Iteracje':>10} {'Błąd max':>15}"
GRID_ROW_TEMPLATE = "{grid_size:>6}{grid_step:>12.5f}{iterations:>12}{max_error:>15.8e}"

ORDER_TITLE = "Przybliżony rząd dokładności p:"
ORDER_ROW_TEMPLATE = "N = {left_grid:>3} -> N = {right_grid:>3}: p = {order:.4f}"

ACCURACY_OK_TEMPLATE = (
    "Dokładność lepsza od 1e-4 osiągnięta dla: "
    "Rozmiar siatki = {grid_size}, Błąd max = {max_error:.8e}"
)
ACCURACY_FAIL_MESSAGE = "Nie osiągnięto poziomu 1e-4 dla zadanych siatek."

CONVERGENCE_OK_TEMPLATE = "Zbieżność osiągnięta po {iterations} iteracjach"
CONVERGENCE_FAIL_MESSAGE = "Nie osiągnięto zbieżności w maksymalnej liczbie iteracji"

PLOT_RESULT_TITLE = "WYNIK DLA WYKRESÓW"
RESULT_GRID_SIZE_TEMPLATE = "Rozmiar siatki = {grid_size}"
RESULT_GRID_STEP_TEMPLATE = "Krok siatki = {grid_step}"
RESULT_ITERATIONS_TEMPLATE = "Liczba iteracji = {iterations}"
RESULT_ERROR_TEMPLATE = "Błąd max = {max_error:.8e}"

X_LABEL = "x"
Y_LABEL = "u(x, y)"
Z_LABEL = "u(x, y)"

SECTION_LABEL_TEMPLATE = "y = {y_value:.2f}"
EXACT_LABEL_TEMPLATE = "dokładne, y = {y_value:.2f}"
APPROX_LABEL_TEMPLATE = "relaksacja, y = {y_value:.2f}"

SECTIONS_TITLE = "Przekroje funkcji na wybranych poziomach y"
ITERATION_TITLE_TEMPLATE = "Przekroje rozwiązania po {iteration} iteracjach"
FINAL_TITLE = "Przekroje rozwiązania po zbieżności"
SOLUTION_3D_TITLE = "Rozwiązanie numeryczne równania Laplace'a - wykres 3D"

SECTIONS_FILENAME = "przekroje.png"
ITERATION_FILENAME_TEMPLATE = "przekroje_iteracja_{iteration}.png"
FINAL_FILENAME = "przekroje_iteracja_koniec.png"
SOLUTION_3D_FILENAME = "rozwiazanie_3d.png"

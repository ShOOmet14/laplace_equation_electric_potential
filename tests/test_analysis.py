import pytest

from src.analysis import (
    convergence_orders,
    estimate_convergence_order,
    run_grid_study,
)
from src.solver import solve_laplace
from src.symbolic import make_exact_solution, real_part_from_complex_function


def test_order_formula_uses_actual_grid_spacing() -> None:
    order = estimate_convergence_order(
        coarse_error=0.04,
        fine_error=0.01,
        coarse_spacing=0.2,
        fine_spacing=0.1,
    )

    assert order == pytest.approx(2.0)


def test_quartic_validation_shows_second_order_grid_convergence() -> None:
    exact_solution = make_exact_solution(real_part_from_complex_function("z**4"))
    results = run_grid_study(
        exact_solution,
        grid_sizes=(9, 17, 33),
        tolerance=1e-10,
        max_iterations=10_000,
    )

    errors = [result.l_inf_error for result in results]
    orders = convergence_orders(results)

    assert all(result.converged for result in results)
    assert errors[0] > errors[1] > errors[2]
    assert orders == pytest.approx([2.0, 2.0], abs=0.03)


def test_zero_error_has_no_defined_convergence_order() -> None:
    zero_solution = make_exact_solution(real_part_from_complex_function("0"))
    results = [
        solve_laplace(
            zero_solution, grid_size=size, tolerance=1e-10, max_iterations=10
        )
        for size in (5, 9, 17)
    ]

    assert convergence_orders(results) == [None, None]


@pytest.mark.parametrize(
    "values",
    [
        (0.0, 0.1, 0.2, 0.1),
        (0.1, 0.0, 0.2, 0.1),
        (0.1, 0.05, 0.1, 0.2),
    ],
)
def test_invalid_convergence_data_is_rejected(values) -> None:
    with pytest.raises(ValueError):
        estimate_convergence_order(*values)

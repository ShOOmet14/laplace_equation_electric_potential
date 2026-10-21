import numpy as np
import pytest

from src.solver import solve_laplace
from src.symbolic import make_exact_solution, real_part_from_complex_function


@pytest.fixture
def quartic_solution():
    expression = real_part_from_complex_function("z**4")
    return make_exact_solution(expression)


def test_exact_values_are_applied_on_every_boundary(quartic_solution) -> None:
    result = solve_laplace(
        quartic_solution, grid_size=11, tolerance=1e-10, max_iterations=5_000
    )

    assert np.array_equal(result.numerical_solution[0, :], result.exact_solution[0, :])
    assert np.array_equal(result.numerical_solution[-1, :], result.exact_solution[-1, :])
    assert np.array_equal(result.numerical_solution[:, 0], result.exact_solution[:, 0])
    assert np.array_equal(result.numerical_solution[:, -1], result.exact_solution[:, -1])


def test_solver_converges_to_accurate_quartic_solution(quartic_solution) -> None:
    result = solve_laplace(
        quartic_solution, grid_size=17, tolerance=1e-10, max_iterations=10_000
    )

    assert result.converged
    assert result.max_update <= 1e-10
    assert result.l_inf_error < 1.2e-3
    assert result.rms_error < result.l_inf_error


def test_saved_snapshot_is_an_independent_copy(quartic_solution) -> None:
    result = solve_laplace(
        quartic_solution,
        grid_size=9,
        tolerance=1e-10,
        max_iterations=5_000,
        snapshot_iterations=(2,),
    )

    snapshot = result.snapshots[2]
    assert not np.shares_memory(snapshot, result.numerical_solution)
    assert not np.array_equal(snapshot, result.numerical_solution)


@pytest.mark.parametrize(
    ("keyword", "value"),
    [("grid_size", 2), ("tolerance", 0.0), ("max_iterations", 0)],
)
def test_invalid_solver_inputs_are_rejected(quartic_solution, keyword, value) -> None:
    arguments = {"grid_size": 9, "tolerance": 1e-8, "max_iterations": 100}
    arguments[keyword] = value

    with pytest.raises(ValueError):
        solve_laplace(quartic_solution, **arguments)


def test_non_finite_exact_solution_is_rejected() -> None:
    def singular_solution(x, y):
        return 1.0 / (x - x)

    with np.errstate(divide="ignore", invalid="ignore"):
        with pytest.raises(ValueError, match="not finite"):
            solve_laplace(
                singular_solution,
                grid_size=5,
                tolerance=1e-8,
                max_iterations=100,
            )

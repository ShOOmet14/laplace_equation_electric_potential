import numpy as np
import pytest
import sympy

from src.symbolic import (
    X,
    Y,
    evaluate_exact_solution,
    is_harmonic,
    make_exact_solution,
    real_part_from_complex_function,
)


def test_real_part_of_z_to_four_is_harmonic() -> None:
    expression = real_part_from_complex_function("z**4")

    assert sympy.simplify(expression - (X**4 - 6 * X**2 * Y**2 + Y**4)) == 0
    assert is_harmonic(expression)


def test_non_harmonic_real_part_is_rejected() -> None:
    with pytest.raises(ValueError, match="not harmonic"):
        real_part_from_complex_function("z * conjugate(z)")


def test_direct_non_harmonic_expression_is_detected() -> None:
    assert not is_harmonic(X**2 + Y**2)


def test_only_z_is_accepted_in_complex_function() -> None:
    with pytest.raises(ValueError, match="only contain z"):
        real_part_from_complex_function("z + a")


def test_constant_solution_is_broadcast_to_grid_shape() -> None:
    exact_solution = make_exact_solution(sympy.Integer(3))
    x, y = np.meshgrid(np.linspace(0, 1, 4), np.linspace(0, 1, 4), indexing="ij")

    values = evaluate_exact_solution(exact_solution, x, y)

    assert values.shape == (4, 4)
    assert np.all(values == 3.0)

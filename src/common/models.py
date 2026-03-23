from typing import TypedDict
import numpy.typing as npt
import numpy as np


class LaplaceResultTypes(TypedDict):
    x: npt.NDArray[np.float64]
    y: npt.NDArray[np.float64]
    numerical_solution: npt.NDArray[np.float64]
    grid_step: float
    iterations: int
    converged: bool
    iteration_snapshots: dict[int, npt.NDArray[np.float64]]
    exact_solution: npt.NDArray[np.float64]
    error: npt.NDArray[np.float64]
    max_error: float
    grid_size: int


class GridStudyRecordTypes(TypedDict):
    grid_size: int
    grid_step: float
    iterations: int
    max_error: float

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(slots=True)
class LaplaceResult:
    x: npt.NDArray[np.float64]
    y: npt.NDArray[np.float64]
    u: npt.NDArray[np.float64]
    grid_step: float
    iterations: int
    snapshots: dict[int, npt.NDArray[np.float64]]
    exact_u: npt.NDArray[np.float64]
    error: npt.NDArray[np.float64]
    max_error: float
    grid_size: int
    converged: bool


@dataclass(slots=True)
class GridStudyRecord:
    grid_size: int
    grid_step: float
    iterations: int
    max_error: float
    converged: bool

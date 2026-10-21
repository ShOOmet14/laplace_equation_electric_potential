from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


FloatArray = npt.NDArray[np.float64]


@dataclass(slots=True)
class LaplaceResult:
    """Numerical solution and validation data for one square grid."""

    x: FloatArray
    y: FloatArray
    numerical_solution: FloatArray
    exact_solution: FloatArray
    absolute_error: FloatArray
    grid_spacing: float
    iterations: int
    converged: bool
    max_update: float
    l_inf_error: float
    rms_error: float
    snapshots: dict[int, FloatArray]

    @property
    def grid_size(self) -> int:
        """Number of grid points along either coordinate direction."""

        return len(self.x)

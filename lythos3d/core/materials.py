"""Material models.

Units are kN, m and kPa throughout, as in the 2D program: stiffness and
strength in kPa, unit weight in kN/m3.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .elements import elastic_matrix


@dataclass(frozen=True)
class LinearElastic:
    """Isotropic linear elasticity."""

    name: str
    E: float
    nu: float
    gamma: float = 0.0

    def __post_init__(self):
        if self.E <= 0:
            raise ValueError(f"{self.name}: E must be positive")
        if not -1.0 < self.nu < 0.5:
            raise ValueError(f"{self.name}: nu must lie in (-1, 0.5)")

    def elastic(self) -> np.ndarray:
        return elastic_matrix(self.E, self.nu)

    @property
    def oedometer_modulus(self) -> float:
        """Constrained modulus ``M``, the stiffness under zero lateral strain."""
        return self.E * (1.0 - self.nu) / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))

    @property
    def k0(self) -> float:
        """Ratio of horizontal to vertical stress under zero lateral strain."""
        return self.nu / (1.0 - self.nu)

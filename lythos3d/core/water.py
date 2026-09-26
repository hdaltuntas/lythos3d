"""Groundwater: a phreatic surface and the hydrostatic pore pressure under it.

The analysis is drained and in effective stress, as in 2D Lythos.  The soil
skeleton carries ``sigma'``; the water carries ``p`` (compression positive),
and total stress is ``sigma = sigma' - p m`` with ``m = [1, 1, 1, 0, 0, 0]``.
Pore pressure is hydrostatic below the phreatic surface and zero above it
(no suction):

    p = gamma_w max(h(x, y) - z, 0)

where the head ``h`` is a constant level, or interpolated between water
levels measured in boreholes, and may be lowered inside drawdown polygons -
a pit pumped dry to its formation level.  There is no seepage analysis: a
level that differs from place to place is taken as it is given, and the
horizontal pressure gradient it implies is the seepage force the skeleton
feels.

Equilibrium in total stress, ``int B^T sigma dV = f``, becomes, for the
skeleton,

    int B^T sigma' dV = f + int B^T m p dV - int_free N^T p n dA

with the saturated unit weight in ``f`` below the water.  The volume term is
integrated element by element, so where the pressure jumps between
neighbouring elements - across a wall with the ground dewatered on one side
- the difference lands on the nodes of the face between them as a net
force: the water pushing on the wall.  The surface term is the water
standing on the free surfaces of the ground (a flooded pit, a lake); on the
fixed faces of the model box it goes into the reactions.  Together, below a
level water table, they reduce to the buoyant weight ``gamma_sat - gamma_w``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

#: unit weight of water, kN/m3 (as in 2D Lythos)
GAMMA_WATER = 9.81


def _inside(polygon, points: np.ndarray) -> np.ndarray:
    """Plan points inside a polygon (even-odd rule)."""
    poly = np.asarray(polygon, float)
    x, y = points[:, 0], points[:, 1]
    inside = np.zeros(len(points), bool)
    for (x1, y1), (x2, y2) in zip(poly, np.roll(poly, -1, axis=0)):
        crosses = (y1 > y) != (y2 > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            xc = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
        inside ^= crosses & (x < xc)
    return inside


@dataclass(frozen=True)
class Drawdown:
    """Inside ``polygon`` (plan) the water stands no higher than ``level``."""

    polygon: tuple
    level: float

    def __post_init__(self):
        object.__setattr__(self, "polygon", tuple(tuple(map(float, p)) for p in self.polygon))
        if len(self.polygon) < 3:
            raise ValueError("a drawdown polygon needs three corners")


@dataclass(frozen=True)
class WaterTable:
    """The phreatic surface.

    ``level`` is a constant level; ``wells`` - ``(x, y, level)`` points,
    water levels read in boreholes - give one interpolated over the plan
    instead (linearly between them, level beyond them, as the soil tops
    are).  ``drawdowns`` lower it locally.  ``level=None`` and no wells is
    dry ground.
    """

    level: float | None = None
    wells: tuple = ()
    drawdowns: tuple = ()
    gamma_w: float = GAMMA_WATER
    _interp: object = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        wells = tuple(tuple(map(float, w)) for w in self.wells)
        object.__setattr__(self, "wells", wells)
        object.__setattr__(self, "drawdowns", tuple(
            d if isinstance(d, Drawdown) else Drawdown(*d) for d in self.drawdowns))
        if self.level is not None and wells:
            raise ValueError("give the water table a level or wells, not both")
        if self.gamma_w <= 0:
            raise ValueError("the unit weight of water must be positive")
        if wells:
            from .site import _Interpolator

            object.__setattr__(self, "_interp", _Interpolator([w[:2] for w in wells]))

    @classmethod
    def dry(cls) -> "WaterTable":
        return cls()

    @property
    def is_dry(self) -> bool:
        return self.level is None and not self.wells

    def lowered(self, polygon, level: float) -> "WaterTable":
        """This table with the water inside ``polygon`` drawn down to ``level``."""
        return replace(self, drawdowns=self.drawdowns + (Drawdown(polygon, level),), _interp=None)

    def head(self, points: np.ndarray) -> np.ndarray:
        """Level of the phreatic surface above plan points (n, 2 or 3); -inf where dry."""
        points = np.atleast_2d(np.asarray(points, float))
        if self.is_dry:
            return np.full(len(points), -np.inf)
        if self.wells:
            h = self._interp(np.array([w[2] for w in self.wells]), points[:, :2])
        else:
            h = np.full(len(points), float(self.level))
        for d in self.drawdowns:
            inside = _inside(d.polygon, points)
            h[inside] = np.minimum(h[inside], d.level)
        return h

    def pressure(self, points: np.ndarray) -> np.ndarray:
        """Hydrostatic pore pressure (kPa, compression positive) at points (n, 3)."""
        points = np.atleast_2d(np.asarray(points, float))
        if self.is_dry:
            return np.zeros(len(points))
        return self.gamma_w * np.maximum(self.head(points) - points[:, 2], 0.0)


def layered_overburden(z: np.ndarray, tops: np.ndarray, bases: np.ndarray, gamma: np.ndarray,
                       gamma_sat: np.ndarray, level: np.ndarray | None, gamma_w: float) -> np.ndarray:
    """Total vertical stress at levels ``z`` under layers ``tops``/``bases`` (n, k).

    Soil above ``level`` weighs ``gamma``, below it ``gamma_sat``; water
    standing above the ground adds its own weight.  Compression positive.
    """
    lo = np.maximum(z[:, None], bases)
    thickness = np.clip(tops - lo, 0.0, None)
    if level is None:
        return thickness @ gamma
    wet = np.clip(np.minimum(tops, level[:, None]) - lo, 0.0, None)
    wet = np.minimum(wet, thickness)
    sv = (thickness - wet) @ gamma + wet @ gamma_sat
    ground = np.maximum(tops[:, 0], z)
    return sv + gamma_w * np.clip(level - ground, 0.0, None)

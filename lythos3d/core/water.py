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


@dataclass(frozen=True)
class Seepage:
    """Steady groundwater flow, solved over the ground rather than assumed hydrostatic.

    ``table`` gives the boundary conditions: its level is the head held on
    the open sides of the model box (where they lie below it) and the level
    of any water standing on the ground; its drawdowns are levels the water
    is pumped down to, and hold the head on the ground inside them.  Ground
    above the water is a seepage face where water flows out of it, and
    closed where it would flow in.  The base of the model is closed, and so
    are the sides named in ``closed`` - ``"xmin"``, ``"xmax"``, ``"ymin"``,
    ``"ymax"`` - such as planes of symmetry.  A wall with an interface is
    impermeable; without one, water passes through it.

    Above the phreatic surface the permeability falls log-linearly from its
    full value at zero pressure to ``k_min`` times it at a suction head of
    ``psi_k`` metres, which is how the surface is found.
    """

    table: WaterTable
    closed: tuple = ()
    psi_k: float = 0.7
    k_min: float = 1.0e-4

    def __post_init__(self):
        object.__setattr__(self, "closed", tuple(self.closed))
        unknown = set(self.closed) - {"xmin", "xmax", "ymin", "ymax"}
        if unknown:
            raise ValueError(f"seepage: unknown side(s) {sorted(unknown)}")
        if self.table.is_dry:
            raise ValueError("seepage needs a water table for its boundary conditions")
        if self.psi_k <= 0 or not 0 < self.k_min <= 1:
            raise ValueError("seepage: need psi_k > 0 and 0 < k_min <= 1")

    @property
    def is_dry(self) -> bool:
        return False

    @property
    def gamma_w(self) -> float:
        return self.table.gamma_w

    def lowered(self, polygon, level: float) -> "Seepage":
        """This flow with the water inside ``polygon`` pumped down to ``level``."""
        return replace(self, table=self.table.lowered(polygon, level))

    def head(self, points: np.ndarray) -> np.ndarray:
        """The water table's level: the hydrostatic guess the K0 procedure uses."""
        return self.table.head(points)


class PoreField:
    """The pore pressure of one stage, at the Gauss points and anywhere inside an element.

    ``gauss`` (n_points,) is zero outside the active ground.  ``at(points,
    owners)`` gives the pressure at points inside the elements ``owners``.
    ``head`` (n_nodes,) and the flow are set for a seepage solution.
    """

    is_dry = False
    head = None
    velocity = None
    flows: dict = {}

    def __init__(self, gauss: np.ndarray):
        self.gauss = gauss

    def at(self, points: np.ndarray, owners: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class DryField(PoreField):
    is_dry = True

    def __init__(self, n_points: int):
        super().__init__(np.zeros(n_points))

    def at(self, points, owners):
        return np.zeros(len(points))


class HydrostaticField(PoreField):
    def __init__(self, table: WaterTable, gauss_xyz: np.ndarray, active_points: np.ndarray):
        p = table.pressure(gauss_xyz)
        p[~active_points] = 0.0
        super().__init__(p)
        self.table = table

    def at(self, points, owners):
        return self.table.pressure(points)


class HeadField(PoreField):
    """Pore pressure from a head field on the quadratic tetrahedra: ``p = gamma_w max(h - z, 0)``."""

    def __init__(self, head: np.ndarray, nodes: np.ndarray, elements: np.ndarray, gauss_N: np.ndarray,
                 gauss_xyz: np.ndarray, active: np.ndarray, gamma_w: float):
        self.head, self.nodes, self.elements, self.gamma_w = head, nodes, elements, gamma_w
        ng = len(gauss_N)
        h = np.zeros((len(elements), ng))
        h[active] = np.nan_to_num(head[elements[active]]) @ gauss_N.T
        p = gamma_w * np.maximum(h.ravel() - gauss_xyz[:, 2], 0.0)
        p[~np.repeat(active, ng)] = 0.0
        super().__init__(p)

    def head_at(self, points: np.ndarray, owners: np.ndarray) -> np.ndarray:
        from .elements import TET10_EDGES

        P = self.nodes[self.elements[owners, :4]]
        lam = np.linalg.solve(np.transpose(P[:, 1:] - P[:, :1], (0, 2, 1)),
                              (points - P[:, 0])[:, :, None])[:, :, 0]
        L = np.column_stack([1.0 - lam.sum(axis=1), lam])
        N = np.column_stack([L * (2.0 * L - 1.0)] + [4.0 * L[:, i] * L[:, j] for i, j in TET10_EDGES])
        return np.einsum("na,na->n", N, np.nan_to_num(self.head[self.elements[owners]]))

    def at(self, points, owners):
        points = np.atleast_2d(np.asarray(points, float))
        return self.gamma_w * np.maximum(self.head_at(points, owners) - points[:, 2], 0.0)

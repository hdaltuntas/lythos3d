"""Ground described by boreholes, and excavations drawn in plan.

A borehole lists the soils it passes through and the level at which each one
starts.  Between boreholes the top of every soil is interpolated linearly
over a Delaunay triangulation of the borehole positions; outside their
convex hull it takes the value at the nearest point of the hull, which keeps
every surface continuous.  A soil a borehole does not meet has zero thickness
there, so layers pinch out between boreholes as they do in the ground.

The same :class:`SoilProfile` decides both the geometry that is meshed and
the overburden the K0 procedure uses, so the two agree.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Soil:
    """A soil unit and its material, identified by name in the boreholes."""

    name: str
    material: object


@dataclass
class Borehole:
    """A borehole at plan position (x, y).

    ``tops`` lists ``(soil name, level of its top)`` from the ground down; the
    first level is the ground surface there.  Soils may be missing, but those
    present must come in the order of :attr:`SoilProfile.soils`.
    """

    name: str
    x: float
    y: float
    tops: list[tuple[str, float]]

    @property
    def ground(self) -> float:
        return self.tops[0][1]


class _Interpolator:
    """Continuous piecewise-linear interpolation of values given at scattered plan points."""

    def __init__(self, xy: np.ndarray):
        self.xy = np.asarray(xy, dtype=float)
        n = len(self.xy)
        self.kind = "constant" if n == 1 else "line"
        if n >= 3:
            centred = self.xy - self.xy.mean(axis=0)
            _, s, vt = np.linalg.svd(centred, full_matrices=False)
            if s[-1] > 1e-9 * max(s[0], 1.0):
                from scipy.spatial import ConvexHull, Delaunay

                self.kind = "triangulated"
                self.tri = Delaunay(self.xy)
                hull = self.xy[ConvexHull(self.xy).vertices]
                self.hull_edges = np.stack([hull, np.roll(hull, -1, axis=0)], axis=1)
        if self.kind == "line":
            # collinear points: interpolate along the line, constant across it
            direction = self.xy[-1] - self.xy[0] if n == 2 else \
                np.linalg.svd(self.xy - self.xy.mean(axis=0))[2][0]
            self.direction = direction / np.linalg.norm(direction)
            self.origin = self.xy.mean(axis=0)
            self.t = (self.xy - self.origin) @ self.direction
            self.order = np.argsort(self.t)

    def __call__(self, values: np.ndarray, points: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        points = np.asarray(points, dtype=float)[:, :2]
        if self.kind == "constant":
            return np.full(len(points), values[0])
        if self.kind == "line":
            t = (points - self.origin) @ self.direction
            return np.interp(t, self.t[self.order], values[self.order])
        inside = self.tri.find_simplex(points) >= 0
        q = points.copy()
        if not inside.all():
            q[~inside] = self._nearest_on_hull(points[~inside])
        simplex = self.tri.find_simplex(q)
        # points projected onto the hull may land a hair outside it
        lost = simplex < 0
        if lost.any():
            centre = self.xy.mean(axis=0)
            q[lost] = q[lost] + 1e-9 * (centre - q[lost])
            simplex[lost] = self.tri.find_simplex(q[lost])
        T = self.tri.transform[simplex]
        b = np.einsum("nij,nj->ni", T[:, :2], q - T[:, 2])
        bary = np.column_stack([b, 1.0 - b.sum(axis=1)])
        return np.einsum("ni,ni->n", bary, values[self.tri.simplices[simplex]])

    def _nearest_on_hull(self, p: np.ndarray) -> np.ndarray:
        a, b = self.hull_edges[:, 0], self.hull_edges[:, 1]
        ab = b - a
        t = np.clip(np.einsum("pej,ej->pe", p[:, None, :] - a[None], ab) / np.einsum("ej,ej->e", ab, ab), 0, 1)
        foot = a[None] + t[..., None] * ab[None]
        d = np.linalg.norm(foot - p[:, None, :], axis=2)
        return foot[np.arange(len(p)), d.argmin(axis=1)]


@dataclass
class SoilProfile:
    """Soil units from the top down, the boreholes that locate them, and the model base."""

    soils: list[Soil]
    boreholes: list[Borehole]
    bottom: float

    def __post_init__(self):
        if not self.soils or not self.boreholes:
            raise ValueError("a soil profile needs at least one soil and one borehole")
        names = [s.name for s in self.soils]
        if len(set(names)) != len(names):
            raise ValueError("soil names must be unique")
        index = {n: i for i, n in enumerate(names)}
        n_soil = len(names)
        tops = np.empty((len(self.boreholes), n_soil))
        for b, hole in enumerate(self.boreholes):
            if not hole.tops:
                raise ValueError(f"borehole {hole.name!r} lists no soils")
            order = []
            for name, level in hole.tops:
                if name not in index:
                    raise ValueError(f"borehole {hole.name!r}: unknown soil {name!r}")
                order.append(index[name])
            if order != sorted(order) or len(set(order)) != len(order):
                raise ValueError(f"borehole {hole.name!r}: soils out of order")
            levels = [lvl for _, lvl in hole.tops]
            if any(a < b for a, b in zip(levels, levels[1:])):
                raise ValueError(f"borehole {hole.name!r}: levels must fall with depth")
            if levels[-1] <= self.bottom:
                raise ValueError(f"borehole {hole.name!r} reaches below the bottom of the model")
            # a soil the borehole does not meet has zero thickness: its top is
            # the top of the next soil down that is present (or the bottom)
            present = dict(zip(order, levels))
            below = self.bottom
            for k in range(n_soil - 1, -1, -1):
                below = present.get(k, below)
                tops[b, k] = below
            # soils above the first one met start at the ground
            tops[b, :order[0]] = levels[0]
        self._tops = tops
        self._interp = _Interpolator([(h.x, h.y) for h in self.boreholes])

    @property
    def n_soils(self) -> int:
        return len(self.soils)

    def tops(self, points: np.ndarray) -> np.ndarray:
        """Level of the top of every soil at plan points: (n, n_soils), non-increasing."""
        out = np.column_stack([self._interp(self._tops[:, k], points) for k in range(self.n_soils)])
        out = np.minimum.accumulate(out, axis=1)
        return np.maximum(out, self.bottom)

    def ground(self, points: np.ndarray) -> np.ndarray:
        return self.tops(points)[:, 0]

    def soil_of(self, points: np.ndarray) -> np.ndarray:
        """Index of the soil containing each point (the lowest soil whose top is above it)."""
        tops = self.tops(points)
        k = (points[:, 2][:, None] <= tops).sum(axis=1) - 1
        return np.clip(k, 0, self.n_soils - 1)

    def overburden(self, points: np.ndarray) -> np.ndarray:
        """Vertical stress (compression positive) from the weight of the soil above each point."""
        tops = self.tops(points)
        bases = np.column_stack([tops[:, 1:], np.full(len(points), self.bottom)])
        z = points[:, 2][:, None]
        thickness = np.clip(tops - np.maximum(z, bases), 0.0, None)
        gamma = np.array([s.material.gamma for s in self.soils])
        return thickness @ gamma


@dataclass
class Excavation:
    """A pit drawn in plan, dug in lifts.

    ``levels`` are the formation levels reached at each lift, falling; lift
    ``k`` removes the ground inside ``polygon`` between level ``k - 1`` (the
    ground surface for the first) and level ``k``.  Its element groups are
    named ``"<name> 1"``, ``"<name> 2"``...
    """

    name: str
    polygon: list[tuple[float, float]]
    levels: list[float]
    mesh_size: float | None = None

    def __post_init__(self):
        if len(self.polygon) < 3:
            raise ValueError(f"excavation {self.name!r}: a polygon needs three corners")
        if not self.levels or any(a <= b for a, b in zip(self.levels, self.levels[1:])):
            raise ValueError(f"excavation {self.name!r}: levels must fall lift by lift")
        area = _signed_area(np.asarray(self.polygon, float))
        if abs(area) < 1e-12:
            raise ValueError(f"excavation {self.name!r}: the polygon has no area")

    @property
    def lift_names(self) -> list[str]:
        return [f"{self.name} {k + 1}" for k in range(len(self.levels))]

    def contains(self, points: np.ndarray) -> np.ndarray:
        """Plan points inside the polygon (even-odd rule)."""
        poly = np.asarray(self.polygon, float)
        x, y = points[:, 0], points[:, 1]
        inside = np.zeros(len(points), bool)
        for (x1, y1), (x2, y2) in zip(poly, np.roll(poly, -1, axis=0)):
            crosses = (y1 > y) != (y2 > y)
            with np.errstate(divide="ignore", invalid="ignore"):
                xc = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            inside ^= crosses & (x < xc)
        return inside

    def lift_of(self, points: np.ndarray) -> np.ndarray:
        """Lift (0-based) each point falls in, or -1 outside the excavation."""
        out = np.full(len(points), -1)
        inside = self.contains(points)
        z = points[:, 2]
        for k, level in enumerate(self.levels):
            upper = np.inf if k == 0 else self.levels[k - 1]
            out[inside & (z > level) & (z <= upper) & (out < 0)] = k
        return out


def _signed_area(p: np.ndarray) -> float:
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


@dataclass
class SiteWall:
    """A wall drawn in plan: a polyline, from ``toe`` level up to ``top``.

    Without ``top`` the wall reaches the ground surface wherever it is.  A
    polyline whose last point repeats the first closes on itself, as a wall
    round a pit does.  ``section`` is a :class:`~lythos3d.core.structures.PlateSection`.
    """

    name: str
    path: list[tuple[float, float]]
    toe: float
    section: object
    top: float | None = None

    def __post_init__(self):
        if len(self.path) < 2:
            raise ValueError(f"wall {self.name!r} needs at least two points")
        if self.top is not None and self.top <= self.toe:
            raise ValueError(f"wall {self.name!r}: its top must be above its toe")
        p = np.asarray(self.path, float)
        if np.any(np.linalg.norm(np.diff(p, axis=0), axis=1) < 1e-9):
            raise ValueError(f"wall {self.name!r} repeats a point")


@dataclass
class SiteAnchor:
    """An anchor or strut from point ``a`` to point ``b`` (3D coordinates).

    With ``fixed_end`` the end at ``b`` is held fixed (a strut to a plane of
    symmetry); otherwise both ends are nodes of the mesh.  ``EA`` (kN) and
    ``prestress`` (kN) are per anchor.
    """

    name: str
    a: tuple[float, float, float]
    b: tuple[float, float, float]
    EA: float
    prestress: float = 0.0
    fixed_end: bool = False

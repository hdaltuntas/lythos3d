"""Layered ground in a box, with excavations: the model description.

A :class:`Model` is what an engineer specifies - strata from the top down,
the volumes of ground that come out at each step, and the construction
stages - and :meth:`Model.build` turns it into a meshed
:class:`~lythos3d.core.problem.Problem`.  The strata are horizontal for now;
when they come from boreholes, only :meth:`Model.overburden` and the region
assignment change.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .mesh import box_mesh, graded
from .problem import Problem, Stage


@dataclass
class Stratum:
    """A soil layer, from its ``top`` level down to the top of the next one."""

    name: str
    material: object
    top: float


@dataclass
class Volume:
    """An axis-aligned block of ground that a stage digs out."""

    name: str
    lo: tuple[float, float, float]
    hi: tuple[float, float, float]

    def contains(self, points: np.ndarray, tol: float = 1e-9) -> np.ndarray:
        lo, hi = np.asarray(self.lo, float), np.asarray(self.hi, float)
        return np.all((points >= lo - tol) & (points <= hi + tol), axis=1)


@dataclass
class Model:
    """Horizontal strata in a box ``x`` by ``y`` down to ``bottom``.

    The ground surface is the top of the first stratum.  Grid lines are put
    on every stratum boundary and every face of every volume, so each of them
    is made of whole elements.
    """

    name: str
    x: tuple[float, float]
    y: tuple[float, float]
    bottom: float
    strata: list[Stratum]
    volumes: list[Volume] = field(default_factory=list)
    stages: list[Stage] = field(default_factory=list)
    mesh_size: float = 2.0
    vertical_mesh_size: float | None = None

    def __post_init__(self):
        if not self.strata:
            raise ValueError("a model needs at least one stratum")
        tops = [s.top for s in self.strata]
        if any(a <= b for a, b in zip(tops, tops[1:])):
            raise ValueError("strata must be listed from the top down, each top below the last")
        if tops[-1] <= self.bottom:
            raise ValueError("the lowest stratum starts below the bottom of the model")
        names = [v.name for v in self.volumes]
        if len(set(names)) != len(names):
            raise ValueError("volume names must be unique")

    @property
    def surface(self) -> float:
        return self.strata[0].top

    def overburden(self, points: np.ndarray) -> np.ndarray:
        """Vertical stress (compression positive) from the weight of the strata above."""
        z = points[:, 2]
        sv = np.zeros(len(points))
        bounds = [s.top for s in self.strata] + [self.bottom]
        for stratum, top, base in zip(self.strata, bounds[:-1], bounds[1:]):
            thickness = np.clip(top - np.maximum(z, base), 0.0, top - base)
            sv += stratum.material.gamma * thickness
        return sv

    def stratum_of(self, points: np.ndarray) -> np.ndarray:
        """Index of the stratum containing each point."""
        tops = np.array([s.top for s in self.strata])
        # number of stratum tops strictly above the point, less one
        return np.clip((points[:, 2][:, None] < tops[None, :]).sum(axis=1) - 1, 0, len(tops) - 1)

    def build(self) -> Problem:
        (x0, x1), (y0, y1) = self.x, self.y
        z0, z1 = self.bottom, self.surface
        h, hz = self.mesh_size, self.vertical_mesh_size or self.mesh_size
        bx = [c for v in self.volumes for c in (v.lo[0], v.hi[0])]
        by = [c for v in self.volumes for c in (v.lo[1], v.hi[1])]
        bz = [s.top for s in self.strata] + [c for v in self.volumes for c in (v.lo[2], v.hi[2])]
        mesh = box_mesh(graded(x0, x1, h, bx), graded(y0, y1, h, by), graded(z0, z1, hz, bz))
        centroids = mesh.centroids()
        mesh.region = self.stratum_of(centroids)
        groups = {}
        for v in self.volumes:
            mask = v.contains(centroids)
            if not mask.any():
                raise ValueError(f"volume {v.name!r} contains no part of the model")
            groups[v.name] = mask
        return Problem(mesh, {i: s.material for i, s in enumerate(self.strata)},
                       groups=groups, vertical_stress=self.overburden)

    def run(self, verbose: bool = False, backend: str = "auto", tolerance: float = 1e-3):
        """Build, then analyse every stage: ``(problem, list of StageResult)``."""
        from .solver import Solver

        problem = self.build()
        solver = Solver(problem, tolerance=tolerance, backend=backend, verbose=verbose)
        return problem, solver.run(self.stages)

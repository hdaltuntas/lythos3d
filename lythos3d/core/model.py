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
class Wall:
    """A plate on the rectangle from ``lo`` to ``hi``, flat in one coordinate direction."""

    name: str
    lo: tuple[float, float, float]
    hi: tuple[float, float, float]
    section: object

    def __post_init__(self):
        flat = [i for i in range(3) if abs(self.hi[i] - self.lo[i]) < 1e-12]
        if len(flat) != 1:
            raise ValueError(f"wall {self.name!r} must be a rectangle flat in exactly one direction")


@dataclass
class Anchor:
    """A bar from point ``a`` to point ``b``; with ``fixed_end`` the end at ``b`` is held fixed.

    ``EA`` (kN) is per anchor and ``prestress`` (kN) its lock-off load.
    """

    name: str
    a: tuple[float, float, float]
    b: tuple[float, float, float]
    EA: float
    prestress: float = 0.0
    fixed_end: bool = False


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
    walls: list[Wall] = field(default_factory=list)
    anchors: list[Anchor] = field(default_factory=list)

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
        boxes = [(v.lo, v.hi) for v in self.volumes] + [(w.lo, w.hi) for w in self.walls]
        points = [a.a for a in self.anchors] + [a.b for a in self.anchors if not a.fixed_end]
        bx = [c for lo, hi in boxes for c in (lo[0], hi[0])] + [pt[0] for pt in points]
        by = [c for lo, hi in boxes for c in (lo[1], hi[1])] + [pt[1] for pt in points]
        bz = ([s.top for s in self.strata] + [c for lo, hi in boxes for c in (lo[2], hi[2])]
              + [pt[2] for pt in points])
        mesh = box_mesh(graded(x0, x1, h, bx), graded(y0, y1, h, by), graded(z0, z1, hz, bz))
        centroids = mesh.centroids()
        mesh.region = self.stratum_of(centroids)
        groups = {}
        for v in self.volumes:
            mask = v.contains(centroids)
            if not mask.any():
                raise ValueError(f"volume {v.name!r} contains no part of the model")
            groups[v.name] = mask
        from .structures import Bar, Plate

        plates = []
        for w in self.walls:
            faces = mesh.faces_in_box(w.lo, w.hi)
            if len(faces) == 0:
                raise ValueError(f"wall {w.name!r} lies outside the model")
            plates.append(Plate(w.name, faces, w.section))
        bars = []
        for a in self.anchors:
            na = mesh.nearest_node(a.a)
            if np.linalg.norm(mesh.nodes[na] - np.asarray(a.a)) > 1e-6:
                raise ValueError(f"anchor {a.name!r} starts outside the model")
            if a.fixed_end:
                bars.append(Bar(a.name, na, None, a.EA, a.prestress, fixed_point=tuple(a.b)))
            else:
                nb = mesh.nearest_node(a.b)
                if np.linalg.norm(mesh.nodes[nb] - np.asarray(a.b)) > 1e-6:
                    raise ValueError(f"anchor {a.name!r} ends outside the model")
                bars.append(Bar(a.name, na, nb, a.EA, a.prestress))
        return Problem(mesh, {i: s.material for i, s in enumerate(self.strata)},
                       groups=groups, vertical_stress=self.overburden, plates=plates, bars=bars)

    def run(self, verbose: bool = False, backend: str = "auto", tolerance: float = 1e-3):
        """Build, then analyse every stage: ``(problem, list of StageResult)``."""
        from .solver import Solver

        problem = self.build()
        solver = Solver(problem, tolerance=tolerance, backend=backend, verbose=verbose)
        return problem, solver.run(self.stages)


@dataclass
class Site:
    """Ground from boreholes, with excavations, walls and anchors drawn in plan: meshed by gmsh.

    The model spans ``x`` by ``y`` in plan, from the base of ``profile`` up
    to the ground surface the boreholes define.  Each lift of each
    excavation becomes an element group named ``"<excavation> <lift>"``.

    Without ``stages`` the sequence is: K0 initial stresses; every wall
    installed; every lift in the order given, each anchor stressed as soon as
    the dig has gone below its head; then the factor of safety.
    """

    name: str
    profile: object
    x: tuple[float, float]
    y: tuple[float, float]
    excavations: list = field(default_factory=list)
    stages: list[Stage] = field(default_factory=list)
    mesh_size: float = 2.0
    walls: list = field(default_factory=list)
    anchors: list = field(default_factory=list)

    def __post_init__(self):
        names = [n for e in self.excavations for n in e.lift_names]
        names += [w.name for w in self.walls] + [a.name for a in self.anchors]
        if len(set(names)) != len(names):
            raise ValueError("excavation lifts, walls and anchors need distinct names")
        if not self.stages:
            self.stages = self.default_stages()

    def default_stages(self) -> list[Stage]:
        stages = [Stage("initial stresses", kind="initial", initial_stress="k0")]
        if self.walls:
            stages.append(Stage("install walls", install=tuple(w.name for w in self.walls)))
        pending = list(self.anchors)
        for exc in self.excavations:
            for name, level in zip(exc.lift_names, exc.levels):
                stages.append(Stage(f"{exc.name}: dig to {level:g}", excavate=(name,)))
                ready = [a for a in pending if a.a[2] >= level]
                if ready:
                    stages.append(Stage(f"stress {', '.join(a.name for a in ready)}",
                                        install=tuple(a.name for a in ready)))
                    pending = [a for a in pending if a not in ready]
        if pending:
            stages.append(Stage("install " + ", ".join(a.name for a in pending),
                                install=tuple(a.name for a in pending)))
        stages.append(Stage("factor of safety", kind="ssr"))
        return stages

    def build(self, verbose: bool = False) -> Problem:
        import warnings

        from .gmsh_mesh import mesh_site
        from .structures import Bar, Plate

        points = [a.a for a in self.anchors] + [a.b for a in self.anchors if not a.fixed_end]
        mesh, groups, wall_faces, point_nodes = mesh_site(
            self.profile, self.x, self.y, self.excavations, mesh_size=self.mesh_size,
            verbose=verbose, walls=self.walls, points=points)
        worst = float(mesh.quality().min())
        if worst < 0.002:
            warnings.warn(f"the mesh has a nearly flat element (radius ratio {worst:.3g}); "
                          "usually a soil boundary crossing an excavation level at a very "
                          "shallow angle", RuntimeWarning, stacklevel=2)
        empty = [name for name, mask in groups.items() if not mask.any()]
        if empty:
            raise ValueError(f"lift(s) {empty} contain no ground: are they above the surface?")
        plates = [Plate(w.name, wall_faces[w.name], w.section) for w in self.walls]
        heads = iter(point_nodes[:len(self.anchors)])
        ends = iter(point_nodes[len(self.anchors):])
        bars = []
        for a in self.anchors:
            head = next(heads)
            if a.fixed_end:
                bars.append(Bar(a.name, head, None, a.EA, a.prestress, fixed_point=tuple(a.b)))
            else:
                bars.append(Bar(a.name, head, next(ends), a.EA, a.prestress))
        materials = {i: s.material for i, s in enumerate(self.profile.soils)}
        return Problem(mesh, materials, groups=groups, vertical_stress=self.profile.overburden,
                       plates=plates, bars=bars)

    def run(self, verbose: bool = False, backend: str = "auto", tolerance: float = 1e-3):
        """Mesh, then analyse every stage: ``(problem, list of StageResult)``."""
        from .solver import Solver

        problem = self.build()
        solver = Solver(problem, tolerance=tolerance, backend=backend, verbose=verbose)
        return problem, solver.run(self.stages)

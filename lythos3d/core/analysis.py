"""Linear elastic analysis of a meshed block.

This is the first building block of the staged, elasto-plastic analysis to
come: one load case, solved once, with gravity, surface tractions and the
usual box restraints.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .assembly import SparsityPattern, assemble_vector, default_backend, solve_constrained
from .elements import ContinuumElements, face_traction
from .mesh import Mesh

AXES = {"x": 0, "y": 1, "z": 2}


@dataclass
class SurfaceLoad:
    """A uniform traction (kPa) on the boundary faces lying in one plane.

    ``where`` optionally restricts the loaded area: a function of the face
    centroids (nf, 3) returning a boolean mask, e.g. a footing
    ``lambda c: (abs(c[:, 0]) < 1) & (abs(c[:, 1]) < 1)``.
    """

    axis: str | int
    value: float
    traction: tuple[float, float, float]
    where: object = None

    def faces(self, mesh: Mesh) -> np.ndarray:
        axis = AXES.get(self.axis, self.axis)
        faces = mesh.faces_on_plane(axis, self.value)
        if self.where is not None and len(faces):
            centroid = mesh.nodes[faces[:, :3]].mean(axis=1)
            faces = faces[np.asarray(self.where(centroid), dtype=bool)]
        return faces


def box_fixities(mesh: Mesh, tol: float = 1e-9) -> np.ndarray:
    """Standard restraints of a block of ground.

    The base is fixed in every direction; the sides are on rollers, free to
    settle but held against movement normal to themselves.
    """
    lo, hi = mesh.bounds
    x = mesh.nodes
    fixed = []
    for axis in range(2):
        on_side = (np.abs(x[:, axis] - lo[axis]) < tol) | (np.abs(x[:, axis] - hi[axis]) < tol)
        fixed.append(3 * np.nonzero(on_side)[0] + axis)
    base = np.nonzero(np.abs(x[:, 2] - lo[2]) < tol)[0]
    fixed.extend(3 * base + k for k in range(3))
    return np.unique(np.concatenate(fixed))


@dataclass
class LinearResult:
    """Displacements and stresses of a linear analysis."""

    mesh: Mesh
    displacement: np.ndarray            # (n, 3)
    strain: np.ndarray                  # (ne * 4, 6) at Gauss points
    stress: np.ndarray                  # (ne * 4, 6) at Gauss points, tension positive
    nodal_stress: np.ndarray            # (n, 6), smoothed
    info: dict = field(default_factory=dict)

    def max_displacement(self) -> float:
        return float(np.linalg.norm(self.displacement, axis=1).max())


def linear_static(mesh: Mesh, materials, gravity: bool = True, loads=(),
                  fixed: np.ndarray | None = None, backend: str = "auto") -> LinearResult:
    """Solve one linear elastic load case.

    ``materials`` maps each region id in ``mesh.region`` to a
    :class:`LinearElastic`; a list is indexed by region id.
    """
    materials = dict(enumerate(materials)) if isinstance(materials, (list, tuple)) else dict(materials)
    missing = set(np.unique(mesh.region).tolist()) - set(materials)
    if missing:
        raise ValueError(f"no material given for region(s) {sorted(missing)}")
    t0 = time.perf_counter()

    ce = ContinuumElements(mesh.nodes, mesh.elements)
    ids = sorted(materials)
    lookup = {r: i for i, r in enumerate(ids)}
    which = np.array([lookup[r] for r in mesh.region], dtype=np.int64)
    D_all = np.stack([materials[r].elastic() for r in ids])
    gamma_all = np.array([materials[r].gamma for r in ids])
    D = D_all[which]                                        # (ne, 6, 6)

    n_dof = 3 * mesh.n_nodes
    dofs = ce.dofs()
    K = SparsityPattern(mesh.elements, mesh.n_nodes).assemble(ce.stiffness(D))
    force = np.zeros(n_dof)
    if gravity:
        force += assemble_vector(n_dof, dofs, ce.body_force(gamma_all[which]))
    for load in loads:
        faces = load.faces(mesh)
        if len(faces) == 0:
            raise ValueError(f"surface load on {load.axis} = {load.value} found no boundary faces")
        fdofs = (3 * faces[:, :, None] + np.arange(3)).reshape(len(faces), 18)
        force += assemble_vector(n_dof, fdofs, face_traction(mesh.nodes, faces, load.traction))
    t_assemble = time.perf_counter() - t0

    if fixed is None:
        fixed = box_fixities(mesh)
    backend_used = default_backend() if backend == "auto" else backend
    t1 = time.perf_counter()
    u = solve_constrained(K, force, fixed, backend=backend, symmetric=True)
    t_solve = time.perf_counter() - t1

    strain = ce.strains(u)
    stress = np.einsum("egij,egj->egi", np.broadcast_to(D[:, None], (ce.n_elements, 4, 6, 6)),
                       strain.reshape(ce.n_elements, 4, 6)).reshape(-1, 6)
    return LinearResult(
        mesh=mesh,
        displacement=u.reshape(-1, 3),
        strain=strain,
        stress=stress,
        nodal_stress=ce.nodal_average(stress, mesh.n_nodes),
        info={
            "n_nodes": mesh.n_nodes,
            "n_elements": mesh.n_elements,
            "n_dof": n_dof,
            "n_free": n_dof - len(fixed),
            "backend": backend_used,
            "assemble_seconds": t_assemble,
            "solve_seconds": t_solve,
        },
    )

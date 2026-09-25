"""A meshed problem ready for the non-linear solver, and its construction stages.

:class:`Problem` is the solver's view of a model: a mesh, a material for each
region, restraints, and named groups of elements that stages switch on and
off.  It does not care how the mesh was made; :mod:`lythos3d.core.model`
builds one from strata and excavation volumes, and a test can build one by
hand.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .analysis import SurfaceLoad, box_fixities
from .assembly import SparsityPattern, assemble_vector
from .elements import ContinuumElements, face_traction
from .mesh import Mesh

INITIAL = "initial"
PLASTIC = "plastic"
SSR = "ssr"


@dataclass
class Stage:
    """One step of the construction sequence.

    ``excavate`` and ``construct`` name element groups removed or added at
    this stage; both are cumulative, so a group dug out stays out.  ``loads``
    are the surface loads acting during this stage - list a load again in a
    later stage to keep it on.

    ``kind`` is ``"initial"`` for the initial stresses (by ``initial_stress``:
    ``"k0"`` or ``"gravity"``), ``"plastic"`` for an ordinary construction
    step, and ``"ssr"`` for a factor of safety by strength reduction.
    """

    name: str
    kind: str = PLASTIC
    increments: int = 10
    excavate: tuple[str, ...] = ()
    construct: tuple[str, ...] = ()
    loads: tuple[SurfaceLoad, ...] = ()
    reset_displacements: bool = True
    initial_stress: str = "k0"
    srf_min: float = 0.8
    srf_max: float = 3.0

    def __post_init__(self):
        if self.kind not in (INITIAL, PLASTIC, SSR):
            raise ValueError(f"stage {self.name!r}: unknown kind {self.kind!r}")
        if self.initial_stress not in ("k0", "gravity"):
            raise ValueError(f"stage {self.name!r}: initial_stress must be 'k0' or 'gravity'")
        self.excavate = tuple(self.excavate)
        self.construct = tuple(self.construct)
        self.loads = tuple(self.loads)


@dataclass
class Problem:
    """Mesh, materials, restraints and element groups.

    ``materials`` maps each region id in ``mesh.region`` to a material (a list
    is indexed by region id).  ``groups`` maps a name to a boolean mask over
    the elements; ``absent`` names the groups that do not exist until a stage
    constructs them.  ``vertical_stress`` is a function of points (n, 3)
    returning the effective overburden pressure there (compression positive)
    for the K0 procedure; without it only gravity loading can set up the
    initial stresses.
    """

    mesh: Mesh
    materials: dict
    fixed: np.ndarray | None = None
    groups: dict[str, np.ndarray] = field(default_factory=dict)
    absent: tuple[str, ...] = ()
    vertical_stress: object = None

    def __post_init__(self):
        if isinstance(self.materials, (list, tuple)):
            self.materials = dict(enumerate(self.materials))
        missing = set(np.unique(self.mesh.region).tolist()) - set(self.materials)
        if missing:
            raise ValueError(f"no material given for region(s) {sorted(missing)}")
        for name, mask in self.groups.items():
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != (self.mesh.n_elements,):
                raise ValueError(f"group {name!r} must be a mask over the {self.mesh.n_elements} elements")
            self.groups[name] = mask
        unknown = set(self.absent) - set(self.groups)
        if unknown:
            raise ValueError(f"absent group(s) {sorted(unknown)} are not defined")
        if self.fixed is None:
            self.fixed = box_fixities(self.mesh)
        self.fixed = np.unique(np.asarray(self.fixed, dtype=np.int64))
        self.continuum = ContinuumElements(self.mesh.nodes, self.mesh.elements)
        self.pattern = SparsityPattern(self.mesh.elements, self.mesh.n_nodes)
        self.dofs = self.continuum.dofs()
        self.n_dof = 3 * self.mesh.n_nodes
        ngp = self.continuum.n_gauss
        self._gauss_of_region = {
            r: (np.nonzero(self.mesh.region == r)[0][:, None] * ngp + np.arange(ngp)).ravel()
            for r in self.materials
        }

    @property
    def n_points(self) -> int:
        return self.continuum.n_points

    def material_groups(self, materials: dict | None = None):
        """(material, Gauss point indices) for each region present in the mesh."""
        materials = self.materials if materials is None else materials
        for r, gp in self._gauss_of_region.items():
            if len(gp):
                yield materials[r], gp

    def check_stage_groups(self, stage: Stage) -> None:
        for name in (*stage.excavate, *stage.construct):
            if name not in self.groups:
                raise ValueError(f"stage {stage.name!r} refers to unknown group {name!r}")

    # ------------------------------------------------------------------ loads
    def gravity(self, active: np.ndarray) -> np.ndarray:
        """Nodal self weight of the active elements."""
        gamma = np.zeros(self.mesh.n_elements)
        for r, mat in self.materials.items():
            gamma[self.mesh.region == r] = mat.gamma
        gamma[~active] = 0.0
        return assemble_vector(self.n_dof, self.dofs, self.continuum.body_force(gamma))

    def surface_loads(self, loads) -> np.ndarray:
        f = np.zeros(self.n_dof)
        for load in loads:
            faces = load.faces(self.mesh)
            if len(faces) == 0:
                raise ValueError(f"surface load on {load.axis} = {load.value} found no boundary faces")
            fdofs = (3 * faces[:, :, None] + np.arange(3)).reshape(len(faces), 18)
            f += assemble_vector(self.n_dof, fdofs, face_traction(self.mesh.nodes, faces, load.traction))
        return f

    # ----------------------------------------------------------- initial state
    def k0_stress(self, active: np.ndarray) -> np.ndarray:
        """Geostatic stress at the Gauss points: ``sv`` from the overburden, ``sh = K0 sv``."""
        if self.vertical_stress is None:
            raise ValueError("the K0 procedure needs the overburden (a stratum profile); "
                             "use initial_stress='gravity' for this problem")
        points = self.continuum.gauss_xyz.reshape(-1, 3)
        sv = np.asarray(self.vertical_stress(points), dtype=float)
        k0 = np.zeros(self.n_points)
        for mat, gp in self.material_groups():
            k0[gp] = mat.k0
        stress = np.zeros((self.n_points, 6))
        stress[:, 2] = -sv
        stress[:, 0] = stress[:, 1] = -k0 * sv
        stress[~np.repeat(active, self.continuum.n_gauss)] = 0.0
        return stress

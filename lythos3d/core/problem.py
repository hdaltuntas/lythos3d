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
from .assembly import CombinedPattern, SparsityPattern, assemble_vector
from .elements import ContinuumElements, face_traction
from .mesh import Mesh
from .structures import Bar, Plate, PlateElements

INITIAL = "initial"
PLASTIC = "plastic"
SSR = "ssr"


@dataclass
class Stage:
    """One step of the construction sequence.

    ``excavate`` and ``construct`` name element groups removed or added at
    this stage; both are cumulative, so a group dug out stays out.
    ``install`` names plates and bars built at this stage: a plate starts
    free of force in the ground as it has deformed so far, and a bar is
    stressed to its lock-off load.  ``loads`` are the surface loads acting
    during this stage - list a load again in a later stage to keep it on.

    ``kind`` is ``"initial"`` for the initial stresses (by ``initial_stress``:
    ``"k0"`` or ``"gravity"``), ``"plastic"`` for an ordinary construction
    step, and ``"ssr"`` for a factor of safety by strength reduction.
    """

    name: str
    kind: str = PLASTIC
    increments: int = 10
    excavate: tuple[str, ...] = ()
    construct: tuple[str, ...] = ()
    install: tuple[str, ...] = ()
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
        self.install = tuple(self.install)
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

    ``plates`` and ``bars`` are the structures; they do nothing until a stage
    installs them.
    """

    mesh: Mesh
    materials: dict
    fixed: np.ndarray | None = None
    groups: dict[str, np.ndarray] = field(default_factory=dict)
    absent: tuple[str, ...] = ()
    vertical_stress: object = None
    plates: list[Plate] = field(default_factory=list)
    bars: list[Bar] = field(default_factory=list)

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
        self._split_for_interfaces()
        if self.fixed is None:
            self.fixed = box_fixities(self.mesh)
        self.fixed = np.unique(np.asarray(self.fixed, dtype=np.int64))
        self.continuum = ContinuumElements(self.mesh.nodes, self.mesh.elements)
        self.pattern = SparsityPattern(self.mesh.elements, self.mesh.n_nodes)
        self.dofs = self.continuum.dofs()
        self.n_translation = 3 * self.mesh.n_nodes
        self._setup_structures()
        ngp = self.continuum.n_gauss
        self._gauss_of_region = {
            r: (np.nonzero(self.mesh.region == r)[0][:, None] * ngp + np.arange(ngp)).ravel()
            for r in self.materials
        }

    def _split_for_interfaces(self) -> None:
        """Give every plate with an interface its own nodes, and the soil behind it its own.

        The tetrahedra touching the plate are sorted into its two sides.  The
        soil on the + side keeps the original nodes; the plate and the soil on
        the - side get new ones at the same places, and an interface element
        joins each side to the plate.  Where the soil runs on past an edge of
        the plate - below the toe of a wall that stops short of the base, past
        a wall's end - splitting would open a crack along the plate's plane,
        so a node is left shared wherever it would split a face that is not
        the plate's own.
        """
        from .interfaces import InterfaceElements

        self.interface_elements = None
        self.interface_plate = np.zeros(0, dtype=np.int64)
        self.interface_support = np.zeros(0, dtype=np.int64)
        split = [i for i, p in enumerate(self.plates) if p.interface is not None]
        if not split:
            return
        seen = set()
        for i in split:
            nodes = set(np.unique(self.plates[i].faces).tolist())
            if nodes & seen:
                raise ValueError("plates with interfaces may not share nodes: model a wall "
                                 "round a corner as one plate")
            seen |= nodes
            if self.plates[i].side is None:
                raise ValueError(f"plate {self.plates[i].name!r} has an interface but no side function")

        mesh = self.mesh
        nodes = mesh.nodes.copy()
        elements = mesh.elements.copy()
        local_faces = np.array([(0, 1, 2, 4, 5, 6), (0, 1, 3, 4, 8, 7), (1, 2, 3, 5, 9, 8), (0, 2, 3, 6, 9, 7)])
        soil_faces, wall_faces, normals, plate_of, support = [], [], [], [], []
        for i in split:
            plate = self.plates[i]
            faces = np.asarray(plate.faces, dtype=np.int64)
            on_wall = np.zeros(len(nodes), bool)
            on_wall[faces.ravel()] = True
            touching = np.nonzero(on_wall[elements].any(axis=1))[0]
            centroids = nodes[elements[touching, :4]].mean(axis=1)
            side = np.sign(np.asarray(plate.side(centroids), float)).astype(np.int64)
            if np.any(side == 0):
                raise ValueError(f"plate {plate.name!r}: an element lies on neither side")

            # faces of the touching elements, keyed by their corners
            tf = elements[touching][:, local_faces].reshape(-1, 6)
            owner = np.repeat(np.arange(len(touching)), 4)
            keys = np.sort(tf[:, :3], axis=1)
            wall_keys = {tuple(k) for k in np.sort(faces[:, :3], axis=1).tolist()}
            order = np.lexsort(keys.T[::-1])
            ks, tf, owner = keys[order], tf[order], owner[order]
            same = np.all(ks[1:] == ks[:-1], axis=1)
            tied = np.zeros(len(nodes), bool)
            for j in np.nonzero(same)[0]:
                if side[owner[j]] != side[owner[j + 1]] and tuple(ks[j]) not in wall_keys:
                    tied[tf[j]] = True
            splitting = on_wall & ~tied

            new_wall = np.arange(len(nodes))
            new_back = np.arange(len(nodes))
            idx = np.nonzero(splitting)[0]
            new_wall[idx] = len(nodes) + np.arange(len(idx))
            new_back[idx] = len(nodes) + len(idx) + np.arange(len(idx))
            nodes = np.vstack([nodes, nodes[idx], nodes[idx]])
            back = touching[side < 0]
            elements[back] = new_back[elements[back]]
            plate.faces = new_wall[faces]

            # interface elements, one per face on each side
            xyz = nodes[faces[:, :3]]
            n = np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0])
            n /= np.linalg.norm(n, axis=1)[:, None]
            h = np.max(np.linalg.norm(xyz[:, [1, 2, 0]] - xyz, axis=2), axis=1)
            probe = np.sign(np.asarray(plate.side(xyz.mean(axis=1) + 1e-3 * h[:, None] * n), float))
            n *= probe[:, None]                                   # now pointing into the + side
            face_of = {}
            for e in touching:
                for lf in local_faces:
                    face_of[tuple(sorted(elements[e, lf[:3]].tolist()))] = e
            for sign, soil in ((1, faces), (-1, new_back[faces])):
                for f in range(len(faces)):
                    key = tuple(sorted(soil[f, :3].tolist()))
                    if key not in face_of:
                        raise ValueError(f"plate {plate.name!r}: a face has no soil on one side")
                    support.append(face_of[key])
                soil_faces.append(soil)
                wall_faces.append(plate.faces)
                normals.append(sign * n)
                plate_of.append(np.full(len(faces), i))

        self.mesh = Mesh(nodes, elements, mesh.region)
        soil_faces = np.concatenate(soil_faces)
        support = np.array(support, dtype=np.int64)
        kn, ks, c, tan_phi, tensile = [], [], [], [], []
        for f, e in enumerate(support):
            spec = self.plates[int(np.concatenate(plate_of)[f])].interface
            mat = self.materials[int(self.mesh.region[e])]
            xyz = nodes[soil_faces[f, :3]]
            h = float(np.max(np.linalg.norm(xyz[[1, 2, 0]] - xyz, axis=1)))
            tv = max(spec.virtual_thickness * h, 1e-3)
            kn.append(mat.oedometer_modulus / tv)
            ks.append(mat.shear_modulus / tv)
            if spec.c is not None:
                c.append(spec.c)
                tan_phi.append(np.tan(np.radians(spec.phi)))
            elif hasattr(mat, "phi"):
                c.append(spec.R * mat.c)
                tan_phi.append(spec.R * np.tan(np.radians(mat.phi)))
            else:                                                  # an elastic soil has no strength
                c.append(np.inf)
                tan_phi.append(np.inf)
            tensile.append(spec.tensile)
        if len(set(tensile)) > 1:
            raise ValueError("all interfaces must have the same tensile capacity")
        self.interface_elements = InterfaceElements(
            nodes, soil_faces, np.concatenate(wall_faces), np.concatenate(normals),
            kn=np.array(kn), ks=np.array(ks), c=np.array(c), tan_phi=np.array(tan_phi), tensile=tensile[0])
        self.interface_plate = np.concatenate(plate_of)
        self.interface_support = support

    def _setup_structures(self) -> None:
        """Rotation dofs for plate nodes, element data and the combined pattern."""
        names = [p.name for p in self.plates] + [b.name for b in self.bars]
        if len(set(names)) != len(names):
            raise ValueError("structure names must be unique")
        clash = set(names) & set(self.groups)
        if clash:
            raise ValueError(f"structure and element group share the name(s) {sorted(clash)}")
        plate_nodes = np.unique(np.concatenate([p.faces.ravel() for p in self.plates])) \
            if self.plates else np.zeros(0, dtype=np.int64)
        self.rotation_of = np.full(self.mesh.n_nodes, -1, dtype=np.int64)
        self.rotation_of[plate_nodes] = self.n_translation + 3 * np.arange(len(plate_nodes))
        self.n_dof = self.n_translation + 3 * len(plate_nodes)

        self.plate_elements, self.plate_dofs = [], []
        for plate in self.plates:
            faces = np.asarray(plate.faces, dtype=np.int64)
            el = PlateElements(self.mesh.nodes, faces, plate.section)
            d = np.empty((len(faces), 36), dtype=np.int64)
            for k in range(3):
                d[:, k::6] = 3 * faces + k
                d[:, 3 + k::6] = self.rotation_of[faces] + k
            self.plate_elements.append(el)
            self.plate_dofs.append(d)

        # bars: unit vector a -> b, length, and dofs (the fixed end has none)
        self.bar_data = []
        x = self.mesh.nodes
        for bar in self.bars:
            end = x[bar.b] if bar.b is not None else np.asarray(bar.fixed_point, dtype=float)
            if bar.b is None and bar.fixed_point is None:
                raise ValueError(f"bar {bar.name!r} needs a second node or a fixed point")
            axis = end - x[bar.a]
            length = float(np.linalg.norm(axis))
            if length <= 0:
                raise ValueError(f"bar {bar.name!r} has no length")
            dofs = [3 * bar.a + k for k in range(3)]
            if bar.b is not None:
                dofs += [3 * bar.b + k for k in range(3)]
            self.bar_data.append((axis / length, length, np.array(dofs, dtype=np.int64)))

        groups = list(self.plate_dofs) + [d[None, :] for _, _, d in self.bar_data]
        self.interface_dofs = (self.interface_elements.dofs() if self.interface_elements is not None
                               else np.zeros((0, 36), dtype=np.int64))
        if len(self.interface_dofs):
            groups.append(self.interface_dofs)
        self.system_pattern = (CombinedPattern(self.pattern, groups, self.n_dof)
                               if groups else self.pattern)

    def structure_names(self) -> set[str]:
        return {p.name for p in self.plates} | {b.name for b in self.bars}

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
        for name in stage.install:
            if name not in self.structure_names():
                raise ValueError(f"stage {stage.name!r} installs unknown structure {name!r}")

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

# SPDX-License-Identifier: AGPL-3.0-only
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
from .elements import TET10_FACES, ContinuumElements, face_traction, tri6_shape
from .mesh import Mesh
from .structures import TRI6_GAUSS_BARY, TRI6_GAUSS_WEIGHT, Bar, Plate, PlateElements
from .water import DryField, HydrostaticField, PoreField, Seepage, WaterTable

INITIAL = "initial"
PLASTIC = "plastic"
SSR = "ssr"
CONSOLIDATION = "consolidation"


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
    step, ``"ssr"`` for a factor of safety by strength reduction, and
    ``"consolidation"`` for ``time`` passing while the excess pore pressure
    drains away (its loads and construction are spread over that time).

    ``water`` is the :class:`~lythos3d.core.water.WaterTable` from this stage
    on; ``None`` keeps the last one (the problem's own at the start).

    ``drained`` treats every soil as drained in this stage, whatever its
    ``drainage``, and lets the excess pore pressure built up so far drain
    away: the soil then consolidates to the drained state (a long pause, as
    PLAXIS's "ignore undrained behaviour").  Initial stages are always
    drained.
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
    water: WaterTable | None = None
    drained: bool = False
    #: consolidation: the time the stage lasts (in the time unit of the
    #: permeabilities), and the sides of the model box that drain besides
    #: the ground surface ("xmin", "xmax", "ymin", "ymax", "base")
    time: float = 0.0
    drained_sides: tuple[str, ...] = ()

    def __post_init__(self):
        if self.kind not in (INITIAL, PLASTIC, SSR, CONSOLIDATION):
            raise ValueError(f"stage {self.name!r}: unknown kind {self.kind!r}")
        if self.initial_stress not in ("k0", "gravity"):
            raise ValueError(f"stage {self.name!r}: initial_stress must be 'k0' or 'gravity'")
        self.excavate = tuple(self.excavate)
        self.construct = tuple(self.construct)
        self.install = tuple(self.install)
        self.loads = tuple(self.loads)
        self.drained_sides = tuple(self.drained_sides)
        if self.kind == CONSOLIDATION and self.time <= 0:
            raise ValueError(f"stage {self.name!r}: a consolidation stage needs a positive time")


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

    ``plates``, ``bars`` and ``piles`` (embedded) are the structures; they
    do nothing until a stage installs them.

    ``water`` is the water table at the start (dry ground without one);
    stages may change it.  With water, ``vertical_stress`` is called as
    ``vertical_stress(points, level)`` with the water level above each
    point, and returns the total overburden pressure.
    """

    mesh: Mesh
    materials: dict
    fixed: np.ndarray | None = None
    groups: dict[str, np.ndarray] = field(default_factory=dict)
    absent: tuple[str, ...] = ()
    vertical_stress: object = None
    plates: list[Plate] = field(default_factory=list)
    bars: list[Bar] = field(default_factory=list)
    piles: list = field(default_factory=list)
    water: WaterTable | None = None
    #: elements that do not exist at the start (new ground above the
    #: surface), besides the ``absent`` groups
    inactive: np.ndarray | None = None
    #: group name -> region its elements take when a stage constructs it:
    #: fill placed where ground was dug out, or a soil improved in place
    construct_region: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.water is None:
            self.water = WaterTable.dry()
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
        for name, r in self.construct_region.items():
            if name not in self.groups:
                raise ValueError(f"construct_region names unknown group {name!r}")
            if r not in self.materials:
                raise ValueError(f"no material given for region {r}")
        if self.inactive is not None:
            self.inactive = np.asarray(self.inactive, bool)
        self._split_for_interfaces()
        if self.fixed is None:
            self.fixed = box_fixities(self.mesh)
        self.fixed = np.unique(np.asarray(self.fixed, dtype=np.int64))
        self.continuum = ContinuumElements(self.mesh.nodes, self.mesh.elements)
        self.pattern = SparsityPattern(self.mesh.elements, self.mesh.n_nodes)
        self.dofs = self.continuum.dofs()
        self.n_translation = 3 * self.mesh.n_nodes
        self._setup_structures()
        self.initial_region = self.mesh.region.copy()
        self._index_regions()

    def _index_regions(self) -> None:
        ngp = self.continuum.n_gauss
        self._gauss_of_region = {
            r: (np.nonzero(self.mesh.region == r)[0][:, None] * ngp + np.arange(ngp)).ravel()
            for r in self.materials
        }

    def set_region(self, mask: np.ndarray, region: int) -> None:
        """Give the elements ``mask`` another material (a stage constructing them does)."""
        self.mesh.region[mask] = region
        self._index_regions()

    def reset_regions(self) -> None:
        """Every element back to the material it started with."""
        self.mesh.region[:] = self.initial_region
        self._index_regions()

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
        #: per plate with an interface: (original nodes, their copies behind the plate)
        self.split_nodes: dict[int, tuple[np.ndarray, np.ndarray]] = {}
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
            self.split_nodes[i] = (idx, new_back[idx])
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
        names = [p.name for p in self.plates] + [b.name for b in self.bars] + [p.name for p in self.piles]
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

        self._setup_piles()
        groups = list(self.plate_dofs) + [d[None, :] for _, _, d in self.bar_data]
        if self.piles:
            groups += [self.beam_dofs, self.skin_dofs, self.tip_dofs]
        self.interface_dofs = (self.interface_elements.dofs() if self.interface_elements is not None
                               else np.zeros((0, 36), dtype=np.int64))
        if len(self.interface_dofs):
            groups.append(self.interface_dofs)
        self.system_pattern = (CombinedPattern(self.pattern, groups, self.n_dof)
                               if groups else self.pattern)

    def _setup_piles(self) -> None:
        """Beam nodes and elements for every embedded pile, and the springs tying them in."""
        from .beams import BeamElements, EmbeddedCoupling, _GAUSS3, beam_frame, line3
        from .elements import tet10_shape

        self.pile_beams, self.pile_node_dofs = [], []
        if not self.piles:
            return
        mesh = self.mesh
        tet_dofs = self.continuum.dofs() if hasattr(self, "continuum") else None
        if tet_dofs is None:
            raise RuntimeError("piles are set up after the continuum")

        B_skin, w_skin, k_skin, t_max, skin_dofs, skin_tet, skin_pile, skin_s = [], [], [], [], [], [], [], []
        B_tip, k_tip, f_max, tip_dofs, tip_tet, tip_pile = [], [], [], [], [], []
        beam_dofs, beam_pile = [], []
        node_tet, node_N = [], []
        for i, pile in enumerate(self.piles):
            head, tip = np.asarray(pile.head, float), np.asarray(pile.tip, float)
            length = float(np.linalg.norm(tip - head))
            size = pile.element_size or max(length / 20.0, 0.25)
            n_el = max(1, int(np.ceil(length / size - 1e-9)))
            t = np.linspace(0.0, 1.0, 2 * n_el + 1)
            nodes = head + t[:, None] * (tip - head)
            elements = np.array([[2 * k, 2 * k + 1, 2 * k + 2] for k in range(n_el)])
            beam = BeamElements(nodes, elements, pile.section)
            start = self.n_dof
            self.n_dof += 6 * len(nodes)
            node_dofs = start + (6 * np.arange(len(nodes))[:, None] + np.arange(6))
            self.pile_beams.append(beam)
            self.pile_node_dofs.append(node_dofs)
            beam_dofs.append(node_dofs[elements].reshape(n_el, 18))
            beam_pile.append(np.full(n_el, i))
            e_nodes, L_nodes = mesh.locate(nodes)
            node_tet.append(e_nodes)
            node_N.append(np.array([tet10_shape(lam)[0] for lam in L_nodes]))

            R = beam_frame(tip - head)
            D = pile.section.diameter
            radius = D / 2.0
            ring = [np.cos(a) * R[1] + np.sin(a) * R[2] for a in 2 * np.pi * np.arange(8) / 8]
            base_points = [np.zeros(3)] + [0.65 * radius * (np.cos(a) * R[1] + np.sin(a) * R[2])
                                           for a in 2 * np.pi * np.arange(6) / 6]
            points, weights, along, owner, Nb, offset = [], [], [], [], [], []
            for e in range(n_el):
                for xi, w in _GAUSS3:
                    N, _ = line3(xi)
                    centre = N @ nodes[elements[e]]
                    for r in ring:
                        points.append(centre + radius * r)
                        weights.append(w * 0.5 * beam.length[e] / len(ring))
                        along.append(float(np.linalg.norm(centre - head)))
                        owner.append(e)
                        Nb.append(N)
                        offset.append(radius * r)
            n_skin = len(points)
            tets, L = mesh.locate(np.vstack(points + [tip + b for b in base_points]))

            def skew(r):
                return np.array([[0.0, -r[2], r[1]], [r[2], 0.0, -r[0]], [-r[1], r[0], 0.0]])

            G = np.array([self.materials[int(mesh.region[t])].shear_modulus for t in tets])
            Ns = np.array([tet10_shape(lam)[0] for lam in L])
            for q in range(n_skin):
                # soil minus the pile's surface point u + theta x r = u - skew(r) theta
                Bq = np.zeros((3, 48))
                for a in range(10):
                    Bq[:, 3 * a:3 * a + 3] = Ns[q, a] * R
                for a in range(3):
                    Bq[:, 30 + 6 * a:30 + 6 * a + 3] = -Nb[q][a] * R
                    Bq[:, 30 + 6 * a + 3:30 + 6 * a + 6] = Nb[q][a] * (R @ skew(offset[q]))
                B_skin.append(Bq)
                w_skin.append(weights[q])
                ks = (pile.isf_skin if pile.isf_skin is not None else 20.0 * np.pi) * G[q]
                kn = (pile.isf_lateral if pile.isf_lateral is not None else 20.0 * np.pi) * G[q]
                k_skin.append([ks, kn, kn])
                if pile.skin is None:
                    t_max.append(np.inf)
                else:
                    top, bottom = pile.skin
                    t_max.append(top + (bottom - top) * along[q] / length)
                skin_dofs.append(np.concatenate([tet_dofs[tets[q]], beam_dofs[-1][owner[q]]]))
                skin_tet.append(tets[q])
                skin_pile.append(i)
                skin_s.append(along[q])
            n_base = len(base_points)
            for j, b in enumerate(base_points):
                q = n_skin + j
                Bq = np.zeros((1, 36))
                for a in range(10):
                    Bq[0, 3 * a:3 * a + 3] = Ns[q, a] * R[0]
                Bq[0, 30:33] = -R[0]
                Bq[0, 33:36] = R[0] @ skew(b)
                B_tip.append(Bq)
                # a layer 0.1 R thick under the base: G A / (0.1 R) = 10 pi G R, shared by the points
                factor = pile.isf_base if pile.isf_base is not None else 10.0 * np.pi
                k_tip.append(factor * G[q] * radius / n_base)
                f_max.append(np.inf if pile.base is None else pile.base / n_base)
                tip_dofs.append(np.concatenate([tet_dofs[tets[q]], node_dofs[-1]]))
                tip_tet.append(tets[q])
                tip_pile.append(i)

        self.embedded = EmbeddedCoupling(np.array(B_skin), np.array(w_skin), np.array(k_skin), np.array(t_max),
                                         np.array(B_tip), np.array(k_tip), np.array(f_max))
        self.beam_dofs = np.concatenate(beam_dofs)
        self.beam_pile = np.concatenate(beam_pile)
        self.skin_dofs, self.skin_tet = np.array(skin_dofs), np.array(skin_tet)
        self.skin_pile, self.skin_s = np.array(skin_pile), np.array(skin_s)
        self.tip_dofs, self.tip_tet, self.tip_pile = np.array(tip_dofs), np.array(tip_tet), np.array(tip_pile)
        self.pile_node_tet, self.pile_node_N = node_tet, node_N

    def structure_names(self) -> set[str]:
        return {p.name for p in self.plates} | {b.name for b in self.bars} | {p.name for p in self.piles}

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
    def pore_field(self, water, active: np.ndarray, installed=()) -> PoreField:
        """The pore pressure a water table or a seepage flow sets up over the active ground.

        ``installed`` lists the plates built so far; a wall with an interface
        stops the water once it is there.
        """
        if isinstance(water, PoreField):
            return water
        if water is None or water.is_dry:
            return DryField(self.n_points)
        if isinstance(water, Seepage):
            from .seepage import solve_seepage

            return solve_seepage(self, water, active, installed)
        return HydrostaticField(water, self.continuum.gauss_xyz.reshape(-1, 3),
                                np.repeat(active, self.continuum.n_gauss))

    def gravity(self, active: np.ndarray, water=None) -> np.ndarray:
        """Nodal self weight of the active elements, saturated where there is pore pressure."""
        ce = self.continuum
        field = self.pore_field(water, active)
        gamma = np.zeros(self.n_points)
        wet = field.gauss > 0.0
        for mat, gp in self.material_groups():
            gamma[gp] = np.where(wet[gp], mat.saturated_weight, mat.gamma)
        gamma = gamma.reshape(ce.n_elements, ce.n_gauss)
        gamma[~active] = 0.0
        f = np.zeros((ce.n_elements, 30))
        f[:, 2::3] = -(gamma * ce.detJw) @ ce.N
        return assemble_vector(self.n_dof, self.dofs, f)

    # ----------------------------------------------------------------- water
    def pore_pressure(self, water, active: np.ndarray) -> np.ndarray:
        """Pore pressure at the Gauss points (compression positive), zero outside the active ground."""
        return self.pore_field(water, active).gauss

    def free_faces(self, active: np.ndarray):
        """Faces of the active elements that no other active element shares: ``(faces (n, 6), owner (n,))``."""
        el = np.nonzero(active)[0]
        faces = self.mesh.elements[el][:, np.array(TET10_FACES)].reshape(-1, 6)
        owner = np.repeat(el, 4)
        key = np.sort(faces[:, :3], axis=1)
        _, inverse, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
        free = counts[inverse.ravel()] == 1
        return faces[free], owner[free]

    def _water_on_faces(self, field: PoreField, faces: np.ndarray, owner: np.ndarray) -> np.ndarray:
        """Nodal forces (n, 18) of the water pressing on faces of the elements ``owner``.

        The pressure is taken just inside the owning element, so that a face
        on the edge of a drawdown feels the water on its own side.
        """
        x = self.mesh.nodes
        xyz = x[faces[:, :3]]
        normal = np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0])
        area = 0.5 * np.linalg.norm(normal, axis=1)
        normal /= (2.0 * area)[:, None]
        centre = x[self.mesh.elements[owner, :4]].mean(axis=1)
        outward = np.sign(np.einsum("ij,ij->i", xyz.mean(axis=1) - centre, normal))
        normal *= outward[:, None]
        f = np.zeros((len(faces), 6, 3))
        for L, w in zip(TRI6_GAUSS_BARY, TRI6_GAUSS_WEIGHT):
            N = tri6_shape(L)
            point = np.einsum("a,faj->fj", N, x[faces])
            point += 1e-6 * (centre - point)
            p = field.at(point, owner)
            f -= (w * area * p)[:, None, None] * N[None, :, None] * normal[:, None, :]
        return f.reshape(len(faces), 18)

    def water_loads(self, water, active: np.ndarray) -> np.ndarray:
        """What the pore water does to the skeleton: ``int B^T m p dV`` less the water on free faces.

        A wall with an interface stands between two free faces of the soil;
        the water pressing on those faces is carried across to the wall.
        """
        f = np.zeros(self.n_dof)
        field = self.pore_field(water, active)
        if field.is_dry:
            return f
        ce = self.continuum
        p = field.gauss.reshape(ce.n_elements, ce.n_gauss)
        # m^T B: the divergence row of B, the sum of its three normal-strain rows
        div = ce.B[:, :, 0] + ce.B[:, :, 1] + ce.B[:, :, 2]            # (ne, ng, 30)
        fe = np.einsum("egi,eg->ei", div, p * ce.detJw)
        f += assemble_vector(self.n_dof, self.dofs, fe)
        faces, owner = self.free_faces(active)
        fdofs = (3 * faces[:, :, None] + np.arange(3)).reshape(len(faces), 18)
        f += assemble_vector(self.n_dof, fdofs, self._water_on_faces(field, faces, owner))
        ie = self.interface_elements
        if ie is not None:
            live = active[self.interface_support]
            if live.any():
                on_soil = self._water_on_faces(field, ie.soil_faces[live], self.interface_support[live])
                wdofs = (3 * ie.wall_faces[live][:, :, None] + np.arange(3)).reshape(-1, 18)
                f -= assemble_vector(self.n_dof, wdofs, on_soil)
        return f

    def surface_loads(self, loads, active: np.ndarray | None = None) -> np.ndarray:
        from .analysis import AreaLoad
        from .beams import PileLoad

        f = np.zeros(self.n_dof)
        names = [p.name for p in self.piles]
        for load in loads:
            if isinstance(load, AreaLoad):
                act = np.ones(self.mesh.n_elements, bool) if active is None else active
                faces, owner = self.free_faces(act)
                xyz = self.mesh.nodes[faces[:, :3]]
                n = np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0])
                n /= np.linalg.norm(n, axis=1)[:, None]
                centre = self.mesh.nodes[self.mesh.elements[owner, :4]].mean(axis=1)
                n *= np.sign(np.einsum("ij,ij->i", xyz.mean(axis=1) - centre, n))[:, None]
                faces, traction = load.select(self.mesh.nodes, faces, n)
                if len(faces) == 0:
                    raise ValueError(f"{load.name}: no ground surface inside its polygon")
                fdofs = (3 * faces[:, :, None] + np.arange(3)).reshape(len(faces), 18)
                f += assemble_vector(self.n_dof, fdofs, face_traction(self.mesh.nodes, faces, traction))
                continue
            if isinstance(load, PileLoad):
                if load.pile not in names:
                    raise ValueError(f"a load names unknown pile {load.pile!r}")
                head = self.pile_node_dofs[names.index(load.pile)][0]
                f[head[:3]] += load.force
                f[head[3:]] += load.moment
                continue
            faces = load.faces(self.mesh)
            if len(faces) == 0:
                raise ValueError(f"surface load on {load.axis} = {load.value} found no boundary faces")
            fdofs = (3 * faces[:, :, None] + np.arange(3)).reshape(len(faces), 18)
            f += assemble_vector(self.n_dof, fdofs, face_traction(self.mesh.nodes, faces, load.traction))
        return f

    # ----------------------------------------------------------- initial state
    def k0_stress(self, active: np.ndarray, water=None, field: PoreField | None = None) -> np.ndarray:
        """Geostatic effective stress at the Gauss points: ``sv'`` from the overburden, ``sh' = K0 sv'``.

        ``sv'`` is the total overburden, saturated below the water table's
        level, less the pore pressure (``field``'s, when a seepage solution
        gives it).
        """
        if self.vertical_stress is None:
            raise ValueError("the K0 procedure needs the overburden (a stratum profile); "
                             "use initial_stress='gravity' for this problem")
        points = self.continuum.gauss_xyz.reshape(-1, 3)
        if water is None or water.is_dry:
            sv = np.asarray(self.vertical_stress(points), dtype=float)
        else:
            sv = np.asarray(self.vertical_stress(points, water.head(points)), dtype=float)
            field = self.pore_field(water, active) if field is None else field
            sv = sv - field.gauss
        k0 = np.zeros(self.n_points)
        for mat, gp in self.material_groups():
            k0[gp] = mat.k0
        stress = np.zeros((self.n_points, 6))
        stress[:, 2] = -sv
        stress[:, 0] = stress[:, 1] = -k0 * sv
        stress[~np.repeat(active, self.continuum.n_gauss)] = 0.0
        return stress

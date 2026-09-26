# SPDX-License-Identifier: AGPL-3.0-only
"""Beams in three dimensions, and piles embedded in the soil.

Beams
-----
:class:`BeamElements` are 3-node Timoshenko beams, nodes ordered (start,
middle, end), six dofs a node ``[ux, uy, uz, rx, ry, rz]`` in global axes.
In the element's own axes (1 along it, 2 and 3 across):

* axial strain ``du1/dx``, torsion ``drx1/dx``;
* bending ``k2 = dr2/dx``, ``k3 = dr3/dx``;
* shear ``g2 = du2/dx - r3``, ``g3 = du3/dx + r2``,

with ``D = diag(EA, GA2, GA3, GJ, EI2, EI3)``.  Everything is integrated by
the 2-point rule: exact for the axial, torsion and bending terms of a
straight quadratic element, and a reduced integration of shear that keeps a
slender beam from locking (as in 2D Lythos).  The element then has exactly
twelve independent strain modes for its eighteen dofs less six rigid
motions.

Embedded piles
--------------
An :class:`EmbeddedPile` is a beam whose nodes are not mesh nodes: it runs
through the tetrahedra wherever it is put, and is tied to the soil by
springs on its surface.  At each of three stations per beam element, eight
points round the perimeter move with the pile's cross-section as a rigid
disc (``u + theta x r``); the tip is tied at seven points over its base.
Coupling at the perimeter rather than on the axis matters: a load put into
the continuum along a line is a line load, whose displacement is singular,
so an axis-coupled pile gets softer without limit as the mesh is refined
(it gave twice the settlement of a pile modelled with solid elements, and
more on finer meshes).  Spread round the perimeter it is a load on a
cylinder of the pile's size, and the pile is as stiff as a real one.  The
perimeter points also resist the pile's twist, which the axis alone would
not.  At each point the relative displacement between the soil
(interpolated in the tetrahedron containing the point) and the pile's
surface is resolved along the pile and across it:

* along the pile, skin friction: elastic up to ``T_max`` (kN per metre of
  pile), then sliding - the pile's capacity in shaft friction;
* across it, elastic: the soil's lateral support of the pile;
* at the tip, end bearing: compression only, up to ``F_max`` (kN).

The springs stand for a thin layer of soil at the pile's surface, as the
interfaces of walls do: stiff, so that the pile moves with the ground
until the skin slides.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_GAUSS2 = ((-1.0 / np.sqrt(3.0), 1.0), (1.0 / np.sqrt(3.0), 1.0))
_GAUSS3 = ((-np.sqrt(0.6), 5.0 / 9.0), (0.0, 8.0 / 9.0), (np.sqrt(0.6), 5.0 / 9.0))


def line3(xi: float):
    """Shape functions (start, middle, end) on [-1, 1] and their derivatives."""
    N = np.array([0.5 * xi * (xi - 1.0), 1.0 - xi * xi, 0.5 * xi * (xi + 1.0)])
    dN = np.array([xi - 0.5, -2.0 * xi, xi + 0.5])
    return N, dN


@dataclass(frozen=True)
class BeamSection:
    """A beam's section: stiffnesses per beam (kN, kNm2) and weight per metre (kN/m)."""

    EA: float
    EI2: float
    EI3: float
    GJ: float
    GA2: float
    GA3: float
    weight: float = 0.0
    diameter: float = 0.0
    hollow: float = 0.0

    @classmethod
    def circular(cls, E: float, D: float, nu: float = 0.2, weight: float = 0.0,
                 hollow: float = 0.0) -> "BeamSection":
        """A round pile of diameter ``D``, solid or with a bore of diameter ``hollow``."""
        A = np.pi / 4.0 * (D ** 2 - hollow ** 2)
        inertia = np.pi / 64.0 * (D ** 4 - hollow ** 4)
        G = E / (2.0 * (1.0 + nu))
        kappa = 0.9 if hollow == 0.0 else 0.5            # shear coefficients of a disc and a tube
        return cls(EA=E * A, EI2=E * inertia, EI3=E * inertia, GJ=G * 2.0 * inertia,
                   GA2=kappa * G * A, GA3=kappa * G * A, weight=weight, diameter=D, hollow=hollow)

    @property
    def D(self) -> np.ndarray:
        return np.diag([self.EA, self.GA2, self.GA3, self.GJ, self.EI2, self.EI3])


def beam_frame(axis: np.ndarray) -> np.ndarray:
    """Rows: the beam's axis and two directions across it."""
    e1 = axis / np.linalg.norm(axis)
    helper = np.array([0.0, 0.0, 1.0]) if abs(e1[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e2 = np.cross(helper, e1)
    e2 /= np.linalg.norm(e2)
    return np.stack([e1, e2, np.cross(e1, e2)])


class BeamElements:
    """Straight 3-node Timoshenko beams, vectorised.

    ``nodes`` (n, 3) positions and ``elements`` (ne, 3) indices into them,
    ordered start, middle, end with the middle node halfway.
    """

    def __init__(self, nodes: np.ndarray, elements: np.ndarray, section: BeamSection):
        self.nodes = np.asarray(nodes, float)
        self.elements = np.asarray(elements, dtype=np.int64)
        self.section = section
        ends = self.nodes[self.elements[:, [0, 2]]]
        axis = ends[:, 1] - ends[:, 0]
        self.length = np.linalg.norm(axis, axis=1)
        self.R = np.stack([beam_frame(a) for a in axis])
        ne = len(self.elements)
        T = np.zeros((ne, 18, 18))
        for k in range(6):
            T[:, 3 * k:3 * k + 3, 3 * k:3 * k + 3] = self.R
        self.T = T
        self.B = []
        K = np.zeros((ne, 18, 18))
        D = section.D
        for xi, w in _GAUSS2:
            B = self._B(xi)
            self.B.append(B)
            K += (w * 0.5 * self.length)[:, None, None] * (B.transpose(0, 2, 1) @ D @ B)
        self.K = T.transpose(0, 2, 1) @ K @ T

    def _B(self, xi: float) -> np.ndarray:
        N, dN = line3(xi)
        dx = dN[None, :] * (2.0 / self.length)[:, None]            # (ne, 3)
        B = np.zeros((len(self.length), 6, 18))
        B[:, 0, 0::6] = dx                                          # axial
        B[:, 1, 1::6] = dx                                          # shear 2
        B[:, 1, 5::6] = -N
        B[:, 2, 2::6] = dx                                          # shear 3
        B[:, 2, 4::6] = N
        B[:, 3, 3::6] = dx                                          # torsion
        B[:, 4, 4::6] = dx                                          # bending about 2
        B[:, 5, 5::6] = dx                                          # bending about 3
        return B

    @property
    def n_elements(self) -> int:
        return len(self.elements)

    def self_weight(self) -> np.ndarray:
        """Consistent nodal forces of the beam's weight, (ne, 18): L/6, 2L/3, L/6."""
        f = np.zeros((self.n_elements, 18))
        share = np.array([1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0])
        f[:, 2::6] = -self.section.weight * self.length[:, None] * share[None, :]
        return f

    def resultants(self, u_elements: np.ndarray) -> np.ndarray:
        """``[N, Q2, Q3, T, M2, M3]`` at the two Gauss points of every element, (ne, 2, 6)."""
        ul = np.einsum("eij,ej->ei", self.T, u_elements)
        return np.stack([np.einsum("ij,ejk,ek->ei", self.section.D, B, ul) for B in self.B], axis=1)

    def gauss_points(self) -> np.ndarray:
        """Positions of the resultants, (ne, 2, 3)."""
        x = self.nodes[self.elements]
        return np.stack([np.einsum("n,enj->ej", line3(xi)[0], x) for xi, _ in _GAUSS2], axis=1)


@dataclass
class EmbeddedPile:
    """A pile from ``head`` to ``tip`` (3D points), tied to the soil by springs.

    ``skin`` is the shaft capacity per metre of pile (kN/m) at the head and at
    the tip, varying linearly between; ``base`` the end bearing capacity
    (kN).  Either may be ``None`` for no limit.  The springs are those of a
    layer of the surrounding soil a tenth of the pile's radius thick: per
    metre of pile ``2 pi G R / (0.1 R) = 20 pi G`` along and across it, and
    ``G A / (0.1 R)`` at the tip.  ``isf_skin``, ``isf_lateral`` and
    ``isf_base`` replace the factors on G (for the tip, on G R).
    """

    name: str
    head: tuple[float, float, float]
    tip: tuple[float, float, float]
    section: BeamSection
    skin: tuple[float, float] | None = None
    base: float | None = None
    element_size: float | None = None
    isf_skin: float | None = None
    isf_lateral: float | None = None
    isf_base: float | None = None

    def __post_init__(self):
        if np.linalg.norm(np.subtract(self.tip, self.head)) <= 0:
            raise ValueError(f"pile {self.name!r} has no length")
        if self.section.diameter <= 0:
            raise ValueError(f"pile {self.name!r}: its section needs a diameter")


@dataclass
class PileLoad:
    """A force (kN) and moment (kNm) on the head of a pile, in global axes."""

    pile: str
    force: tuple[float, float, float]
    moment: tuple[float, float, float] = (0.0, 0.0, 0.0)


class EmbeddedCoupling:
    """The springs tying embedded piles to the soil: along the shaft, and at the tips.

    Skin points carry 48 dofs each (the 30 of the tetrahedron containing
    the point, then the 18 of the beam element); tip points 36 (the
    tetrahedron's, then the tip node's six).  Tractions are
    incremental from the committed state; the skin slides along the pile at
    its capacity and the tip can neither pull nor exceed its bearing.
    """

    residual_stiffness = 1e-3

    def __init__(self, B_skin, w_skin, k_skin, t_max, B_tip, k_tip, f_max):
        self.B_skin = B_skin            # (n, 3, 48): relative movement along the pile and across it
        self.w_skin = w_skin            # (n,): integration weight, metres of pile
        self.k_skin = k_skin            # (n, 3): stiffness along and across, kN/m per m
        self.t_max = t_max              # (n,): shaft capacity, kN/m (inf for none)
        self.B_tip = B_tip              # (m, 1, 33): relative displacement along the pile
        self.k_tip = k_tip              # (m,): kN/m
        self.f_max = f_max              # (m,): kN (inf for none)

    def skin(self, u_points: np.ndarray, committed: np.ndarray):
        """``(Fe (n, 48), Ke (n, 48, 48), trial (n, 6))`` for the shaft springs."""
        d = np.einsum("nij,nj->ni", self.B_skin, u_points)
        t = committed[:, 3:] + self.k_skin * (d - committed[:, :3])
        D = np.zeros((len(d), 3, 3))
        D[:, [0, 1, 2], [0, 1, 2]] = self.k_skin
        slide = np.abs(t[:, 0]) > self.t_max
        if slide.any():
            t[slide, 0] = np.sign(t[slide, 0]) * self.t_max[slide]
            D[slide, 0, 0] = self.residual_stiffness * self.k_skin[slide, 0]
        w = self.w_skin
        Fe = w[:, None] * np.einsum("nij,ni->nj", self.B_skin, t)
        Ke = w[:, None, None] * np.einsum("nki,nkl,nlj->nij", self.B_skin, D, self.B_skin)
        return Fe, Ke, np.concatenate([d, t], axis=1)

    def tips(self, u_points: np.ndarray, committed: np.ndarray):
        """``(Fe (m, 36), Ke (m, 36, 36), trial (m, 2))`` for the tip springs.

        The traction follows ``d`` = soil minus pile along the pile's axis,
        which points from head to tip: a pile pushed on moves ahead of the
        soil, ``d`` goes negative, and so does the traction - compression.
        """
        d = np.einsum("mij,mj->mi", self.B_tip, u_points)[:, 0]
        t = committed[:, 1] + self.k_tip * (d - committed[:, 0])
        k = self.k_tip.copy()
        lifted, crushed = t > 0.0, t < -self.f_max
        t[lifted] = 0.0
        t[crushed] = -self.f_max[crushed]
        k[lifted | crushed] *= self.residual_stiffness
        Fe = self.B_tip[:, 0, :] * t[:, None]
        Ke = k[:, None, None] * np.einsum("mi,mj->mij", self.B_tip[:, 0], self.B_tip[:, 0])
        return Fe, Ke, np.column_stack([d, t])

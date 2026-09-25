"""Structural elements: plates for walls and rafts, bars for anchors and struts.

Plates
------
:class:`PlateElements` are 6-node flat triangular shells on faces of the
10-node tetrahedra, so a wall shares its nodes with the soil.  Each node has
the three translations it already had and three rotations about the global
axes.  In the element's own frame (x, y in its plane, z normal):

* membrane:  ``N = Dm [du/dx, dv/dy, du/dy + dv/dx]``, plane stress;
* bending (Reissner-Mindlin): ``u(z) = u + z thy``, ``v(z) = v - z thx``, so
  ``k = [dthy/dx, -dthx/dy, dthy/dy - dthx/dx]`` and ``M = Db k``;
* transverse shear: ``g = [dw/dx + thy, dw/dy - thx]``, ``Q = Ds g``;
* drilling: the rotation about the normal is tied to the in-plane rotation
  of the membrane, ``thz = (dv/dx - du/dy) / 2``, by a penalty (Hughes and
  Brezzi).  Tying it rather than springing it to zero keeps a rigid
  rotation free of energy, so the element has exactly six rigid body modes.

Mindlin elements lock in shear when thin and coarse.  The shear stiffness is
therefore scaled by ``t^2 / (t^2 + a h^2)`` (Lyly, Stenberg and Vihinen), which
leaves thick plates alone and stops a thin plate from being stiffer than
Kirchhoff theory allows; ``a`` is ``PlateSection.shear_stabilisation``.

Membrane and bending are integrated by the 3-point rule, exact on a flat
element; shear and drilling by the 6-point rule, exact too.  Integrating
those two by 3 points loses rank and leaves spurious zero-energy modes.

Anchors and struts
------------------
:class:`Bar` is a two-node elastic bar between two mesh nodes, or between a
mesh node and a fixed point.  It is stressed to its lock-off load in the stage
that installs it: during that stage it is a pair of forces, and from then on
it carries that force plus ``EA / L`` times its extension.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .elements import TRI_GAUSS_BARY, TRI_GAUSS_WEIGHT

_A1, _W1 = 0.445948490915965, 0.223381589678011
_A2, _W2 = 0.091576213509771, 0.109951743655322
#: 6-point rule for triangles, exact to degree 4: the transverse shear term,
#: whose integrand is quartic, integrated exactly
TRI6_GAUSS_BARY = np.array([
    [_A1, _A1, 1 - 2 * _A1], [_A1, 1 - 2 * _A1, _A1], [1 - 2 * _A1, _A1, _A1],
    [_A2, _A2, 1 - 2 * _A2], [_A2, 1 - 2 * _A2, _A2], [1 - 2 * _A2, _A2, _A2],
])
TRI6_GAUSS_WEIGHT = np.array([_W1, _W1, _W1, _W2, _W2, _W2])


@dataclass(frozen=True)
class PlateSection:
    """An isotropic plate: modulus ``E`` (kPa), Poisson's ratio, thickness ``t`` (m).

    ``weight`` is the self weight per unit area (kN/m2), acting downwards.
    For a wall it is usually the weight of the wall less that of the soil it
    replaces, as in 2D Lythos.
    """

    E: float
    nu: float
    t: float
    weight: float = 0.0
    #: ``a`` in the shear scaling ``t^2 / (t^2 + a h^2)``.  0.03 is calibrated on a
    #: simply supported plate with t/a = 0.005: on a 4 x 4 mesh no stabilisation
    #: locks it 19% too stiff, 0.1 makes it 10% too soft, 0.03 is within 3%
    shear_stabilisation: float = 0.03

    def __post_init__(self):
        if self.E <= 0 or self.t <= 0:
            raise ValueError("a plate needs positive E and thickness")

    @classmethod
    def from_stiffness(cls, EA: float, EI: float, nu: float = 0.2, weight: float = 0.0) -> "PlateSection":
        """The plate with axial stiffness ``EA`` (kN/m) and bending stiffness ``EI`` (kNm2/m).

        This is how a pile or diaphragm wall is specified per metre run:
        ``t = sqrt(12 EI / EA)`` and ``E = EA / t``.
        """
        t = float(np.sqrt(12.0 * EI / EA))
        return cls(E=EA / t, nu=nu, t=t, weight=weight)

    @property
    def EA(self) -> float:
        return self.E * self.t

    @property
    def EI(self) -> float:
        return self.E * self.t ** 3 / 12.0

    def _plane(self, scale: float) -> np.ndarray:
        nu = self.nu
        return scale / (1.0 - nu * nu) * np.array([[1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, 0.5 * (1.0 - nu)]])

    @property
    def Dm(self) -> np.ndarray:
        return self._plane(self.E * self.t)

    @property
    def Db(self) -> np.ndarray:
        return self._plane(self.E * self.t ** 3 / 12.0)

    @property
    def G(self) -> float:
        return self.E / (2.0 * (1.0 + self.nu))


def _tri6(L):
    """Shape functions of the 6-node triangle (corners, then mid-edges ab, bc, ca) and
    their derivatives with respect to the natural coordinates (L1, L2)."""
    a, b, c = L
    N = np.array([a * (2 * a - 1), b * (2 * b - 1), c * (2 * c - 1), 4 * a * b, 4 * b * c, 4 * c * a])
    dL = np.array([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]])            # d(a, b, c)/d(L1, L2)
    dN = np.array([
        (4 * a - 1) * dL[0], (4 * b - 1) * dL[1], (4 * c - 1) * dL[2],
        4 * (b * dL[0] + a * dL[1]), 4 * (c * dL[1] + b * dL[2]), 4 * (a * dL[2] + c * dL[0]),
    ]).T
    return N, dN


class PlateElements:
    """Flat 6-node shells on triangle faces, vectorised over the elements.

    ``faces`` (nf, 6) are node indices ordered corners then mid-edges
    (ab, bc, ca), as :meth:`Mesh.boundary_faces` gives them.  Element vectors
    and matrices are ordered node by node, ``[ux, uy, uz, rx, ry, rz]`` in
    global axes.
    """

    def __init__(self, nodes: np.ndarray, faces: np.ndarray, section: PlateSection):
        self.faces = np.asarray(faces, dtype=np.int64)
        self.section = section
        xyz = nodes[self.faces]                                        # (nf, 6, 3)
        nf = len(self.faces)
        e1 = xyz[:, 1] - xyz[:, 0]
        normal = np.cross(e1, xyz[:, 2] - xyz[:, 0])
        area2 = np.linalg.norm(normal, axis=1)
        if np.any(area2 <= 1e-14 * np.max(area2)):
            raise ValueError("a plate element has no area")
        e1 /= np.linalg.norm(e1, axis=1)[:, None]
        e3 = normal / area2[:, None]
        e2 = np.cross(e3, e1)
        self.R = np.stack([e1, e2, e3], axis=1)                        # rows: local axes in global
        local = np.einsum("fij,fnj->fni", self.R, xyz - xyz[:, :1])    # (nf, 6, 3)
        xy = local[:, :, :2]

        s = section
        h = np.max(np.linalg.norm(xy[:, [1, 2, 0]] - xy[:, [0, 1, 2]], axis=2), axis=1)
        a = s.shear_stabilisation
        ds = (5.0 / 6.0) * s.G * s.t * s.t ** 2 / (s.t ** 2 + a * h ** 2)   # (nf,)
        drill = 1e-3 * s.G * s.t

        K = np.zeros((nf, 36, 36))
        self.area = np.zeros(nf)
        self._gauss = []
        for L, w in zip(TRI_GAUSS_BARY, TRI_GAUSS_WEIGHT):
            N, dN = _tri6(L)
            J = np.einsum("kn,fnj->fkj", dN, xy)                       # (nf, 2, 2)
            det = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
            dNxy = np.linalg.solve(J, np.broadcast_to(dN, (nf, 2, 6)))  # (nf, 2, 6)
            dx, dy = dNxy[:, 0], dNxy[:, 1]
            wd = 0.5 * w * det
            self.area += wd
            Bm = np.zeros((nf, 3, 36))
            Bm[:, 0, 0::6] = dx
            Bm[:, 1, 1::6] = dy
            Bm[:, 2, 0::6] = dy
            Bm[:, 2, 1::6] = dx
            Bb = np.zeros((nf, 3, 36))
            Bb[:, 0, 4::6] = dx
            Bb[:, 1, 3::6] = -dy
            Bb[:, 2, 4::6] = dy
            Bb[:, 2, 3::6] = -dx
            Bs = self._shear_B(N, dx, dy)
            K += wd[:, None, None] * (
                Bm.transpose(0, 2, 1) @ (s.Dm @ Bm)
                + Bb.transpose(0, 2, 1) @ (s.Db @ Bb))
            self._gauss.append((N, Bm, Bb, Bs, ds))
        # Transverse shear and drilling, integrated exactly.  Their integrands
        # are quartic, and under the 3-point rule each loses rank: the
        # drilling term then leaves three spurious zero-energy modes in the
        # plane of the element (nine modes with no stiffness instead of the
        # six rigid body motions).  The 6-point rule removes them, and the
        # shear stabilisation above keeps a thin plate from locking.
        for L, w in zip(TRI6_GAUSS_BARY, TRI6_GAUSS_WEIGHT):
            N, dN = _tri6(L)
            J = np.einsum("kn,fnj->fkj", dN, xy)
            det = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
            dNxy = np.linalg.solve(J, np.broadcast_to(dN, (nf, 2, 6)))
            dx, dy = dNxy[:, 0], dNxy[:, 1]
            Bs = self._shear_B(N, dx, dy)
            Bd = np.zeros((nf, 1, 36))
            Bd[:, 0, 5::6] = N
            Bd[:, 0, 0::6] = 0.5 * dy
            Bd[:, 0, 1::6] = -0.5 * dx
            K += (0.5 * w * det)[:, None, None] * (
                ds[:, None, None] * (Bs.transpose(0, 2, 1) @ Bs)
                + drill * (Bd.transpose(0, 2, 1) @ Bd))
        # to global axes: every node's translations and rotations turn with R
        T = np.zeros((nf, 36, 36))
        for k in range(12):
            T[:, 3 * k:3 * k + 3, 3 * k:3 * k + 3] = self.R
        self.T = T
        self.K = T.transpose(0, 2, 1) @ K @ T

    @staticmethod
    def _shear_B(N, dx, dy):
        Bs = np.zeros((len(dx), 2, 36))
        Bs[:, 0, 2::6] = dx
        Bs[:, 0, 4::6] = N
        Bs[:, 1, 2::6] = dy
        Bs[:, 1, 3::6] = -N
        return Bs

    @property
    def n_elements(self) -> int:
        return len(self.faces)

    def stiffness(self) -> np.ndarray:
        """Element stiffness matrices in global axes, (nf, 36, 36)."""
        return self.K

    def self_weight(self) -> np.ndarray:
        """Consistent nodal forces from the plate's weight, (nf, 36)."""
        f = np.zeros((self.n_elements, 36))
        # integral of each shape function over a flat triangle, per unit area
        weights = sum(w * _tri6(L)[0] for L, w in zip(TRI_GAUSS_BARY, TRI_GAUSS_WEIGHT))
        f[:, 2::6] = -self.section.weight * self.area[:, None] * weights[None, :]
        return f

    def resultants(self, u_elements: np.ndarray):
        """Stress resultants at the Gauss points, in each element's own axes.

        ``u_elements`` (nf, 36) in global axes.  Returns ``N`` (nf, 3, 3)
        membrane forces [Nxx, Nyy, Nxy] (kN/m), ``M`` (nf, 3, 3) moments
        [Mxx, Myy, Mxy] (kNm/m) and ``Q`` (nf, 3, 2) shear forces (kN/m),
        indexed by element, Gauss point, component.
        """
        s = self.section
        ul = np.einsum("fij,fj->fi", self.T, u_elements)
        N, M, Q = [], [], []
        for _, Bm, Bb, Bs, ds in self._gauss:
            N.append(np.einsum("ij,fjk,fk->fi", s.Dm, Bm, ul))
            M.append(np.einsum("ij,fjk,fk->fi", s.Db, Bb, ul))
            Q.append(ds[:, None] * np.einsum("fjk,fk->fj", Bs, ul))
        return np.stack(N, axis=1), np.stack(M, axis=1), np.stack(Q, axis=1)


@dataclass
class Plate:
    """A named plate - a wall, a raft - on the mesh faces ``faces`` (nf, 6).

    With ``interface`` (an :class:`~lythos3d.core.interfaces.InterfaceSpec`)
    the soil may slip against the plate on both faces; ``side`` then tells
    the two faces apart: a function of points (n, 3) returning +1 or -1.
    """

    name: str
    faces: np.ndarray
    section: PlateSection
    interface: object = None
    side: object = None


@dataclass
class Bar:
    """An anchor or strut: an elastic bar from node ``a`` to node ``b``.

    With ``b`` None the far end is held fixed at ``fixed_point`` (a strut
    against a symmetry plane, or an anchor whose grout body is taken as
    rigid).  ``EA`` is per bar (kN), and ``prestress`` (kN, tension positive)
    the lock-off load applied in the installing stage: positive for an
    anchor, which pulls the wall back; negative for a strut, which is jacked
    against the wall.
    """

    name: str
    a: int
    b: int | None
    EA: float
    prestress: float = 0.0
    fixed_point: tuple[float, float, float] | None = None

# SPDX-License-Identifier: AGPL-3.0-only
"""Element formulations.

* :class:`ContinuumElements` - 10-node quadratic tetrahedra, vectorised over
  the whole mesh.

Node order follows VTK's quadratic tetrahedron so that results go to ParaView
without a permutation: the four corners, then the mid-edge nodes of the edges
(0,1), (1,2), (0,2), (0,3), (1,3), (2,3).

Stresses and strains are stored in Voigt order ``[xx, yy, zz, xy, yz, zx]``
with engineering shear strains, tension positive.  ``z`` points upwards, so
gravity acts in ``-z``.
"""

from __future__ import annotations

import numpy as np

#: local corner pairs of the six mid-edge nodes 4..9
TET10_EDGES = ((0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3))

#: the four faces as 6-node triangles (corners, then mid-edge nodes in the
#: order ab, bc, ca), each ordered so that its normal points out of the element
TET10_FACES = (
    (0, 2, 1, 6, 5, 4),
    (0, 1, 3, 4, 8, 7),
    (1, 2, 3, 5, 9, 8),
    (0, 3, 2, 7, 9, 6),
)

_A = 0.5854101966249685
_B = 0.1381966011250105

#: 4-point Gauss rule for tetrahedra as barycentric coordinates, exact to
#: degree 2 - which is the degree of B^T D B on a straight-sided element
TET_GAUSS_BARY = np.array([
    [_A, _B, _B, _B],
    [_B, _A, _B, _B],
    [_B, _B, _A, _B],
    [_B, _B, _B, _A],
])
TET_GAUSS_WEIGHT = np.full(4, 0.25)

#: 3-point Gauss rule for triangles as barycentric coordinates, exact to degree 2
TRI_GAUSS_BARY = np.array([
    [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0],
    [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
    [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0],
])
TRI_GAUSS_WEIGHT = np.full(3, 1.0 / 3.0)

#: derivatives of the barycentric coordinates L0..L3 with respect to the
#: natural coordinates (xi, eta, zeta) = (L1, L2, L3)
_DL = np.array([
    [-1.0, -1.0, -1.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
])


def tet10_shape(L: np.ndarray):
    """Shape functions and natural derivatives of the 10-node tetrahedron.

    ``L`` holds the four barycentric coordinates.  Returns ``N`` (10,) and
    ``dN`` (3, 10), derivatives with respect to (xi, eta, zeta).
    """
    L = np.asarray(L, dtype=float)
    N = np.empty(10)
    dN = np.empty((3, 10))
    for i in range(4):
        N[i] = L[i] * (2.0 * L[i] - 1.0)
        dN[:, i] = (4.0 * L[i] - 1.0) * _DL[i]
    for k, (i, j) in enumerate(TET10_EDGES):
        N[4 + k] = 4.0 * L[i] * L[j]
        dN[:, 4 + k] = 4.0 * (L[j] * _DL[i] + L[i] * _DL[j])
    return N, dN


def tri6_shape(L: np.ndarray) -> np.ndarray:
    """Shape functions of the 6-node triangle ordered (a, b, c, ab, bc, ca)."""
    a, b, c = L
    return np.array([
        a * (2.0 * a - 1.0), b * (2.0 * b - 1.0), c * (2.0 * c - 1.0),
        4.0 * a * b, 4.0 * b * c, 4.0 * c * a,
    ])


def elastic_matrix(E: float, nu: float) -> np.ndarray:
    """6x6 isotropic elasticity matrix for ``[xx, yy, zz, xy, yz, zx]``."""
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    D = np.zeros((6, 6))
    D[:3, :3] = lam
    D[0, 0] = D[1, 1] = D[2, 2] = lam + 2.0 * mu
    D[3, 3] = D[4, 4] = D[5, 5] = mu
    return D


class ContinuumElements:
    """All 10-node tetrahedra of a mesh, with pre-computed strain operators."""

    n_gauss = 4
    dof_per_node = 3
    n_dof_element = 30

    def __init__(self, nodes: np.ndarray, elements: np.ndarray):
        self.nodes = nodes
        self.elements = elements
        ne = len(elements)
        self.B = np.zeros((ne, self.n_gauss, 6, 30))
        self.detJw = np.zeros((ne, self.n_gauss))
        self.N = np.zeros((self.n_gauss, 10))
        self.gauss_xyz = np.zeros((ne, self.n_gauss, 3))

        xyz = nodes[elements]                                  # (ne, 10, 3)
        for g, (L, w) in enumerate(zip(TET_GAUSS_BARY, TET_GAUSS_WEIGHT)):
            N, dN = tet10_shape(L)
            self.N[g] = N
            J = np.einsum("kn,enj->ekj", dN, xyz)              # (ne, 3, 3)
            det = np.linalg.det(J)
            if np.any(det <= 0):
                bad = int(np.argmin(det))
                raise ValueError(f"element {bad} has a non-positive Jacobian ({det[bad]:.3e})")
            dNxyz = np.linalg.solve(J, np.broadcast_to(dN, (ne, 3, 10)))   # (ne, 3, 10)
            dx, dy, dz = dNxyz[:, 0], dNxyz[:, 1], dNxyz[:, 2]
            Bg = self.B[:, g]
            Bg[:, 0, 0::3] = dx
            Bg[:, 1, 1::3] = dy
            Bg[:, 2, 2::3] = dz
            Bg[:, 3, 0::3] = dy
            Bg[:, 3, 1::3] = dx
            Bg[:, 4, 1::3] = dz
            Bg[:, 4, 2::3] = dy
            Bg[:, 5, 0::3] = dz
            Bg[:, 5, 2::3] = dx
            self.detJw[:, g] = det * w / 6.0                   # 1/6 = volume of the reference tetrahedron
            self.gauss_xyz[:, g] = np.einsum("n,enj->ej", N, xyz)

    @property
    def n_elements(self) -> int:
        return len(self.elements)

    @property
    def n_points(self) -> int:
        return self.n_elements * self.n_gauss

    def dofs(self) -> np.ndarray:
        """Global dof indices of every element, shape (ne, 30)."""
        d = np.empty((len(self.elements), 30), dtype=np.int64)
        for k in range(3):
            d[:, k::3] = 3 * self.elements + k
        return d

    def strains(self, u: np.ndarray) -> np.ndarray:
        """Strain at every Gauss point, shape (ne * ngp, 6)."""
        ue = u[self.dofs()]                                    # (ne, 30)
        eps = np.matmul(self.B, ue[:, None, :, None])[..., 0]
        return eps.reshape(-1, 6)

    def internal_forces(self, stress: np.ndarray) -> np.ndarray:
        """Element internal force vectors, shape (ne, 30)."""
        sig = stress.reshape(self.n_elements, self.n_gauss, 6) * self.detJw[:, :, None]
        return np.matmul(self.B.transpose(0, 1, 3, 2), sig[..., None])[..., 0].sum(axis=1)

    def stiffness(self, tangent: np.ndarray) -> np.ndarray:
        """Element stiffness matrices, shape (ne, 30, 30).

        ``tangent`` is either one 6x6 matrix per Gauss point, shape
        (ne * ngp, 6, 6), or one per element, shape (ne, 6, 6).
        """
        if tangent.shape[0] == self.n_elements and tangent.ndim == 3:
            D = tangent[:, None]                               # (ne, 1, 6, 6)
        else:
            D = tangent.reshape(self.n_elements, self.n_gauss, 6, 6)
        # Scale B by the square root of the integration weight and stack the
        # Gauss points, so that each element costs one (30 x 24)(24 x 30)
        # product instead of four (30 x 6)(6 x 30) ones: twenty times faster,
        # because numpy's batched matmul pays per product, not per flop.
        ne = self.n_elements
        Bw = self.B * np.sqrt(self.detJw)[:, :, None, None]   # (ne, ngp, 6, 30)
        DB = np.matmul(D, Bw).reshape(ne, 6 * self.n_gauss, 30)
        return np.matmul(Bw.reshape(ne, 6 * self.n_gauss, 30).transpose(0, 2, 1), DB)

    def body_force(self, gamma_per_element: np.ndarray) -> np.ndarray:
        """Consistent nodal forces from a vertical body force (downwards)."""
        f = np.zeros((self.n_elements, 30))
        f[:, 2::3] = -np.outer(gamma_per_element, np.ones(10)) * (self.detJw @ self.N)
        return f

    def volumes(self) -> np.ndarray:
        return self.detJw.sum(axis=1)

    def nodal_average(self, values: np.ndarray, n_nodes: int) -> np.ndarray:
        """Smooth Gauss point values onto the nodes.

        The Gauss point values of each element are extrapolated to its corners
        through the linear field they define - exact for the linear stress of a
        straight-sided quadratic element - the mid-edge nodes take the mean of
        their two corners, and every node then averages over its elements.
        """
        ne = self.n_elements
        v = values.reshape(ne, self.n_gauss, -1)
        corner = np.einsum("cg,egk->eck", np.linalg.inv(TET_GAUSS_BARY), v)
        mids = np.stack([0.5 * (corner[:, i] + corner[:, j]) for i, j in TET10_EDGES], axis=1)
        per_node = np.concatenate([corner, mids], axis=1)      # (ne, 10, k)
        total = np.zeros((n_nodes, per_node.shape[2]))
        count = np.zeros(n_nodes)
        np.add.at(total, self.elements.ravel(), per_node.reshape(-1, per_node.shape[2]))
        np.add.at(count, self.elements.ravel(), 1.0)
        return total / np.maximum(count, 1.0)[:, None]


def face_traction(nodes: np.ndarray, faces: np.ndarray, traction: np.ndarray) -> np.ndarray:
    """Consistent nodal forces from a uniform traction on 6-node triangle faces.

    ``faces`` (nf, 6) are node indices, ``traction`` (nf, 3) or (3,) is force
    per unit area.  Returns forces of shape (nf, 18) ordered node by node.
    """
    xyz = nodes[faces[:, :3]]
    area = 0.5 * np.linalg.norm(np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0]), axis=1)
    # integral of each shape function over a flat triangle, per unit area
    weights = sum(w * tri6_shape(L) for L, w in zip(TRI_GAUSS_BARY, TRI_GAUSS_WEIGHT))
    t = np.broadcast_to(np.asarray(traction, dtype=float), (len(faces), 3))
    f = area[:, None, None] * weights[None, :, None] * t[:, None, :]
    return f.reshape(len(faces), 18)


def face_normals(nodes: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Unit normals of flat triangle faces, following their node order."""
    xyz = nodes[faces[:, :3]]
    n = np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0])
    return n / np.linalg.norm(n, axis=1)[:, None]

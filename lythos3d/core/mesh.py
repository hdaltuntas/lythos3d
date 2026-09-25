"""Tetrahedral meshes of 10-node elements.

:func:`box_mesh` builds a structured mesh of a rectangular block, which is all
the first stage of Lythos 3D needs: layered ground in a box.  Arbitrary
geometry will come through an unstructured mesher; everything downstream of
:class:`Mesh` is independent of how the mesh was made.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations

import numpy as np

from .elements import TET10_EDGES, TET10_FACES


@dataclass
class Mesh:
    """Nodes, 10-node tetrahedra and the region each element belongs to."""

    nodes: np.ndarray                       # (n, 3)
    elements: np.ndarray                    # (ne, 10), VTK quadratic tetra order
    region: np.ndarray = field(default=None)  # (ne,) integer region ids

    def __post_init__(self):
        self.nodes = np.asarray(self.nodes, dtype=float)
        self.elements = np.asarray(self.elements, dtype=np.int64)
        if self.region is None:
            self.region = np.zeros(len(self.elements), dtype=np.int64)
        self.region = np.asarray(self.region, dtype=np.int64)

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_elements(self) -> int:
        return len(self.elements)

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self.nodes.min(axis=0), self.nodes.max(axis=0)

    def centroids(self) -> np.ndarray:
        return self.nodes[self.elements[:, :4]].mean(axis=1)

    def boundary_faces(self) -> np.ndarray:
        """Faces used by exactly one element, as outward 6-node triangles (nf, 6)."""
        faces = self.elements[:, np.array(TET10_FACES)].reshape(-1, 6)
        key = np.sort(faces[:, :3], axis=1)
        _, inverse, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
        return faces[counts[inverse.ravel()] == 1]

    def faces_in_box(self, lo, hi, tol: float = 1e-9) -> np.ndarray:
        """Every element face (interior or boundary) whose corners all lie in the box.

        A box of zero thickness in one direction is a rectangle in a plane,
        which is how a wall is found.  Each face is returned once, as a 6-node
        triangle (corners, then mid-edges ab, bc, ca).
        """
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        faces = self.elements[:, np.array([(0, 1, 2, 4, 5, 6), (0, 1, 3, 4, 8, 7),
                                           (1, 2, 3, 5, 9, 8), (0, 2, 3, 6, 9, 7)])].reshape(-1, 6)
        corners = self.nodes[faces[:, :3]]
        inside = np.all((corners >= lo - tol) & (corners <= hi + tol), axis=(1, 2))
        faces = faces[inside]
        _, first = np.unique(np.sort(faces[:, :3], axis=1), axis=0, return_index=True)
        return faces[np.sort(first)]

    def nearest_node(self, point, corners_only: bool = True) -> int:
        candidates = np.unique(self.elements[:, :4]) if corners_only else np.arange(self.n_nodes)
        d = np.linalg.norm(self.nodes[candidates] - np.asarray(point, float), axis=1)
        return int(candidates[np.argmin(d)])

    def faces_on_plane(self, axis: int, value: float, tol: float = 1e-9) -> np.ndarray:
        """Boundary faces lying in the plane ``x[axis] == value``."""
        faces = self.boundary_faces()
        on = np.all(np.abs(self.nodes[faces][:, :, axis] - value) < tol, axis=1)
        return faces[on]

    def mapped(self, transform) -> "Mesh":
        """A copy with the corner nodes moved by ``transform`` (n, 3) -> (n, 3).

        Mid-edge nodes are put back at the middle of their edges, so the
        elements stay straight-sided.  This is how a structured block becomes
        a slope or a tapered layer.
        """
        nodes = self.nodes.copy()
        corners = np.unique(self.elements[:, :4])
        nodes[corners] = transform(nodes[corners])
        for k, (i, j) in enumerate(TET10_EDGES):
            nodes[self.elements[:, 4 + k]] = 0.5 * (nodes[self.elements[:, i]] + nodes[self.elements[:, j]])
        return Mesh(nodes, self.elements.copy(), self.region.copy())

    def quality(self) -> np.ndarray:
        """Radius ratio ``3 r_in / r_circ`` of every element's corner tetrahedron.

        1 for a regular tetrahedron, 0 for a flat one.  A soil boundary that
        crosses an excavation level at a shallow angle leaves a thin wedge of
        ground, and no mesh can fill a wedge of angle a with elements much
        better than about a (0.08 for a 5 degree dip); such elements, down to
        about 0.005, are harmless to the direct solver.  A value near zero is
        a degenerate element and a fault in the mesh.
        """
        p = self.nodes[self.elements[:, :4]]
        a, b, c, d = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
        ab, ac, ad = b - a, c - a, d - a
        six_v = np.abs(np.einsum("ij,ij->i", np.cross(ab, ac), ad))
        area = sum(0.5 * np.linalg.norm(np.cross(q - o, r - o), axis=1)
                   for o, q, r in ((a, b, c), (a, b, d), (a, c, d), (b, c, d)))
        r_in = 0.5 * six_v / area
        num = (np.einsum("ij,ij->i", ad, ad)[:, None] * np.cross(ab, ac)
               + np.einsum("ij,ij->i", ac, ac)[:, None] * np.cross(ad, ab)
               + np.einsum("ij,ij->i", ab, ab)[:, None] * np.cross(ac, ad))
        r_circ = np.linalg.norm(num, axis=1) / (2.0 * six_v)
        return 3.0 * r_in / r_circ

    def assign_regions(self, classify) -> None:
        """Set ``region`` from a function of the element centroids (ne, 3) -> (ne,)."""
        self.region = np.asarray(classify(self.centroids()), dtype=np.int64)


def graded(start: float, stop: float, size: float, breaks=()) -> np.ndarray:
    """Grid coordinates from ``start`` to ``stop`` with spacing at most ``size``.

    Every value in ``breaks`` between the ends is kept as a grid line, so that
    layer boundaries and excavation levels fall on element faces.
    """
    stops = sorted({float(start), float(stop), *(float(b) for b in breaks if start < b < stop)})
    out = [stops[0]]
    for a, b in zip(stops[:-1], stops[1:]):
        n = max(1, int(np.ceil((b - a) / size - 1e-9)))
        out.extend(np.linspace(a, b, n + 1)[1:])
    return np.array(out)


def box_mesh(xs, ys, zs) -> Mesh:
    """Structured 10-node tetrahedral mesh of a block on the grid ``xs, ys, zs``.

    Each hexahedral cell is cut into six tetrahedra around its main diagonal
    (the Kuhn subdivision).  Every cell is cut the same way, so the diagonals
    on shared faces agree and the mesh is conforming.
    """
    xs, ys, zs = (np.asarray(a, dtype=float) for a in (xs, ys, zs))
    nx, ny, nz = len(xs) - 1, len(ys) - 1, len(zs) - 1
    if min(nx, ny, nz) < 1:
        raise ValueError("each direction needs at least two grid coordinates")
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    corners = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    def index(i, j, k):
        return (i * (ny + 1) + j) * (nz + 1) + k

    i, j, k = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    i, j, k = i.ravel(), j.ravel(), k.ravel()
    tets = []
    for perm in permutations(range(3)):
        path = [np.zeros(3, dtype=int)]
        for axis in perm:
            step = path[-1].copy()
            step[axis] = 1
            path.append(step)
        tets.append(np.column_stack([index(i + p[0], j + p[1], k + p[2]) for p in path]))
    tets = np.concatenate(tets)

    # orient every tetrahedron positively
    p = corners[tets]
    vol = np.einsum("ij,ij->i", np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), p[:, 3] - p[:, 0])
    flip = vol < 0
    tets[flip, 1], tets[flip, 2] = tets[flip, 2].copy(), tets[flip, 1].copy()
    return quadratic_from_linear(corners, tets)


def quadratic_from_linear(nodes: np.ndarray, tets: np.ndarray, region=None) -> Mesh:
    """Add a node at the middle of every edge of a 4-node tetrahedral mesh."""
    tets = np.asarray(tets, dtype=np.int64)
    edges = np.sort(tets[:, np.array(TET10_EDGES)].reshape(-1, 2), axis=1)
    unique, inverse = np.unique(edges, axis=0, return_inverse=True)
    mid = 0.5 * (nodes[unique[:, 0]] + nodes[unique[:, 1]])
    elements = np.concatenate([tets, len(nodes) + inverse.reshape(-1, 6)], axis=1)
    return Mesh(np.vstack([nodes, mid]), elements, region)

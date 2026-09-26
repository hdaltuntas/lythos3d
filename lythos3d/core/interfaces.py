# SPDX-License-Identifier: AGPL-3.0-only
"""Zero-thickness interfaces between a wall and the soil.

An interface element joins a 6-node face of the soil to the matching face of
the wall (the same positions, different nodes), 12 nodes in all.  The
relative displacement ``d = u_soil - u_wall`` is resolved in the element's
frame: ``dn`` along the normal, which points from the wall into the soil so
that a positive value opens a gap, and ``ds`` in the plane.  The tractions
follow a Mohr-Coulomb contact:

* elastic, ``tn = kn dn``, ``ts = ks ds``, with the stiffnesses of a virtual
  layer of soil a tenth of an element thick (as in 2D Lythos);
* sliding when ``|ts| > c - tn tan(phi)`` (tn is negative in contact): the
  shear traction is returned radially onto the limit, which keeps its
  direction in the plane, with no dilation;
* open when ``tn`` exceeds the tensile capacity (zero by default): no
  traction at all.

The update is incremental from the last converged state, so unloading after
slip is elastic.  The tangent is the consistent one of the radial return,
including the coupling d(ts)/d(dn) = -tan(phi) kn m that sliding brings;
2D Lythos found that leaving it out stalls the global iteration.

Before its wall is installed an interface ties the two sides rigidly (a
penalty a hundred times the contact stiffness), since the ground is
continuous until the wall is built.

Integration is by the 6-point rule, exact for the quartic integrand: the
3-point rule would give each element rank 9 against its 18 relative dofs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .structures import TRI6_GAUSS_BARY, TRI6_GAUSS_WEIGHT, _tri6

#: state per Gauss point: relative displacement (n, s1, s2), then traction (n, s1, s2)
STATE_WIDTH = 6
N_GAUSS = len(TRI6_GAUSS_WEIGHT)


@dataclass(frozen=True)
class InterfaceSpec:
    """How a wall meets the soil.

    ``R`` scales the adjacent soil's strength: ``c_i = R c'`` and
    ``tan(phi_i) = R tan(phi')``.  Give ``c`` (kPa) and ``phi`` (degrees)
    instead to set the interface's own strength - a wall friction angle
    ``delta`` - whatever the soil.  The contact stiffness is that of a layer
    of the soil ``virtual_thickness`` times the local element size thick.
    ``tensile`` is the tensile capacity (kPa); zero lets a gap open freely.
    """

    R: float = 0.67
    virtual_thickness: float = 0.1
    tensile: float = 0.0
    c: float | None = None
    phi: float | None = None

    def __post_init__(self):
        if not 0.0 < self.R <= 1.0:
            raise ValueError("R must lie in (0, 1]")
        if (self.c is None) != (self.phi is None):
            raise ValueError("give both c and phi for the interface, or neither")


class InterfaceElements:
    """Interface elements over matching soil and wall faces, vectorised.

    ``residual_stiffness`` is the fraction of the elastic stiffness kept
    where the exact tangent has none (an open gap, the shear of a sliding
    point), so that the global matrix stays invertible.

    ``soil_faces`` and ``wall_faces`` (n, 6) list nodes at the same positions
    in the same order; ``normals`` (n, 3) point from the wall into the soil.
    ``kn``, ``ks``, ``c``, ``tan_phi`` are per element; ``tan_phi`` may be
    infinite for a soil without strength (a purely elastic contact).
    """

    def __init__(self, nodes, soil_faces, wall_faces, normals, kn, ks, c, tan_phi, tensile=0.0):
        self.soil_faces = np.asarray(soil_faces, dtype=np.int64)
        self.wall_faces = np.asarray(wall_faces, dtype=np.int64)
        n = len(self.soil_faces)
        self.kn = np.broadcast_to(np.asarray(kn, float), (n,)).copy()
        self.ks = np.broadcast_to(np.asarray(ks, float), (n,)).copy()
        self.c = np.broadcast_to(np.asarray(c, float), (n,)).copy()
        self.tan_phi = np.broadcast_to(np.asarray(tan_phi, float), (n,)).copy()
        self.tensile = float(tensile)
        normals = np.asarray(normals, float)
        normals = normals / np.linalg.norm(normals, axis=1)[:, None]
        xyz = nodes[self.wall_faces]
        t1 = xyz[:, 1] - xyz[:, 0]
        t1 -= np.einsum("ij,ij->i", t1, normals)[:, None] * normals
        t1 /= np.linalg.norm(t1, axis=1)[:, None]
        t2 = np.cross(normals, t1)
        self.R = np.stack([normals, t1, t2], axis=1)                # rows: n, s1, s2

        # B maps the element's 36 dofs (soil nodes, then wall nodes) to the
        # relative displacement in local axes at each Gauss point; w is the
        # integration weight including the area
        area2 = np.linalg.norm(np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0]), axis=1)
        self.B = np.zeros((n, N_GAUSS, 3, 36))
        self.w = np.zeros((n, N_GAUSS))
        for g, (L, wt) in enumerate(zip(TRI6_GAUSS_BARY, TRI6_GAUSS_WEIGHT)):
            N, _ = _tri6(L)
            for a in range(6):
                self.B[:, g, :, 3 * a:3 * a + 3] = N[a] * self.R
                self.B[:, g, :, 18 + 3 * a:18 + 3 * a + 3] = -N[a] * self.R
            self.w[:, g] = 0.5 * wt * area2
        self.area = 0.5 * area2
        self.tie = 100.0 * np.maximum(self.kn, self.ks)
        self.residual_stiffness = 1e-3

    @property
    def n_elements(self) -> int:
        return len(self.soil_faces)

    def dofs(self) -> np.ndarray:
        """Global dofs, (n, 36): the soil nodes' translations, then the wall nodes'."""
        nodes = np.concatenate([self.soil_faces, self.wall_faces], axis=1)
        return (3 * nodes[:, :, None] + np.arange(3)).reshape(len(nodes), 36)

    def relative(self, u_elements: np.ndarray) -> np.ndarray:
        """Relative displacement in local axes, (n, ngp, 3)."""
        return np.einsum("egij,ej->egi", self.B, u_elements)

    #: contact modes, per Gauss point
    STICK, SLIDE, OPEN = 0, 1, 2

    def respond(self, u_elements: np.ndarray, committed: np.ndarray, rigid: np.ndarray,
                strength_factor: float = 1.0, modes: np.ndarray | None = None,
                pore: np.ndarray | None = None):
        """Tractions, element forces and tangents.

        ``committed`` (n, ngp, 6) is the converged state, ``rigid`` (n,) marks
        elements still tying the two sides (wall not yet installed), and
        ``strength_factor`` divides c and tan(phi), as strength reduction
        does.  ``modes`` (n, ngp), when given, holds every point in the
        contact state it had rather than letting it switch: a point near the
        limit that sticks in one iteration and slides in the next leaves an
        imbalance Newton cannot remove.  Returns ``(Fe (n, 36), Ke (n, 36, 36),
        trial state, modes)``.
        """
        n = self.n_elements
        d = self.relative(u_elements)                                       # (n, g, 3)
        k = np.stack([self.kn, self.ks, self.ks], axis=1)[:, None, :]       # (n, 1, 3)
        trial_t = committed[..., 3:] + k * (d - committed[..., :3])
        t = trial_t.copy()
        D = np.zeros((n, N_GAUSS, 3, 3))
        D[..., 0, 0], D[..., 1, 1], D[..., 2, 2] = self.kn[:, None], self.ks[:, None], self.ks[:, None]

        c = self.c[:, None] / strength_factor
        tan_phi = self.tan_phi[:, None] / strength_factor
        residual = self.residual_stiffness
        tn = trial_t[..., 0]
        opened = tn > self.tensile
        # Friction acts on the effective normal stress.  Next to soil loaded
        # undrained the contact carries the skeleton and its excess pore
        # water together; ``pore`` (n,), that excess (compression positive),
        # is taken off the contact pressure for the strength.  It is the
        # committed value, so the tangent leaves out its change.
        tn_eff = tn if pore is None else np.minimum(tn + pore[:, None], 0.0)
        tau = trial_t[..., 1:]
        tau_norm = np.linalg.norm(tau, axis=2)
        with np.errstate(invalid="ignore"):
            limit = np.where(np.isinf(tan_phi), np.inf, c - tn_eff * np.where(np.isinf(tan_phi), 0.0, tan_phi))
        sliding = ~opened & (tau_norm > limit)
        if modes is not None:
            opened = modes == self.OPEN
            sliding = modes == self.SLIDE
        mode = np.where(opened, self.OPEN, np.where(sliding, self.SLIDE, self.STICK))

        # open: no traction, a token stiffness
        if opened.any():
            t[opened] = [self.tensile, 0.0, 0.0]
            D[opened] = residual * D[opened]
        # sliding: radial return of the shear traction onto the limit
        if sliding.any():
            m = tau[sliding] / tau_norm[sliding][:, None]
            lim = np.maximum(limit[sliding], 0.0)
            lim = np.where(np.isfinite(lim), lim, tau_norm[sliding])
            t[sliding, 1:] = lim[:, None] * m
            ks = np.broadcast_to(self.ks[:, None], sliding.shape)[sliding]
            kn = np.broadcast_to(self.kn[:, None], sliding.shape)[sliding]
            tp = np.broadcast_to(tan_phi, sliding.shape)[sliding]
            ratio = lim / tau_norm[sliding]
            proj = np.eye(2)[None] - m[:, :, None] * m[:, None, :]
            Ds = np.zeros((len(m), 3, 3))
            Ds[:, 0, 0] = kn
            Ds[:, 1:, 1:] = ks[:, None, None] * (ratio[:, None, None] * proj + residual * np.eye(2)[None])
            Ds[:, 1:, 0] = -tp[:, None] * kn[:, None] * m
            D[sliding] = Ds

        # not yet installed: a rigid tie on the total relative displacement
        if rigid.any():
            t[rigid] = self.tie[rigid, None, None] * d[rigid]
            D[rigid] = self.tie[rigid, None, None, None] * np.eye(3)

        Fe = np.einsum("eg,egij,egi->ej", self.w, self.B, t)
        Ke = np.einsum("eg,egki,egkl,eglj->eij", self.w, self.B, D, self.B, optimize=True)
        trial = np.concatenate([d, t], axis=2)
        return Fe, Ke, trial, mode

    def centroids(self, nodes: np.ndarray) -> np.ndarray:
        return nodes[self.wall_faces[:, :3]].mean(axis=1)

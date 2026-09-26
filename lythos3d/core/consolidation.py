# SPDX-License-Identifier: AGPL-3.0-only
"""Consolidation: the excess pore pressure dissipating in time, coupled to the soil (Biot).

Displacements stay on the 10-node tetrahedra; the excess pore pressure ``p``
(compression positive) lives on their four corners and varies linearly -
the Taylor-Hood pair, which satisfies the inf-sup condition, so the pressure
does not oscillate in the first steps after a load as equal-order
interpolation does.

Equilibrium and the storage equation, with ``Q = int B^T m N_p dV``, the
flow matrix ``H = int grad N_p^T (k / gamma_w) grad N_p dV`` and the storage
``S = int N_p^T (n / K_w) N_p dV`` (``n / K_w`` the inverse of each soil's
:attr:`~lythos3d.core.materials.LinearElastic.water_bulk_modulus`), read

    f_int(u) - Q p = f_ext
    Q^T du/dt + S dp/dt + H p = 0

and are integrated by backward Euler.  Each time step is a Newton
iteration on both, with the soil's consistent tangent in the displacement
block:

    [ K     -Q        ] [du]   [ f_ext - f_int + Q p                   ]
    [ -Q^T  -(S + dt H)] [dp] = [ Q^T (u - u_n) + S (p - p_n) + dt H p ]

a symmetric, indefinite system.

**Boundaries.**  The excess pore pressure is zero on the ground surface -
every free face of the active ground not on the model box - and on the
sides of the box named as drained.  The base and the other sides are
closed.  A wall with an interface is impermeable once installed, as in
seepage.

**Units.**  Only ``k / gamma_w`` and time appear together: with ``k`` in m
per day, time is in days.
"""

from __future__ import annotations

import numpy as np

from .elements import TET10_FACES, TET_GAUSS_BARY

SIDES = ("xmin", "xmax", "ymin", "ymax", "base")


def time_steps(duration: float, n: int, ratio: float | None = None) -> np.ndarray:
    """``n`` time steps growing geometrically and summing to ``duration``.

    Consolidation is fastest at the start, so the steps start small; but
    backward Euler is only first order, so the last steps must shrink with
    ``n`` too: by default each is ``1 + 2/n`` times the one before, which
    makes the first about a fiftieth and the last about a ninth of the
    time at 20 steps.
    """
    if duration <= 0:
        raise ValueError("a consolidation stage needs a positive time")
    ratio = 1.0 + 2.0 / max(n, 1) if ratio is None else ratio
    w = ratio ** np.arange(n)
    return duration * w / w.sum()


class PressureSpace:
    """Corner-node excess pore pressure over the active ground, and the matrices of the coupling."""

    def __init__(self, problem, active: np.ndarray, installed=(), drained_sides=(), gamma_w: float = 9.81):
        unknown = set(drained_sides) - set(SIDES)
        if unknown:
            raise ValueError(f"unknown drained side(s) {sorted(unknown)}; use {SIDES}")
        mesh, ce = problem.mesh, problem.continuum
        n = mesh.n_nodes
        rep = np.arange(n)
        for i, (orig, back) in problem.split_nodes.items():
            if i not in set(installed):
                rep[back] = orig
        self.rep = rep
        act = np.nonzero(active)[0]
        self.act = act
        corners = rep[mesh.elements[act, :4]]
        nodes = np.unique(corners)
        self.index = np.full(n, -1, dtype=np.int64)
        self.index[nodes] = np.arange(len(nodes))
        self.nodes = nodes
        self.n = len(nodes)
        self.pdofs = self.index[corners]                            # (na, 4)
        self.udofs = problem.dofs[act]                              # (na, 30)

        x = mesh.nodes
        P = x[mesh.elements[act, :4]]
        J = np.transpose(P[:, 1:] - P[:, :1], (0, 2, 1))            # columns: edges
        inv = np.linalg.inv(J)                                      # rows: d(lambda_1..3)/dx
        dL = np.concatenate([-inv.sum(axis=1, keepdims=True), inv], axis=1)   # (na, 4, 3)
        self.dL = np.transpose(dL, (0, 2, 1))                       # (na, 3, 4)
        w = ce.detJw[act]                                           # (na, 4)
        Np = TET_GAUSS_BARY                                         # (4 gauss, 4 corners)
        self.Np = Np
        div = ce.B[act, :, 0] + ce.B[act, :, 1] + ce.B[act, :, 2]   # (na, 4, 30)
        self.Qe = np.einsum("ega,eg,gb->eab", div, w, Np)           # (na, 30, 4)

        kdiag = np.zeros((len(act), 3))
        storage = np.zeros(len(act))
        for r, mat in problem.materials.items():
            mine = mesh.region[act] == r
            kh = float(getattr(mat, "k", 1.0))
            kv = getattr(mat, "k_v", None)
            kdiag[mine] = (kh, kh, kh if kv is None else float(kv))
            storage[mine] = 1.0 / mat.water_bulk_modulus if hasattr(mat, "water_bulk_modulus") else 0.0
        vol = w.sum(axis=1)
        self.He = np.einsum("eia,ei,eib->eab", self.dL, kdiag / gamma_w, self.dL) * vol[:, None, None]
        self.Se = np.einsum("eg,ga,gb->eab", w, Np, Np) * storage[:, None, None]
        self.weights = np.einsum("eg,ga->ea", w, Np)                # int N_p dV per element corner

        # drained boundary: free faces of the active ground (walls with
        # interfaces excluded), on the surface and on drained sides
        faces = rep[mesh.elements[act]][:, np.array(TET10_FACES)[:, :3]].reshape(-1, 3)
        key = np.sort(faces, axis=1)
        _, inverse, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
        free = counts[inverse.ravel()] == 1
        faces, key = faces[free], key[free]
        ie = problem.interface_elements
        if ie is not None and len(problem.interface_plate):
            walled = np.isin(problem.interface_plate, list(installed)) & active[problem.interface_support]
            if walled.any():
                wall_keys = {tuple(k) for k in np.sort(rep[ie.soil_faces[walled, :3]], axis=1).tolist()}
                faces = faces[np.array([tuple(k) not in wall_keys for k in key.tolist()], dtype=bool)]
        lo, hi = mesh.bounds
        eps = 1e-6 * float(max(hi - lo))
        c = x[faces]

        def on(axis, value):
            return np.all(np.abs(c[:, :, axis] - value) < eps, axis=1)

        side = {"xmin": on(0, lo[0]), "xmax": on(0, hi[0]), "ymin": on(1, lo[1]), "ymax": on(1, hi[1]),
                "base": on(2, lo[2])}
        box = np.zeros(len(faces), bool)
        drained = np.zeros(len(faces), bool)
        for name, mask in side.items():
            box |= mask
            if name in drained_sides:
                drained |= mask
        drained |= ~box
        self.drained = np.unique(self.index[faces[drained].ravel()])

    def from_gauss(self, excess: np.ndarray, n_gauss: int) -> np.ndarray:
        """Nodal pressure from Gauss point values: their average weighted by ``int N_p``."""
        g = excess.reshape(-1, n_gauss)[self.act]                   # (na, 4)
        num = np.zeros(self.n)
        den = np.zeros(self.n)
        wN = self.weights
        # the Gauss values projected per element onto its corners (exact for
        # a linear field), then averaged
        corner = g @ np.linalg.inv(self.Np).T
        np.add.at(num, self.pdofs.ravel(), (corner * wN).ravel())
        np.add.at(den, self.pdofs.ravel(), wN.ravel())
        return num / np.maximum(den, 1e-300)

    def to_gauss(self, p: np.ndarray, n_elements: int, n_gauss: int) -> np.ndarray:
        out = np.zeros((n_elements, n_gauss))
        out[self.act] = p[self.pdofs] @ self.Np.T
        return out.ravel()

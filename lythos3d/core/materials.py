"""Constitutive models.

Units are kN, m and kPa throughout, as in the 2D program: stiffness and
strength in kPa, unit weight in kN/m3.  Stresses are tension positive and
stored as ``[sxx, syy, szz, sxy, syz, szx]``; strains in the same order with
engineering shear strains.

The Mohr-Coulomb update is the one Lythos uses in 2D, an exact return mapping
in principal stress space (Clausen, Damkilde & Andersen, 2006).  The criterion
is linear in the sorted principal stresses, so the return to a face, an edge
or the apex is a small linear system solved in closed form, and nothing in it
depends on the number of dimensions.  What 3D changes is getting into and out
of principal space: a symmetric 3x3 eigenproblem instead of a 2D rotation,
and a consistent tangent that carries a spin term for each of the three pairs
of principal axes rather than one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from .elements import elastic_matrix

#: tensor indices of the six Voigt components
VOIGT_PAIRS = ((0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (2, 0))


# ---------------------------------------------------------------------------
# principal space
# ---------------------------------------------------------------------------
def voigt_to_tensor(s: np.ndarray) -> np.ndarray:
    """(n, 6) Voigt stresses to (n, 3, 3) symmetric tensors."""
    t = np.empty((len(s), 3, 3))
    t[:, 0, 0], t[:, 1, 1], t[:, 2, 2] = s[:, 0], s[:, 1], s[:, 2]
    t[:, 0, 1] = t[:, 1, 0] = s[:, 3]
    t[:, 1, 2] = t[:, 2, 1] = s[:, 4]
    t[:, 2, 0] = t[:, 0, 2] = s[:, 5]
    return t


def principal_stresses(s: np.ndarray):
    """Principal values sorted ``s1 >= s2 >= s3`` and their directions.

    Returns ``(values (n, 3), vectors (n, 3, 3))`` with the direction of
    ``values[:, k]`` in column ``k`` of ``vectors``.
    """
    w, V = np.linalg.eigh(voigt_to_tensor(s))          # ascending
    return w[:, ::-1], V[:, :, ::-1]


def principal_values(s: np.ndarray) -> np.ndarray:
    """Principal values sorted ``s1 >= s2 >= s3``, in closed form.

    The trigonometric solution of the characteristic cubic: several times
    faster than an eigensolver, and accurate to round-off relative to the
    stress magnitude, which is all a yield check needs.
    """
    q = s[:, :3].sum(axis=1) / 3.0
    d = s[:, :3] - q[:, None]
    p1 = (s[:, 3:] ** 2).sum(axis=1)
    p = np.sqrt(np.maximum(((d ** 2).sum(axis=1) + 2.0 * p1) / 6.0, 0.0))
    safe = np.where(p > 0.0, p, 1.0)
    b = d / safe[:, None]
    t = s[:, 3:] / safe[:, None]
    det = (b[:, 0] * (b[:, 1] * b[:, 2] - t[:, 1] ** 2)
           - t[:, 0] * (t[:, 0] * b[:, 2] - t[:, 1] * t[:, 2])
           + t[:, 2] * (t[:, 0] * t[:, 1] - b[:, 1] * t[:, 2]))
    angle = np.arccos(np.clip(0.5 * det, -1.0, 1.0)) / 3.0
    s1 = q + 2.0 * p * np.cos(angle)
    s3 = q + 2.0 * p * np.cos(angle + 2.0 * np.pi / 3.0)
    return np.column_stack([s1, 3.0 * q - s1 - s3, s3])


def from_principal(values: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Voigt stresses from principal values and directions."""
    t = np.einsum("nik,nk,njk->nij", V, values, V)
    return np.stack([t[:, i, j] for i, j in VOIGT_PAIRS], axis=1)


def rotation_to_global(V: np.ndarray) -> np.ndarray:
    """Voigt transformation ``T`` taking stresses from the principal frame to global axes.

    ``sigma_global = T sigma_principal`` and, with engineering shear strains,
    ``eps_principal = T^T eps_global``, so a principal-frame stiffness maps to
    global axes as ``T D T^T``.
    """
    n = len(V)
    T = np.empty((n, 6, 6))
    for row, (i, j) in enumerate(VOIGT_PAIRS):
        for col, (k, m) in enumerate(VOIGT_PAIRS):
            if k == m:
                T[:, row, col] = V[:, i, k] * V[:, j, k]
            else:
                T[:, row, col] = V[:, i, k] * V[:, j, m] + V[:, i, m] * V[:, j, k]
    return T


# ---------------------------------------------------------------------------
# material state
# ---------------------------------------------------------------------------
@dataclass
class MaterialState:
    """Per-Gauss-point history."""

    stress: np.ndarray            # (n, 6) current stress
    plastic_strain: np.ndarray    # (n, 6) accumulated plastic strain
    eps_p_eq: np.ndarray          # (n,) equivalent plastic strain
    yielding: np.ndarray          # (n,) bool, plastic at the last update
    #: interface state, (n_interface_elements, n_gauss, 6), carried with the
    #: soil's so that it is committed and rolled back together with it
    interface: np.ndarray | None = None
    #: embedded piles: shaft springs (n, 6) and tip springs (m, 2)
    embedded: np.ndarray | None = None
    tips: np.ndarray | None = None

    @classmethod
    def zeros(cls, n: int) -> "MaterialState":
        return cls(np.zeros((n, 6)), np.zeros((n, 6)), np.zeros(n), np.zeros(n, bool))

    def copy(self) -> "MaterialState":
        return MaterialState(self.stress.copy(), self.plastic_strain.copy(),
                             self.eps_p_eq.copy(), self.yielding.copy(),
                             *(None if a is None else a.copy() for a in (self.interface, self.embedded, self.tips)))

    def take(self, idx) -> "MaterialState":
        return MaterialState(self.stress[idx], self.plastic_strain[idx],
                             self.eps_p_eq[idx], self.yielding[idx])

    def put(self, idx, other: "MaterialState") -> None:
        self.stress[idx] = other.stress
        self.plastic_strain[idx] = other.plastic_strain
        self.eps_p_eq[idx] = other.eps_p_eq
        self.yielding[idx] = other.yielding


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LinearElastic:
    """Isotropic linear elasticity.

    ``K0`` is the ratio of horizontal to vertical stress the K0 procedure
    uses; by default it is the elastic value under zero lateral strain,
    ``nu / (1 - nu)``.
    """

    name: str
    E: float
    nu: float
    gamma: float = 0.0
    K0: float | None = None

    def __post_init__(self):
        if self.E <= 0:
            raise ValueError(f"{self.name}: E must be positive")
        if not -1.0 < self.nu < 0.5:
            raise ValueError(f"{self.name}: nu must lie in (-1, 0.5)")

    def elastic(self) -> np.ndarray:
        return elastic_matrix(self.E, self.nu)

    @property
    def shear_modulus(self) -> float:
        return self.E / (2.0 * (1.0 + self.nu))

    @property
    def oedometer_modulus(self) -> float:
        """Constrained modulus ``M``, the stiffness under zero lateral strain."""
        return self.E * (1.0 - self.nu) / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))

    @property
    def k0(self) -> float:
        """Ratio of horizontal to vertical stress at rest."""
        return self.nu / (1.0 - self.nu) if self.K0 is None else self.K0

    def reduced(self, srf: float) -> "LinearElastic":
        """Material with strength divided by ``srf``; elastic ones have none."""
        return self

    def update(self, state: MaterialState, dstrain: np.ndarray):
        """Advance ``state`` by ``dstrain``: ``(stress, tangent (n, 6, 6), new state)``."""
        D = self.elastic()
        stress = state.stress + dstrain @ D.T
        n = len(stress)
        new = MaterialState(stress, state.plastic_strain.copy(), state.eps_p_eq.copy(),
                            np.zeros(n, bool))
        return stress, np.broadcast_to(D, (n, 6, 6)).copy(), new


@dataclass(frozen=True)
class MohrCoulomb(LinearElastic):
    """Elastic-perfectly plastic Mohr-Coulomb with a tension cut-off.

    ``psi < phi`` gives non-associated flow, the realistic choice for soil.
    Its consistent tangent is unsymmetric and is used as it is: symmetrising
    it would cost Newton its convergence rate exactly where the soil fails.
    The default ``K0`` is Jaky's ``1 - sin(phi)``.
    """

    c: float = 5.0             # effective cohesion [kPa]
    phi: float = 30.0          # effective friction angle [deg]
    psi: float = 0.0           # dilatancy angle [deg]
    #: fraction of the elastic stiffness kept where the material has no
    #: strength left (the apex, or a point fully in tension), so that the
    #: global matrix stays invertible where a crack has opened
    residual_stiffness: float = 1.0e-3
    #: tensile strength [kPa]; 0 carries no tension at all, None removes the
    #: cut-off so that the material may reach the apex at c cot(phi)
    tension_cutoff: float | None = 0.0

    def __post_init__(self):
        super().__post_init__()
        if self.c < 0 or not 0.0 <= self.phi < 90.0:
            raise ValueError(f"{self.name}: need c >= 0 and 0 <= phi < 90")
        if self.psi > self.phi:
            raise ValueError(f"{self.name}: psi may not exceed phi")

    @property
    def k0(self) -> float:
        return 1.0 - math.sin(math.radians(self.phi)) if self.K0 is None else self.K0

    def reduced(self, srf: float) -> "MohrCoulomb":
        """Strength divided by ``srf``: ``c / srf`` and ``tan(phi) / srf``."""
        phi_r = math.degrees(math.atan(math.tan(math.radians(self.phi)) / srf))
        psi_r = math.degrees(math.atan(math.tan(math.radians(self.psi)) / srf))
        cut = None if self.tension_cutoff is None else self.tension_cutoff / srf
        return replace(self, c=self.c / srf, phi=phi_r, psi=min(psi_r, phi_r),
                       tension_cutoff=cut)

    # ------------------------------------------------------------------ update
    def update(self, state: MaterialState, dstrain: np.ndarray):
        D = self.elastic()
        trial = state.stress + dstrain @ D.T
        n = len(trial)
        stress = trial.copy()
        tangent = np.broadcast_to(D, (n, 6, 6)).copy()
        plastic = np.zeros(n, bool)

        # Most points stay elastic, and for those the principal directions are
        # never needed: screen them with closed-form principal values, and
        # solve the eigenproblem only where the trial stress is at or beyond
        # the surface.  The margin covers round-off in the closed form.
        sphi, _spsi, k = self._params()
        v = principal_values(trial)
        margin = 1e-8 * (np.abs(v).max(axis=1) + abs(k) + 1.0)
        near = (((1.0 + sphi) * v[:, 0] - (1.0 - sphi) * v[:, 2] - k > -margin)
                | (v[:, 0] > self.tension_limit() - margin))
        if near.any():
            idx = np.nonzero(near)[0]
            values, V = principal_stresses(trial[idx])
            returned, D_principal, yielded = self._return_map(values)
            if yielded.any():
                p = idx[yielded]
                plastic[p] = True
                stress[p] = from_principal(returned[yielded], V[yielded])
                tangent[p] = self._cartesian_tangent(D_principal[yielded], values[yielded],
                                                     returned[yielded], V[yielded])

        dplastic = dstrain - (stress - state.stress) @ np.linalg.inv(D).T
        dplastic[~plastic] = 0.0
        vol = dplastic[:, :3].sum(axis=1, keepdims=True) / 3.0
        dev = dplastic.copy()
        dev[:, :3] -= vol
        deq = np.sqrt(np.maximum(2.0 / 3.0 * ((dev[:, :3] ** 2).sum(axis=1)
                                              + 0.5 * (dev[:, 3:] ** 2).sum(axis=1)), 0.0))
        new = MaterialState(stress, state.plastic_strain + dplastic, state.eps_p_eq + deq, plastic)
        return stress, tangent, new

    def yield_function(self, stress: np.ndarray) -> np.ndarray:
        """Mohr-Coulomb yield function, positive outside the surface."""
        s = principal_values(stress)
        sphi, _spsi, k = self._params()
        return (1.0 + sphi) * s[:, 0] - (1.0 - sphi) * s[:, 2] - k

    # ---------------------------------------------------------- principal space
    def _params(self):
        sphi = math.sin(math.radians(self.phi))
        spsi = math.sin(math.radians(self.psi))
        cphi = math.cos(math.radians(self.phi))
        return sphi, spsi, 2.0 * self.c * cphi

    def _elastic_principal(self) -> np.ndarray:
        lam = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
        return lam * np.ones((3, 3)) + 2.0 * self.shear_modulus * np.eye(3)

    def tension_limit(self) -> float:
        """Largest admissible major principal stress (tension positive)."""
        apex = self.c / math.tan(math.radians(self.phi)) if self.phi > 1e-9 else math.inf
        if self.tension_cutoff is None:
            return apex
        return min(self.tension_cutoff, apex)

    def _criteria(self):
        """Yield planes as (normal, flow direction, offset) in sorted principal space.

        The Mohr-Coulomb pyramid contributes its main face and the two faces
        bounding the sorted sector: meeting the main face, f(s2, s3) gives the
        edge s1 = s2 and f(s1, s2) the edge s2 = s3.  The tension cut-off
        contributes one plane per principal stress.
        """
        sphi, spsi, k = self._params()
        st = self.tension_limit()
        eye = np.eye(3)
        return {
            "main": (np.array([1.0 + sphi, 0.0, -1.0 + sphi]),
                     np.array([1.0 + spsi, 0.0, -1.0 + spsi]), k),
            "s12": (np.array([1.0 + sphi, -1.0 + sphi, 0.0]),
                    np.array([1.0 + spsi, -1.0 + spsi, 0.0]), k),
            "s23": (np.array([0.0, 1.0 + sphi, -1.0 + sphi]),
                    np.array([0.0, 1.0 + spsi, -1.0 + spsi]), k),
            "t0": (eye[0], eye[0], st),
            "t1": (eye[1], eye[1], st),
            "t2": (eye[2], eye[2], st),
        }

    #: candidate active sets, tried in order of increasing size; the first
    #: that returns an admissible stress with non-negative multipliers is the
    #: right region for that point
    ACTIVE_SETS = (
        ("main",),
        ("t0",),
        ("main", "s23"),
        ("main", "s12"),
        ("main", "t0"),
        ("t0", "t1"),
        ("main", "s23", "t0"),
        ("main", "s12", "t0"),
        ("main", "t0", "t1"),
        ("t0", "t1", "t2"),
    )

    def _return_map(self, s: np.ndarray):
        """Map sorted trial principal stresses back onto the yield surface.

        ``s`` holds s1 >= s2 >= s3.  For a given set of active planes the
        return is the solution of a small linear system and its consistent
        tangent follows in closed form; which planes are active is found by
        trying the candidate sets in turn.  Returns the principal stresses,
        the principal-space tangent (n, 3, 3) and which points yielded.
        """
        crit = self._criteria()
        a_main, _b, k = crit["main"]
        De = self._elastic_principal()
        n = len(s)
        st_max = self.tension_limit()
        scale = max(abs(k), self.E * 1e-6, 1.0)
        tol = 1e-10 * scale

        plastic = (s @ a_main - k > tol) | (s[:, 0] > st_max + tol)
        out = s.copy()
        tangents = np.broadcast_to(De, (n, 3, 3)).copy()
        if not plastic.any():
            return out, tangents, plastic

        idx = np.nonzero(plastic)[0]
        trial = s[idx]
        res = trial.copy()
        tang = np.broadcast_to(De, (len(idx), 3, 3)).copy()
        unsolved = np.ones(len(idx), bool)

        for names in self.ACTIVE_SETS:
            if not unsolved.any():
                break
            if st_max == math.inf and any(name.startswith("t") for name in names):
                continue
            A = np.column_stack([crit[nm][0] for nm in names])
            B = np.column_stack([crit[nm][1] for nm in names])
            offsets = np.array([crit[nm][2] for nm in names])
            M = A.T @ De @ B
            if abs(np.linalg.det(M)) < 1e-12 * scale ** len(names):
                continue
            Minv = np.linalg.inv(M)
            rows = np.nonzero(unsolved)[0]
            lam = (trial[rows] @ A - offsets) @ Minv.T
            cand = trial[rows] - lam @ (De @ B).T
            ok = (np.all(lam >= -1e-9 * scale, axis=1)
                  & (cand @ a_main - k <= 1e-7 * scale)
                  & (cand[:, 0] <= st_max + 1e-7 * scale)
                  & (cand[:, 0] >= cand[:, 1] - 1e-7 * scale)
                  & (cand[:, 1] >= cand[:, 2] - 1e-7 * scale))
            take = rows[ok]
            res[take] = cand[ok]
            tang[take] = De - (De @ B) @ Minv @ (A.T @ De)
            unsolved[take] = False

        if unsolved.any():
            # The tip of the cone: the only admissible point is the apex (or
            # the cut-off, where that is lower).  It carries no strength,
            # hence the token stiffness.
            sub = np.nonzero(unsolved)[0]
            limit = st_max if math.isfinite(st_max) else float(np.min(trial[sub]))
            res[sub] = limit
            tang[sub] = self.residual_stiffness * De

        out[idx] = res
        tangents[idx] = tang
        return out, tangents, plastic

    def _cartesian_tangent(self, D_pr, trial_pr, final_pr, V) -> np.ndarray:
        """Rotate a principal-space tangent into global axes.

        Besides the principal stiffness, the rotation of the principal axes
        contributes, for each pair (a, b), the shear stiffness
        ``mu (sa - sb) / (sa_trial - sb_trial)``, which is the shear modulus
        for an elastic step.  Leaving it out would cost quadratic convergence
        wherever the material is plastic.
        """
        n = len(trial_pr)
        mu = self.shear_modulus
        Dp = np.zeros((n, 6, 6))
        Dp[:, :3, :3] = D_pr
        for slot, (a, b) in zip((3, 4, 5), ((0, 1), (1, 2), (2, 0))):
            d_trial = trial_pr[:, a] - trial_pr[:, b]
            close = np.abs(d_trial) <= 1e-10 * max(self.E, 1.0)
            ratio = (final_pr[:, a] - final_pr[:, b]) / np.where(close, 1.0, d_trial)
            Dp[:, slot, slot] = np.where(close, mu, np.clip(mu * ratio, self.residual_stiffness * mu, mu))
        T = rotation_to_global(V)
        return np.matmul(T, np.matmul(Dp, T.transpose(0, 2, 1)))

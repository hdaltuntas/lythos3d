"""Steady seepage: the head over the ground, with a phreatic surface found as part of the solution.

Darcy's law ``q = -k grad h`` and continuity ``div q = 0`` give, for the
total head ``h = z + p / gamma_w``,

    div (k grad h) = 0

solved on the same 10-node tetrahedra as the soil, with ``h`` at every node.
The permeability is diagonal, ``k`` horizontally and ``k_v`` vertically,
per soil.  The element matrix ``int G^T k G dV`` uses the soil's Gauss
points.

**Unconfined flow.**  Where does the water stop?  Above the phreatic surface
the pressure head ``psi = h - z`` is negative, and the permeability is cut
back log-linearly from its full value at ``psi = 0`` to ``k_min`` times it
at ``psi = -psi_k``.  Almost no water then flows above the surface, which
settles where the flow puts it.  The permeability depends on ``h``, so the
equations are non-linear, and are solved by Newton's method with the
change of permeability in the Jacobian.  (Picard iteration - solve, update
the permeability, solve again - cycles on a sharp transition and never
settles.)

**Boundaries.**  The open sides of the model box hold the far-field head
below the water table, and are seepage face candidates above it; the base
and closed sides carry no flow.  On the
ground (every free face that is not on the box), water standing above it
holds the head at its level: a lake, a flooded pit, a pit pumped down to a
drawdown level.  Ground above the water is a *seepage face* candidate:
water may leave there at atmospheric pressure (``h = z``) but may not enter.
Each iteration holds ``h = z`` on the candidates where water leaves, and
frees the ones where the held head would draw water in - until every held
point has water flowing out and no free one has pressure above zero.

A wall with an interface has soil on either side with nodes of its own, so
the two sides are not connected and the wall is impermeable, as a
diaphragm or sheet pile wall is.  Until the wall is installed its two sides
are joined again.  A wall without an interface lets water through.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import scipy.sparse as sp

from .assembly import LinearSolver, solve_constrained
from .elements import TET10_FACES
from .water import HeadField, Seepage, _inside


def head_gradients(continuum) -> np.ndarray:
    """d N / d x at every Gauss point, (ne, ng, 3, 10), read out of the strain matrices."""
    B = continuum.B
    return np.stack([B[:, :, 0, 0::3], B[:, :, 1, 1::3], B[:, :, 2, 2::3]], axis=2)


def solve_seepage(problem, spec: Seepage, active: np.ndarray, installed=(), tolerance: float = 1e-5,
                  max_iterations: int = 60, backend: str = "auto", verbose: bool = False) -> HeadField:
    """Steady head over the active ground: a :class:`~lythos3d.core.water.HeadField`.

    ``installed`` are the indices of the plates built so far: an interface
    wall blocks the water only once it is there.
    """
    mesh, ce = problem.mesh, problem.continuum
    x = mesh.nodes
    n = mesh.n_nodes
    table = spec.table
    installed = set(installed)

    # nodes split for a wall not yet built are one node to the water
    rep = np.arange(n)
    for i, (orig, back) in problem.split_nodes.items():
        if i not in installed:
            rep[back] = orig
    act = np.nonzero(active)[0]
    E = rep[mesh.elements[act]]
    G = head_gradients(ce)[act]
    w = ce.detJw[act]
    zg = ce.gauss_xyz[act, :, 2]
    kdiag = np.zeros((len(act), 3))
    for r, mat in problem.materials.items():
        mine = mesh.region[act] == r
        kh = float(getattr(mat, "k", 1.0))
        kv = getattr(mat, "k_v", None)
        kdiag[mine] = (kh, kh, kh if kv is None else float(kv))
    if np.any(kdiag <= 0):
        raise ValueError("permeabilities must be positive")

    # ------------------------------------------------------------ boundaries
    faces = E[:, np.array(TET10_FACES)].reshape(-1, 6)
    owner = np.repeat(act, 4)
    key = np.sort(faces[:, :3], axis=1)
    _, inverse, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    free = counts[inverse.ravel()] == 1
    faces, key, owner = faces[free], key[free], owner[free]
    ie = problem.interface_elements
    if ie is not None and len(problem.interface_plate):
        walled = np.isin(problem.interface_plate, list(installed)) & active[problem.interface_support]
        if walled.any():
            wall_keys = {tuple(k) for k in np.sort(rep[ie.soil_faces[walled, :3]], axis=1).tolist()}
            keep = np.array([tuple(k) not in wall_keys for k in key.tolist()], dtype=bool)
            faces, owner = faces[keep], owner[keep]
    lo, hi = mesh.bounds
    eps = 1e-6 * float(max(hi - lo))
    corners = x[faces[:, :3]]

    def on(axis, value):
        return np.all(np.abs(corners[:, :, axis] - value) < eps, axis=1)

    base = on(2, lo[2])
    sides = {"xmin": on(0, lo[0]), "xmax": on(0, hi[0]), "ymin": on(1, lo[1]), "ymax": on(1, hi[1])}
    box = base.copy()
    open_side = np.zeros(len(faces), bool)
    for name, mask in sides.items():
        box |= mask
        if name not in spec.closed:
            open_side |= mask
    ground = ~box

    level = table.head(x)
    held = np.zeros(n, bool)
    value = np.zeros(n)
    open_nodes = np.unique(faces[open_side])
    side_nodes = open_nodes[x[open_nodes, 2] <= level[open_nodes] + eps]
    held[side_nodes] = True
    value[side_nodes] = level[side_nodes]
    # On the ground the level is read just inside the ground's own element:
    # a pit's edge is often the edge of its drawdown and the line of its
    # wall, and a node there belongs to the pit or to the ground outside by
    # the side it is on, not by a point-in-polygon test on the line itself.
    gf, go = faces[ground], np.repeat(owner[ground], 6)
    gn = gf.ravel()
    inward = x[gn] + 1e-6 * (x[mesh.elements[go, :4]].mean(axis=1) - x[gn])
    ground_level = replace(table, drawdowns=()).head(x)
    drawn = np.zeros(n, bool)
    for d in table.drawdowns:
        inside = np.zeros(n, bool)
        np.logical_or.at(inside, gn, _inside(d.polygon, inward))
        ground_level[inside] = np.minimum(ground_level[inside], d.level)
        drawn |= inside
    ground_nodes = np.setdiff1d(np.unique(gn), side_nodes)
    level[ground_nodes] = ground_level[ground_nodes]
    ponded = ground_nodes[x[ground_nodes, 2] <= level[ground_nodes] + eps]
    held[ponded] = True
    value[ponded] = level[ponded]
    # an open side above the water is a seepage face too: the downstream
    # face of a dam, or nothing at all where the ground is dry
    candidates = np.setdiff1d(np.union1d(ground_nodes, open_nodes), np.nonzero(held)[0])
    seeping = np.zeros(len(candidates), bool)         # closed to begin with; opened where water stands above them

    used = np.zeros(n, bool)
    used[E.ravel()] = True
    if not (held & used).any():
        raise ValueError("seepage: no boundary holds the head; is the water table below the model?")

    # ------------------------------------------------------------- iterate
    rows = np.repeat(E, 10, axis=1).ravel()
    cols = np.tile(E, (1, 10)).ravel()
    h = np.where(np.isfinite(level), np.maximum(level, x[:, 2]), x[:, 2])
    h[held] = value[held]
    gN = ce.N
    linear = LinearSolver(backend)
    scale = float(np.ptp(value[held & used])) + float(np.ptp(x[used, 2])) + 1e-12
    log_min = np.log10(spec.k_min)

    def evaluate(h, jacobian, psi_k):
        """Flux matrix K(h), its product with h, k_rel, and (on request) the Newton Jacobian."""
        psi = h[E] @ gN.T - zg
        # log10(k_rel) = log10(k_min) S(-psi / psi_k), with S the smooth step
        # 3t^2 - 2t^3: no kink at either end for Newton to trip on
        t = np.clip(-psi / psi_k, 0.0, 1.0)
        krel = 10.0 ** (log_min * t * t * (3.0 - 2.0 * t))
        Ke = np.einsum("egia,eg,ei,egib->eab", G, w * krel, kdiag, G, optimize=True)
        K = sp.coo_matrix((Ke.ravel(), (rows, cols)), shape=(n, n)).tocsr()
        J = None
        if jacobian:
            # d(K h)/dh adds the change of permeability with the pressure head
            dk = krel * np.log(10.0) * log_min * 6.0 * t * (1.0 - t) * (-1.0 / psi_k)
            flux = np.einsum("ei,egia,ea->egi", kdiag, G, h[E])
            v = np.einsum("egia,egi->ega", G, flux) * (w * dk)[:, :, None]
            Je = Ke + np.einsum("ega,gb->eab", v, gN)
            J = sp.coo_matrix((Je.ravel(), (rows, cols)), shape=(n, n)).tocsr()
        return K @ h, krel, J

    def fixed_mask(seeping):
        fix = held.copy()
        fix[candidates[seeping]] = True
        return fix | ~used

    prescribed = np.where(held, value, x[:, 2])
    prescribed[~used] = 0.0
    count = [0]
    size = float(max(hi - lo))
    nominal = 1e-3 * float(kdiag.max()) * scale * size

    def settle(psi_k, h, seeping):
        """Seepage faces (outer loop) and Newton for the head with the faces as they stand (inner).

        The faces are judged only on a converged head; judged mid-iteration
        they flip back and forth for ever.  Returns ``(converged, h, seeping)``.
        """
        for _ in range(max_iterations):
            fix = fixed_mask(seeping)
            h[fix] = prescribed[fix]
            idx = np.nonzero(fix)[0]
            settled = False
            for _ in range(max_iterations):
                count[0] += 1
                inflow, _, J = evaluate(h, True, psi_k)
                # the largest imbalance at a node, against all the water passing through
                # (or against a nominal flow, where little or none passes)
                flow = max(0.5 * float(np.abs(inflow[fix & used]).sum()), nominal)
                residual = float(np.abs(inflow[~fix & used]).max(initial=0.0)) / flow
                if verbose:
                    print(f"    seepage {count[0]} (psi_k {psi_k:.3g}): residual {residual:.2e}, "
                          f"{int(seeping.sum())} seeping", flush=True)
                if residual < tolerance:
                    settled = True
                    break
                r = -inflow
                r[fix] = 0.0
                dh = solve_constrained(J, r, idx, np.zeros(len(idx)), solver=linear)
                linear.forget()
                r0 = float(np.linalg.norm(r))
                for alpha in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125):
                    step = h + alpha * dh
                    rt = evaluate(step, False, psi_k)[0]
                    rt[fix] = 0.0
                    if np.linalg.norm(rt) < r0:
                        break
                h = step
            if not settled:
                return False, h, seeping
            # water must leave where the head is held, and may not stand
            # above the ground where it is not
            release = seeping & (inflow[candidates] > 1e-8 * flow)
            capture = ~seeping & (h[candidates] > x[candidates, 2] + 1e-6 * scale)
            if verbose and (release.any() or capture.any()):
                print(f"    seepage faces: {int(release.sum())} released, {int(capture.sum())} captured",
                      flush=True)
            if not release.any() and not capture.any():
                return True, h, seeping
            seeping = (seeping & ~release) | capture
        return False, h, seeping

    # Continuation in the width of the unsaturated transition: a sharp one
    # is solved from the solution of a gentler one.  The width is halved
    # after each success; a step that fails is retried at a smaller ratio.
    width = spec.psi_k
    while width < 0.25 * scale:
        width *= 2.0
    converged, h, seeping = settle(width, h, seeping)
    ratio = 2.0
    while converged and width > spec.psi_k * (1.0 + 1e-9):
        trial = max(spec.psi_k, width / ratio)
        ok, h_new, seeping_new = settle(trial, h.copy(), seeping.copy())
        if ok:
            width, h, seeping = trial, h_new, seeping_new
            ratio = min(2.0, ratio * 1.5)
        elif ratio > 1.05:
            ratio = np.sqrt(ratio)
        else:
            converged = False
    iteration = count[0]
    inflow, krel, _ = evaluate(h, False, width)
    linear.release()
    if not converged:
        raise RuntimeError("the seepage iteration did not settle; try a larger psi_k or a finer mesh near the water table")

    head = h.copy()
    head[rep != np.arange(n)] = h[rep[rep != np.arange(n)]]
    unused = ~used
    unused[rep != np.arange(n)] = ~used[rep[rep != np.arange(n)]]
    head[unused] = np.nan

    field = HeadField(head, x, mesh.elements, gN, ce.gauss_xyz.reshape(-1, 3), active, spec.gamma_w)
    # Darcy velocity at the Gauss points
    velocity = np.zeros((mesh.n_elements, ce.n_gauss, 3))
    gh = np.einsum("egia,ea->egi", G, h[E])
    velocity[act] = -(kdiag[:, None, :] * krel[:, :, None]) * gh
    field.velocity = velocity.reshape(-1, 3)
    # Flows through each kind of boundary, net.  (Node by node the signs
    # mean little: a quadratic face's corner nodes can carry a small flux
    # against the flow through it.)
    seep_nodes = np.zeros(n, bool)
    seep_nodes[candidates[seeping]] = True
    side = np.zeros(n, bool)
    side[side_nodes] = True
    q = np.where(used, inflow, 0.0)
    pumped = -float(q[drawn & (held | seep_nodes) & ~side].sum())
    net = {"ground": float(q[held & ~side & ~drawn].sum()),
           "pumped": -pumped,
           "seepage_face": float(q[seep_nodes & ~drawn & ~side].sum())}
    taken = np.zeros(n, bool)
    for name, mask in sides.items():
        mine = np.zeros(n, bool)
        mine[np.unique(faces[mask & open_side])] = True
        mine &= side & ~taken
        taken |= mine
        net[name] = float(q[mine].sum())
    field.flows = {
        "in": sum(v for v in net.values() if v > 0),
        "out": -sum(v for v in net.values() if v < 0),
        # what the pumps must lift: water leaving through the ground inside drawdowns
        "pumped": pumped,
        "seepage_face": -net["seepage_face"],
        # net inflow through each open side of the box, and through ponded ground
        "sides": {k: net[k] for k in sides if k not in spec.closed},
        "ground": net["ground"],
        "iterations": iteration,
    }
    return field

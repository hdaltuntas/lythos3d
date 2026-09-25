"""The 6-node shell against closed-form plate and beam solutions."""

import numpy as np
import pytest
import scipy.sparse as sp

from lythos3d.core.assembly import solve_constrained
from lythos3d.core.structures import PlateElements, PlateSection


def plate_mesh(a: float, b: float, nx: int, ny: int, frame=np.eye(3), origin=(0, 0, 0)):
    """A rectangle a x b of 6-node triangles in the plane spanned by frame rows 0 and 1."""
    xs, ys = np.linspace(0, a, nx + 1), np.linspace(0, b, ny + 1)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    corners = np.column_stack([X.ravel(), Y.ravel()])
    idx = lambda i, j: i * (ny + 1) + j                                      # noqa: E731
    tris = []
    for i in range(nx):
        for j in range(ny):
            p, q, r, s = idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)
            tris += [(p, q, r), (p, r, s)] if (i + j) % 2 == 0 else [(p, q, s), (q, r, s)]
    tris = np.array(tris)
    edges = np.sort(tris[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
    unique, inverse = np.unique(edges, axis=0, return_inverse=True)
    mids = 0.5 * (corners[unique[:, 0]] + corners[unique[:, 1]])
    plane = np.vstack([corners, mids])
    faces = np.column_stack([tris, len(corners) + inverse.reshape(-1, 3)])
    nodes = np.asarray(origin, float) + plane[:, :1] * frame[0] + plane[:, 1:2] * frame[1]
    return nodes, faces, plane


def assemble(elements: PlateElements, n_nodes: int):
    dofs = (6 * elements.faces[:, :, None] + np.arange(6)).reshape(len(elements.faces), 36)
    K = elements.stiffness()
    rows = np.repeat(dofs, 36, axis=1).ravel()
    cols = np.tile(dofs, (1, 36)).ravel()
    return sp.coo_matrix((K.ravel(), (rows, cols)), shape=(6 * n_nodes,) * 2).tocsr(), dofs


def test_single_element_has_six_rigid_body_modes():
    nodes, faces, _ = plate_mesh(1.0, 1.0, 1, 1)
    tilt = np.linalg.qr(np.random.default_rng(0).normal(size=(3, 3)))[0]
    el = PlateElements(nodes @ tilt.T, faces[:1], PlateSection(E=3e7, nu=0.2, t=0.3))
    K = el.stiffness()[0]
    assert np.abs(K - K.T).max() < 1e-9 * np.abs(K).max()
    eig = np.linalg.eigvalsh(K)
    assert np.sum(eig < 1e-9 * eig.max()) == 6
    # a rigid rotation about any axis, with the node rotations that go with it
    x = (nodes @ tilt.T)[faces[0]]
    for omega in np.eye(3):
        u = np.zeros(36)
        for n in range(6):
            u[6 * n:6 * n + 3] = np.cross(omega, x[n])
            u[6 * n + 3:6 * n + 6] = omega
        assert np.abs(K @ u).max() < 1e-8 * np.abs(K).max()


def test_membrane_patch_test_on_a_tilted_plate():
    """A linear in-plane displacement field gives constant membrane forces."""
    frame = np.linalg.qr(np.random.default_rng(1).normal(size=(3, 3)))[0].T
    nodes, faces, plane = plate_mesh(2.0, 1.0, 4, 3, frame=frame, origin=(1, 2, 3))
    section = PlateSection(E=2e5, nu=0.25, t=0.1)
    el = PlateElements(nodes, faces, section)
    K, _ = assemble(el, len(nodes))
    grad = np.array([[1e-3, 4e-4], [-2e-4, 5e-4]])           # d(u, v)/d(x, y) in plane axes
    u_plane = plane @ grad.T
    exact = np.zeros((len(nodes), 6))
    exact[:, :3] = u_plane[:, :1] * frame[0] + u_plane[:, 1:] * frame[1]
    exact[:, 3:] = 0.5 * (grad[1, 0] - grad[0, 1]) * frame[2]   # drilling rotation of the field
    boundary = np.nonzero(np.any(np.isclose(plane, 0) | np.isclose(plane, [2.0, 1.0]), axis=1))[0]
    fixed = (6 * boundary[:, None] + np.arange(6)).ravel()
    free_rot = np.setdiff1d(np.arange(6 * len(nodes)), fixed)
    u = solve_constrained(K, np.zeros(6 * len(nodes)), fixed, exact.ravel()[fixed], backend="superlu")
    assert np.abs(u - exact.ravel())[free_rot].max() < 1e-10
    N, M, _ = el.resultants(u[(6 * faces[:, :, None] + np.arange(6)).reshape(len(faces), 36)])
    # the element axes differ from element to element; compare invariants
    trace = N[..., 0] + N[..., 1]
    eps = np.array([[grad[0, 0], 0.5 * (grad[0, 1] + grad[1, 0])], [0.5 * (grad[0, 1] + grad[1, 0]), grad[1, 1]]])
    expected_trace = section.E * section.t / (1 - section.nu) * np.trace(eps)
    assert np.allclose(trace, expected_trace, rtol=1e-9)
    assert np.abs(M).max() < 1e-9 * np.abs(N).max()


def _cantilever(t: float, n_along: int = 10, n_across: int = 2):
    L, b, P = 10.0, 1.0, 1.0
    section = PlateSection(E=1e6, nu=0.0, t=t)
    nodes, faces, plane = plate_mesh(L, b, n_along, n_across)
    el = PlateElements(nodes, faces, section)
    K, _ = assemble(el, len(nodes))
    f = np.zeros(6 * len(nodes))
    tip = np.nonzero(np.isclose(plane[:, 0], L))[0]
    # a consistent line load P / b along the free edge: 1/6, 4/6, 1/6 per quadratic edge
    y = plane[tip, 1]
    order = np.argsort(y)
    weights = np.zeros(len(tip))
    for k in range(0, len(tip) - 1, 2):
        seg = order[k:k + 3]
        weights[seg] += (y[seg[2]] - y[seg[0]]) * np.array([1, 4, 1]) / 6
    f[6 * tip + 2] = -P / b * weights
    clamped = np.nonzero(np.isclose(plane[:, 0], 0))[0]
    fixed = (6 * clamped[:, None] + np.arange(6)).ravel()
    u = solve_constrained(K, f, fixed, backend="superlu")
    deflection = -u[6 * tip + 2].mean()
    inertia, G = b * t ** 3 / 12, section.G
    return deflection, P * L ** 3 / (3 * section.E * inertia) + P * L / (5 / 6 * G * b * t)


@pytest.mark.parametrize("t", [1.0, 0.1, 0.01])
def test_cantilever_strip_bends_as_a_beam(t):
    """Thick to very thin (L/t = 10 to 1000): no shear locking."""
    deflection, expected = _cantilever(t)
    assert deflection == pytest.approx(expected, rel=0.02)


def test_simply_supported_square_plate_under_pressure():
    """Navier's solution for a thin plate: w = 0.00406 q a^4 / D at the centre (nu = 0.3)."""
    a, q = 10.0, 10.0
    section = PlateSection(E=1e7, nu=0.3, t=0.05)
    nodes, faces, plane = plate_mesh(a, a, 12, 12)
    el = PlateElements(nodes, faces, section)
    K, dofs = assemble(el, len(nodes))
    f = np.zeros(6 * len(nodes))
    weights = sum(w * np.array([L[0] * (2 * L[0] - 1), L[1] * (2 * L[1] - 1), L[2] * (2 * L[2] - 1),
                                4 * L[0] * L[1], 4 * L[1] * L[2], 4 * L[2] * L[0]])
                  for L, w in zip([[2 / 3, 1 / 6, 1 / 6], [1 / 6, 2 / 3, 1 / 6], [1 / 6, 1 / 6, 2 / 3]], [1 / 3] * 3))
    np.add.at(f, dofs[:, 2::6].ravel(), (-q * el.area[:, None] * weights).ravel())
    edge = np.nonzero(np.isclose(plane, 0).any(axis=1) | np.isclose(plane, a).any(axis=1))[0]
    fixed = [6 * edge + 2]                                    # w = 0 on all edges (soft simple support)
    fixed.append(6 * np.arange(len(nodes)) + 0)               # no membrane action in a flat plate
    fixed.append(6 * np.arange(len(nodes)) + 1)
    fixed.append(6 * np.arange(len(nodes)) + 5)
    fixed = np.unique(np.concatenate(fixed))
    u = solve_constrained(K, f, fixed, backend="superlu")
    centre = np.nonzero(np.all(np.isclose(plane, a / 2), axis=1))[0][0]
    D = section.E * section.t ** 3 / (12 * (1 - section.nu ** 2))
    assert -u[6 * centre + 2] == pytest.approx(0.00406 * q * a ** 4 / D, rel=0.02)


def test_coarse_thin_plate_does_not_lock():
    """A 4 x 4 mesh of a plate with t/a = 0.005: elements 70 times their thickness.

    Without the shear stabilisation this comes out 19% too stiff.
    """
    a, q = 10.0, 10.0
    section = PlateSection(E=1e7, nu=0.3, t=0.05)
    nodes, faces, plane = plate_mesh(a, a, 4, 4)
    el = PlateElements(nodes, faces, section)
    K, dofs = assemble(el, len(nodes))
    f = np.zeros(6 * len(nodes))
    np.add.at(f, dofs[:, 2::6].ravel(), (-q * el.area[:, None] * np.array([0, 0, 0, 1, 1, 1]) / 3).ravel())
    edge = np.nonzero(np.isclose(plane, 0).any(axis=1) | np.isclose(plane, a).any(axis=1))[0]
    all_nodes = np.arange(len(nodes))
    fixed = np.unique(np.concatenate([6 * edge + 2, 6 * all_nodes, 6 * all_nodes + 1, 6 * all_nodes + 5]))
    u = solve_constrained(K, f, fixed, backend="superlu")
    centre = np.nonzero(np.all(np.isclose(plane, a / 2), axis=1))[0][0]
    D = section.E * section.t ** 3 / (12 * (1 - section.nu ** 2))
    assert -u[6 * centre + 2] == pytest.approx(0.00406 * q * a ** 4 / D, rel=0.05)


def test_section_from_wall_stiffness():
    s = PlateSection.from_stiffness(EA=2.4e7, EI=1.28e6)
    assert s.EA == pytest.approx(2.4e7) and s.EI == pytest.approx(1.28e6)

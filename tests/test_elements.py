"""10-node tetrahedron: shape functions, integration, rigid motion, patch test."""

import numpy as np
import pytest

from lythos3d.core.assembly import SparsityPattern, solve_constrained
from lythos3d.core.elements import (
    TET10_EDGES, TET10_FACES, TET_GAUSS_BARY, ContinuumElements, elastic_matrix,
    face_normals, tet10_shape,
)
from lythos3d.core.mesh import Mesh, box_mesh, graded

REFERENCE = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)


def _reference_tet10():
    mids = [0.5 * (REFERENCE[i] + REFERENCE[j]) for i, j in TET10_EDGES]
    return np.vstack([REFERENCE, mids])


def _bary(x):
    return np.array([1.0 - x.sum(), *x])


def test_shape_functions_interpolate_the_nodes_and_sum_to_one():
    nodes = _reference_tet10()
    for k, x in enumerate(nodes):
        N, dN = tet10_shape(_bary(x))
        assert np.allclose(N, np.eye(10)[k], atol=1e-14)
    for L in TET_GAUSS_BARY:
        N, dN = tet10_shape(L)
        assert N.sum() == pytest.approx(1.0)
        assert np.allclose(dN.sum(axis=1), 0.0, atol=1e-14)


def test_shape_function_derivatives_match_finite_differences():
    x = np.array([0.21, 0.17, 0.33])
    _, dN = tet10_shape(_bary(x))
    h = 1e-6
    for k in range(3):
        step = np.zeros(3)
        step[k] = h
        fd = (tet10_shape(_bary(x + step))[0] - tet10_shape(_bary(x - step))[0]) / (2 * h)
        assert np.allclose(dN[k], fd, atol=1e-8)


def test_faces_point_outwards():
    nodes = _reference_tet10()
    centre = REFERENCE.mean(axis=0)
    faces = np.array(TET10_FACES)
    normals = face_normals(nodes, faces)
    for face, n in zip(faces, normals):
        assert np.dot(nodes[face[:3]].mean(axis=0) - centre, n) > 0
        # the mid-edge nodes sit between the corners they are listed after
        a, b, c = face[:3]
        for mid, (p, q) in zip(face[3:], ((a, b), (b, c), (c, a))):
            assert np.allclose(nodes[mid], 0.5 * (nodes[p] + nodes[q]))


def test_integration_recovers_volume_of_a_distorted_element():
    corners = np.array([[0, 0, 0], [2.0, 0.1, 0], [0.3, 1.5, 0.2], [0.1, 0.4, 3.0]])
    mids = [0.5 * (corners[i] + corners[j]) for i, j in TET10_EDGES]
    ce = ContinuumElements(np.vstack([corners, mids]), np.arange(10)[None])
    exact = np.linalg.det(corners[1:] - corners[0]) / 6.0
    assert ce.volumes()[0] == pytest.approx(exact, rel=1e-13)


def test_inverted_element_is_rejected():
    nodes = _reference_tet10()
    elem = np.array([[0, 2, 1, 3, 5, 4, 6, 7, 9, 8]])
    with pytest.raises(ValueError, match="non-positive Jacobian"):
        ContinuumElements(nodes, elem)


def test_stiffness_is_symmetric_with_exactly_six_rigid_body_modes():
    ce = ContinuumElements(_reference_tet10(), np.arange(10)[None])
    K = ce.stiffness(elastic_matrix(1.0e4, 0.3)[None])[0]
    assert np.abs(K - K.T).max() < 1e-10 * np.abs(K).max()
    eig = np.linalg.eigvalsh(K)
    assert np.sum(eig < 1e-8 * eig.max()) == 6

    x = _reference_tet10()
    rotation = np.cross([0.3, -0.2, 0.7], x).ravel()
    for motion in (np.tile([1.0, 0, 0], 10), np.tile([0, 0, 1.0], 10), rotation):
        assert np.abs(K @ motion).max() < 1e-9 * np.abs(K).max()


def _distorted_block():
    """A 3x3x3 block with its interior corner nodes moved at random."""
    mesh = box_mesh(*(graded(0, 1, 1 / 3) for _ in range(3)))
    corners = np.unique(mesh.elements[:, :4])
    x = mesh.nodes[corners]
    interior = np.all((x > 1e-9) & (x < 1 - 1e-9), axis=1)
    rng = np.random.default_rng(1)
    mesh.nodes[corners[interior]] += rng.uniform(-0.08, 0.08, (interior.sum(), 3))
    for k, (i, j) in enumerate(TET10_EDGES):
        mesh.nodes[mesh.elements[:, 4 + k]] = 0.5 * (mesh.nodes[mesh.elements[:, i]]
                                                     + mesh.nodes[mesh.elements[:, j]])
    return mesh


def test_patch_test_reproduces_a_linear_displacement_field_exactly():
    """Prescribing a linear field on the boundary must give it everywhere.

    An element that fails this cannot converge to the right answer however
    fine the mesh.  The interior nodes are displaced at random so that the
    elements are not all the same shape.
    """
    mesh = _distorted_block()
    ce = ContinuumElements(mesh.nodes, mesh.elements)
    D = elastic_matrix(3.0e4, 0.2)
    grad = np.array([[1e-3, -4e-4, 2e-4], [7e-4, 2e-3, -1e-4], [3e-4, 5e-4, -1.5e-3]])
    exact = (mesh.nodes @ grad.T).ravel()

    K = SparsityPattern(mesh.elements, mesh.n_nodes).assemble(ce.stiffness(D[None].repeat(ce.n_elements, 0)))
    x = mesh.nodes
    on_boundary = np.any((np.abs(x) < 1e-9) | (np.abs(x - 1) < 1e-9), axis=1)
    fixed = (3 * np.nonzero(on_boundary)[0][:, None] + np.arange(3)).ravel()
    u = solve_constrained(K, np.zeros(3 * mesh.n_nodes), fixed, exact[fixed], symmetric=True)

    assert np.abs(u - exact).max() < 1e-12
    eps = ce.strains(u)
    expected = [grad[0, 0], grad[1, 1], grad[2, 2],
                grad[0, 1] + grad[1, 0], grad[1, 2] + grad[2, 1], grad[2, 0] + grad[0, 2]]
    assert np.allclose(eps, expected, atol=1e-12)


def test_nodal_average_is_exact_for_a_linear_field():
    mesh = _distorted_block()
    ce = ContinuumElements(mesh.nodes, mesh.elements)

    def field(p):
        return 3.0 + p @ np.array([1.0, -2.0, 0.5])

    at_gauss = field(ce.gauss_xyz.reshape(-1, 3))[:, None]
    assert np.allclose(ce.nodal_average(at_gauss, mesh.n_nodes)[:, 0], field(mesh.nodes))


def test_body_force_totals_the_weight():
    mesh = _distorted_block()
    ce = ContinuumElements(mesh.nodes, mesh.elements)
    f = ce.body_force(np.full(ce.n_elements, 18.0))
    assert f[:, 2::3].sum() == pytest.approx(-18.0 * 1.0, rel=1e-12)
    assert np.abs(f[:, 0::3]).max() == 0 and np.abs(f[:, 1::3]).max() == 0


def test_mesh_rejects_nothing_it_should_accept():
    mesh = Mesh(_reference_tet10(), np.arange(10)[None])
    assert mesh.n_elements == 1 and mesh.region.tolist() == [0]

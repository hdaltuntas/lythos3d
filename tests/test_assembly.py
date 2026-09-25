"""Sparse assembly and the linear solver backends."""

import numpy as np
import pytest
import scipy.sparse as sp

from lythos3d.core import assembly
from lythos3d.core.assembly import SparsityPattern, solve, solve_constrained
from lythos3d.core.elements import ContinuumElements, elastic_matrix
from lythos3d.core.mesh import box_mesh, graded

needs_pardiso = pytest.mark.skipif(assembly.pypardiso is None, reason="pypardiso not installed")


def _mesh():
    return box_mesh(graded(0, 3, 1), graded(0, 2, 1), graded(-2, 0, 1))


def test_pattern_assembly_matches_triplet_assembly():
    mesh = _mesh()
    ce = ContinuumElements(mesh.nodes, mesh.elements)
    Ke = np.random.default_rng(0).random((mesh.n_elements, 30, 30))
    d = ce.dofs()
    n = 3 * mesh.n_nodes
    ref = sp.coo_matrix((Ke.ravel(), (np.repeat(d, 30, axis=1).ravel(), np.tile(d, (1, 30)).ravel())),
                        shape=(n, n)).toarray()
    pattern = SparsityPattern(mesh.elements, mesh.n_nodes)
    K = pattern.assemble(Ke)
    K.check_format(full_check=True)
    assert np.abs(K.toarray() - ref).max() == 0.0

    active = np.arange(mesh.n_elements) % 3 != 0
    ref_active = sp.coo_matrix((Ke[active].ravel(), (np.repeat(d[active], 30, axis=1).ravel(),
                                                     np.tile(d[active], (1, 30)).ravel())),
                               shape=(n, n)).toarray()
    assert np.abs(pattern.assemble(Ke, active).toarray() - ref_active).max() == 0.0


def _elastic_system():
    mesh = _mesh()
    ce = ContinuumElements(mesh.nodes, mesh.elements)
    K = SparsityPattern(mesh.elements, mesh.n_nodes).assemble(
        ce.stiffness(elastic_matrix(1e4, 0.3)[None].repeat(ce.n_elements, 0)))
    base = np.nonzero(mesh.nodes[:, 2] < -2 + 1e-9)[0]
    fixed = (3 * base[:, None] + np.arange(3)).ravel()
    f = np.random.default_rng(2).standard_normal(K.shape[0])
    return K, f, fixed


def test_constrained_solution_satisfies_the_free_equations():
    K, f, fixed = _elastic_system()
    prescribed = np.linspace(-1e-3, 1e-3, len(fixed))
    u = solve_constrained(K, f, fixed, prescribed, backend="superlu")
    free = np.ones(K.shape[0], bool)
    free[fixed] = False
    assert np.allclose(u[fixed], prescribed)
    r = K @ u - f
    assert np.abs(r[free]).max() < 1e-8 * np.abs(f).max()


@needs_pardiso
@pytest.mark.parametrize("symmetric", [False, True])
def test_pardiso_agrees_with_superlu(symmetric):
    K, f, fixed = _elastic_system()
    a = solve_constrained(K, f, fixed, backend="superlu")
    b = solve_constrained(K, f, fixed, backend="pardiso", symmetric=symmetric)
    assert np.abs(a - b).max() < 1e-9 * np.abs(a).max()


@needs_pardiso
def test_pardiso_solves_an_unsymmetric_system():
    rng = np.random.default_rng(3)
    A = sp.random(200, 200, density=0.05, random_state=4, format="csr") + 10 * sp.eye(200)
    b = rng.standard_normal(200)
    x = solve(A, b, backend="pardiso")
    assert np.abs(A @ x - b).max() < 1e-10


def test_unknown_backend_is_refused():
    with pytest.raises(ValueError, match="unknown solver backend"):
        solve(sp.eye(3, format="csr"), np.ones(3), backend="umfpack")


def test_unrestrained_model_is_reported_as_singular():
    K, f, _ = _elastic_system()
    for backend in assembly.available_backends():
        with pytest.raises(np.linalg.LinAlgError, match="rigid body"), np.errstate(all="ignore"):
            solve_constrained(K, f, np.array([], dtype=int), backend=backend)

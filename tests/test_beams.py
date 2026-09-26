# SPDX-License-Identifier: AGPL-3.0-only
"""The 3D Timoshenko beam against closed-form solutions."""

import numpy as np
import pytest
import scipy.sparse as sp

from lythos3d.core.assembly import solve_constrained
from lythos3d.core.beams import BeamElements, BeamSection


def _line(start, end, n):
    start, end = np.asarray(start, float), np.asarray(end, float)
    t = np.linspace(0, 1, 2 * n + 1)
    nodes = start + t[:, None] * (end - start)
    elements = np.array([[2 * k, 2 * k + 1, 2 * k + 2] for k in range(n)])
    return nodes, elements


def _assemble(beam: BeamElements, n_nodes: int):
    dofs = (6 * beam.elements[:, :, None] + np.arange(6)).reshape(beam.n_elements, 18)
    K = sp.coo_matrix((beam.K.ravel(), (np.repeat(dofs, 18, axis=1).ravel(), np.tile(dofs, (1, 18)).ravel())),
                      shape=(6 * n_nodes,) * 2).tocsr()
    return K, dofs


def test_element_has_exactly_six_rigid_body_modes():
    nodes, elements = _line((1, 2, 3), (2.5, 3.1, 0.4), 1)
    beam = BeamElements(nodes, elements, BeamSection.circular(E=3e7, D=0.8))
    K = beam.K[0]
    assert np.abs(K - K.T).max() < 1e-9 * np.abs(K).max()
    eig = np.linalg.eigvalsh(K)
    assert np.sum(eig < 1e-9 * eig.max()) == 6
    for omega in np.eye(3):                                   # rigid rotations with their node rotations
        u = np.zeros(18)
        for a in range(3):
            u[6 * a:6 * a + 3] = np.cross(omega, nodes[a])
            u[6 * a + 3:6 * a + 6] = omega
        assert np.abs(K @ u).max() < 1e-8 * np.abs(K).max()


@pytest.mark.parametrize("direction", [(0, 0, -1), (1, 1, 0), (0.3, -0.5, 0.8)])
def test_cantilever_tip_load(direction):
    """Tip deflection PL^3/3EI + PL/GA in any orientation, thick or slender."""
    L, P = 8.0, 10.0
    axis = np.asarray(direction, float) / np.linalg.norm(direction)
    section = BeamSection.circular(E=3e7, D=0.3)
    nodes, elements = _line((0, 0, 0), L * axis, 6)
    beam = BeamElements(nodes, elements, section)
    K, _ = _assemble(beam, len(nodes))
    across = np.cross(axis, [0.0, 1.0, 0.0] if abs(axis[1]) < 0.9 else [1.0, 0.0, 0.0])
    across /= np.linalg.norm(across)
    f = np.zeros(6 * len(nodes))
    f[6 * (len(nodes) - 1):6 * (len(nodes) - 1) + 3] = P * across
    u = solve_constrained(K, f, np.arange(6), backend="superlu")
    tip = u[6 * (len(nodes) - 1):6 * (len(nodes) - 1) + 3] @ across
    expected = P * L ** 3 / (3 * section.EI2) + P * L / section.GA2
    assert tip == pytest.approx(expected, rel=1e-6)


def test_axial_and_torsion():
    L, P, T = 5.0, 100.0, 20.0
    section = BeamSection.circular(E=2e8, D=0.5, hollow=0.45)
    nodes, elements = _line((0, 0, 0), (0, 0, -L), 3)
    beam = BeamElements(nodes, elements, section)
    K, _ = _assemble(beam, len(nodes))
    f = np.zeros(6 * len(nodes))
    tip = 6 * (len(nodes) - 1)
    f[tip + 2] = -P                                     # pull along the axis (which points down)
    f[tip + 5] = T                                      # twist about z
    u = solve_constrained(K, f, np.arange(6), backend="superlu")
    assert u[tip + 2] == pytest.approx(-P * L / section.EA, rel=1e-9)
    assert u[tip + 5] == pytest.approx(T * L / section.GJ, rel=1e-9)
    res = beam.resultants(u[(6 * elements[:, :, None] + np.arange(6)).reshape(len(elements), 18)])
    assert np.allclose(res[..., 0], P)                  # tension along the beam
    assert np.allclose(np.abs(res[..., 3]), T)


def test_slender_beam_does_not_lock():
    L, P = 10.0, 1.0
    section = BeamSection(EA=1e9, EI2=1e3, EI3=1e3, GJ=1e3, GA2=1e10, GA3=1e10)
    nodes, elements = _line((0, 0, 0), (L, 0, 0), 2)
    beam = BeamElements(nodes, elements, section)
    K, _ = _assemble(beam, len(nodes))
    f = np.zeros(6 * len(nodes))
    f[6 * (len(nodes) - 1) + 2] = P
    u = solve_constrained(K, f, np.arange(6), backend="superlu")
    assert u[6 * (len(nodes) - 1) + 2] == pytest.approx(P * L ** 3 / (3 * 1e3), rel=1e-6)


def test_self_weight_totals_the_weight():
    nodes, elements = _line((0, 0, 0), (3, 4, 0), 4)
    beam = BeamElements(nodes, elements, BeamSection.circular(E=3e7, D=0.6, weight=7.0))
    assert beam.self_weight()[:, 2::6].sum() == pytest.approx(-7.0 * 5.0)

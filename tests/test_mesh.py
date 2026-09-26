# SPDX-License-Identifier: AGPL-3.0-only
"""Structured block meshes."""

import numpy as np
import pytest

from lythos3d.core.elements import ContinuumElements, face_normals
from lythos3d.core.mesh import box_mesh, graded


def test_graded_keeps_break_lines_and_respects_the_size():
    z = graded(-10, 0, 1.5, breaks=[-3.2, -7.0, 5.0])
    assert z[0] == -10 and z[-1] == 0
    assert np.any(np.isclose(z, -3.2)) and np.any(np.isclose(z, -7.0))
    assert np.diff(z).max() <= 1.5 + 1e-12


def test_box_mesh_fills_the_block_exactly_and_is_conforming():
    xs, ys, zs = graded(0, 4, 1), graded(0, 3, 1.5), graded(-5, 0, 1, breaks=[-2.5])
    mesh = box_mesh(xs, ys, zs)
    assert mesh.n_elements == 6 * (len(xs) - 1) * (len(ys) - 1) * (len(zs) - 1)
    ce = ContinuumElements(mesh.nodes, mesh.elements)       # would raise on an inverted element
    assert ce.volumes().sum() == pytest.approx(4 * 3 * 5)

    # a conforming mesh has no internal face on its boundary: the boundary
    # faces cover exactly the surface of the block
    faces = mesh.boundary_faces()
    p = mesh.nodes[faces[:, :3]]
    area = 0.5 * np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1)
    assert area.sum() == pytest.approx(2 * (4 * 3 + 4 * 5 + 3 * 5))
    # and every boundary face points out of the block
    centre = np.array([2.0, 1.5, -2.5])
    assert np.all(np.einsum("ij,ij->i", face_normals(mesh.nodes, faces), p.mean(axis=1) - centre) > 0)


def test_mid_edge_nodes_are_shared_between_elements():
    mesh = box_mesh(graded(0, 2, 1), graded(0, 2, 1), graded(0, 2, 1))
    n_corners = 27
    # every node is used; mid-edge nodes were not duplicated per element
    assert len(np.unique(mesh.elements)) == mesh.n_nodes
    edges = mesh.n_nodes - n_corners
    # Euler for a triangulated ball: V - E + F - T = 1
    faces = mesh.elements[:, [[0, 1, 2], [0, 1, 3], [1, 2, 3], [0, 2, 3]]].reshape(-1, 3)
    n_faces = len(np.unique(np.sort(faces, axis=1), axis=0))
    assert n_corners - edges + n_faces - mesh.n_elements == 1


def test_faces_on_plane_and_regions():
    mesh = box_mesh(graded(0, 2, 1), graded(0, 2, 1), graded(-4, 0, 1, breaks=[-1.5]))
    assert len(mesh.faces_on_plane(2, 0.0)) == 2 * 2 * 2
    mesh.assign_regions(lambda c: np.where(c[:, 2] > -1.5, 0, 1))
    ce = ContinuumElements(mesh.nodes, mesh.elements)
    v = ce.volumes()
    assert v[mesh.region == 0].sum() == pytest.approx(2 * 2 * 1.5)
    assert v[mesh.region == 1].sum() == pytest.approx(2 * 2 * 2.5)

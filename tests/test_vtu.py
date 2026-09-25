"""ParaView output."""

import xml.etree.ElementTree as ET

import numpy as np

from lythos3d.core.analysis import linear_static
from lythos3d.core.materials import LinearElastic
from lythos3d.core.mesh import box_mesh, graded
from lythos3d.io.vtu import VTK_QUADRATIC_TETRA, read_vtu_array, write_result


def test_result_round_trips_through_vtu(tmp_path):
    mesh = box_mesh(graded(0, 2, 1), graded(0, 2, 1), graded(-2, 0, 1))
    r = linear_static(mesh, [LinearElastic("s", 1e4, 0.3, 20.0)])
    path = write_result(tmp_path / "out.vtu", r)
    text = open(path).read()

    piece = ET.fromstring(text).find("UnstructuredGrid/Piece")
    assert int(piece.get("NumberOfPoints")) == mesh.n_nodes
    assert int(piece.get("NumberOfCells")) == mesh.n_elements
    assert np.array_equal(read_vtu_array(text, "Points"), mesh.nodes)
    assert np.array_equal(read_vtu_array(text, "connectivity").reshape(-1, 10), mesh.elements)
    assert np.all(read_vtu_array(text, "types") == VTK_QUADRATIC_TETRA)
    assert np.array_equal(read_vtu_array(text, "displacement"), r.displacement)
    assert np.array_equal(read_vtu_array(text, "stress"), r.nodal_stress)

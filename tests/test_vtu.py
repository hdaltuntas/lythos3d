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


def test_stage_output_leaves_excavated_ground_out(tmp_path):
    from lythos3d.core.model import Model, Stratum, Volume
    from lythos3d.core.problem import Stage
    from lythos3d.io.vtu import write_stage

    fill = LinearElastic("fill", 2e4, 0.3, 18.0)
    clay = LinearElastic("clay", 1e4, 0.3, 20.0)
    model = Model("dig", (0, 4), (0, 3), -10, [Stratum("fill", fill, 0), Stratum("clay", clay, -2)],
                  volumes=[Volume("top", (0, 0, -2), (4, 3, 0))], mesh_size=1.0,
                  stages=[Stage("initial", kind="initial"), Stage("dig", excavate=("top",))])
    problem, results = model.run()
    text = open(write_stage(tmp_path / "dig.vtu", problem, results[1])).read()

    kept = ~problem.groups["top"]
    assert np.array_equal(read_vtu_array(text, "connectivity").reshape(-1, 10), problem.mesh.elements[kept])
    assert np.all(read_vtu_array(text, "region") == 1)
    stress = read_vtu_array(text, "stress")
    z = problem.mesh.nodes[:, 2]
    # the new formation carries no vertical stress, the base the weight of the clay
    assert np.abs(stress[np.isclose(z, -2.0), 2]).max() < 1e-9
    assert np.allclose(stress[np.isclose(z, -10.0), 2], -20.0 * 8.0)

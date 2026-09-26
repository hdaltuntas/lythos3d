"""Loads drawn in plan on whatever the ground surface is."""

import numpy as np
import pytest

from lythos3d.core.analysis import AreaLoad
from lythos3d.core.materials import LinearElastic
from lythos3d.core.mesh import box_mesh, graded
from lythos3d.core.problem import Problem, Stage
from lythos3d.core.solver import Solver

SOIL = LinearElastic("clay", E=2e4, nu=0.3)


def test_an_area_load_over_everything_compresses_a_column_one_dimensionally():
    H, q = 5.0, 40.0
    mesh = box_mesh(graded(0, 2, 1.0), graded(0, 2, 1.0), graded(-H, 0, 1.0))
    load = AreaLoad([(-1, -1), (3, -1), (3, 3), (-1, 3)], q)
    r = Solver(Problem(mesh, [SOIL]), tolerance=1e-10).run([Stage("load", loads=(load,), increments=1)])[0]
    top = np.isclose(mesh.nodes[:, 2], 0.0)
    assert np.allclose(r.displacement[top, 2], -q * H / SOIL.oedometer_modulus, rtol=1e-9)


def test_on_sloping_ground_the_load_is_per_plan_area_and_follows_the_dig():
    base = box_mesh(graded(0, 6, 1.0), graded(0, 4, 1.0), graded(-5, 0, 1.0))
    # ground rising 1 in 6 across x
    mesh = base.mapped(lambda p: np.column_stack([p[:, 0], p[:, 1],
                                                  p[:, 2] + (p[:, 0] / 6.0) * (p[:, 2] + 5.0) / 5.0]))
    c = mesh.centroids()
    problem = Problem(mesh, [SOIL], groups={"dig": (c[:, 0] < 2.0) & (c[:, 2] > -2.0)})
    load = AreaLoad([(0, 0), (4, 0), (4, 4), (0, 4)], 25.0, horizontal=(5.0, 0.0))
    active = np.ones(mesh.n_elements, bool)
    f = problem.surface_loads([load], active)
    assert f[2::3].sum() == pytest.approx(-25.0 * 16.0, rel=1e-9)
    assert f[0::3].sum() == pytest.approx(5.0 * 16.0, rel=1e-9)
    # dig out x < 2 to 2 m deep: the load now bears on the pit floor and the ground beyond
    active = ~problem.groups["dig"]
    f = problem.surface_loads([load], active)
    assert f[2::3].sum() == pytest.approx(-25.0 * 16.0, rel=1e-9)
    loaded = np.nonzero(f[2::3])[0]
    floor = mesh.nodes[loaded][mesh.nodes[loaded, 0] < 2.0 - 1e-9]
    assert len(floor) and np.all(floor[:, 2] < -1.5)


def test_area_and_pile_loads_round_trip_through_json():
    from lythos3d.core.beams import PileLoad
    from lythos3d.io.site_json import load_from_dict, load_to_dict

    for load in (AreaLoad([(0, 0), (1, 0), (1, 1)], 10.0, (1.0, 2.0), "crane"),
                 PileLoad("P1", (0.0, 0.0, -500.0), (10.0, 0.0, 0.0))):
        back = load_from_dict(load_to_dict(load))
        assert load_to_dict(back) == load_to_dict(load)

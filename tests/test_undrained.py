# SPDX-License-Identifier: AGPL-3.0-only
"""Undrained loading: excess pore pressure, its dissipation, and undrained strength."""

import numpy as np
import pytest

from lythos3d.core.analysis import SurfaceLoad
from lythos3d.core.materials import LinearElastic, MohrCoulomb
from lythos3d.core.mesh import box_mesh, graded
from lythos3d.core.problem import Problem, Stage
from lythos3d.core.solver import Solver

CLAY = LinearElastic("clay", E=1.0e4, nu=0.3, drainage="undrained")


def test_water_stiffness_gives_the_undrained_poisson_ratio():
    """Skeleton plus water behave as an elastic solid with the same G and Poisson's ratio nu_u."""
    G = CLAY.shear_modulus
    K = CLAY.E / (3 * (1 - 2 * CLAY.nu)) + CLAY.water_bulk_modulus
    nu = (3 * K - 2 * G) / (2 * (3 * K + G))
    assert nu == pytest.approx(0.495, rel=1e-12)
    with pytest.raises(ValueError):
        LinearElastic("x", 1e4, 0.3, drainage="partly")
    with pytest.raises(ValueError):
        LinearElastic("x", 1e4, 0.499, drainage="undrained")         # nu above nu_u


def test_one_dimensional_loading_undrained_then_consolidated():
    """A surcharge on a laterally confined column, first undrained, then drained.

    Undrained, the column compresses against the skeleton and the water in
    parallel: w = q H / (M + Kw/n), and the water takes Kw/n / (M + Kw/n)
    of the load.  Once the water has drained, the skeleton carries it all:
    w = q H / M, and no excess pore pressure is left.
    """
    H, q = 6.0, 60.0
    mesh = box_mesh(graded(0, 2, 1.0), graded(0, 2, 1.0), graded(-H, 0, 1.0))
    load = SurfaceLoad("z", 0.0, (0, 0, -q))
    solver = Solver(Problem(mesh, [CLAY]), tolerance=1e-10)
    undrained = solver.run_stage(Stage("load", loads=(load,), increments=1))
    consolidated = solver.run_stage(Stage("consolidate", loads=(load,), drained=True, increments=1,
                                          reset_displacements=False))
    M, kw = CLAY.oedometer_modulus, CLAY.water_bulk_modulus
    top = np.isclose(mesh.nodes[:, 2], 0.0)
    assert np.allclose(undrained.displacement[top, 2], -q * H / (M + kw), rtol=1e-9)
    assert np.allclose(undrained.excess_pore_pressure, q * kw / (M + kw), rtol=1e-9)
    # total vertical stress is the surcharge: sigma' - p
    total = undrained.state.stress[:, 2] - undrained.pore_pressure
    assert np.allclose(total, -q, rtol=1e-9)
    assert np.allclose(consolidated.displacement[top, 2], -q * H / M, rtol=1e-9)
    assert np.abs(consolidated.excess_pore_pressure).max() == 0.0
    assert np.allclose(consolidated.state.stress[:, 2], -q, rtol=1e-9)


def test_cohesion_can_grow_with_depth():
    soil = MohrCoulomb("soft clay", E=5e3, nu=0.3, c=10.0, phi=0.0, c_inc=2.0, z_ref=0.0, tension_cutoff=None)
    assert np.allclose(soil.cohesion(np.array([1.0, 0.0, -5.0])), [10.0, 10.0, 20.0])
    assert soil.reduced(2.0).c_inc == 1.0
    # pure shear of 15 kPa: beyond the strength at the surface, within it at 5 m
    stress = np.array([[0, 0, 0, 15.0, 0, 0]] * 2)
    f = soil.yield_function(stress, np.array([0.0, -5.0]))
    assert f[0] > 0 > f[1]
    # the return map puts each point on its own surface
    from lythos3d.core.materials import MaterialState

    state = MaterialState.zeros(2)
    s, _, new = soil.update(state, np.array([[0, 0, 0, 2 * 20.0 / soil.shear_modulus, 0, 0]] * 2),
                            np.array([0.0, -2.5]))
    assert np.allclose(s[:, 3], [10.0, 15.0])
    assert new.yielding.all()


def _footing(su, q, h=0.5):
    """A strip footing 2 m wide (half of it, by symmetry) on undrained clay, as a plane-strain slice."""
    soil = MohrCoulomb("clay", E=1e4, nu=0.3, gamma=18.0, c=su, phi=0.0, tension_cutoff=None,
                       drainage="undrained")
    mesh = box_mesh(graded(0, 8, h, [1.0]), np.array([0.0, h]), graded(-6, 0, h))
    load = SurfaceLoad("z", 0.0, (0, 0, -q), where=lambda c: c[:, 0] < 1.0)
    return Problem(mesh, [soil]), load


@pytest.mark.slow
def test_undrained_bearing_capacity_by_strength_reduction():
    """Prandtl: a strip on undrained clay fails at (2 + pi) su.

    With q = 50 kPa on su = 30 kPa the factor of safety is 5.14 x 30 / 50 =
    3.08.  The slice, loaded undrained with the water's stiffness in, must
    land a little above it, as a finite element mesh does.
    """
    problem, load = _footing(30.0, 50.0)
    results = Solver(problem).run([Stage("initial", kind="initial", initial_stress="gravity"),
                                   Stage("load", loads=(load,)),
                                   Stage("fos", kind="ssr", loads=(load,), srf_min=1.5, srf_max=5.0)])
    assert results[1].converged
    assert results[1].excess_pore_pressure.max() > 0.0
    assert results[-1].srf == pytest.approx((2 + np.pi) * 30.0 / 50.0, rel=0.06), results[-1].srf


def test_undrained_json_round_trip(tmp_path):
    from lythos3d.io.site_json import material_from_dict, material_to_dict, site_from_dict

    soil = MohrCoulomb("clay", E=5e3, nu=0.3, c=15.0, phi=0.0, c_inc=1.5, z_ref=-1.0, drainage="undrained")
    assert material_from_dict(material_to_dict(soil)) == soil
    d = {"extent": {"x": [0, 10], "y": [0, 10]}, "bottom": -10,
         "soils": [{"name": "clay", **{k: v for k, v in material_to_dict(soil).items() if k != "name"}}],
         "boreholes": [{"name": "A", "x": 0, "y": 0, "tops": [["clay", 0.0]]}],
         "stages": [{"name": "i", "kind": "initial"}, {"name": "wait", "drained": True}]}
    site = site_from_dict(d)
    assert site.stages[1].drained

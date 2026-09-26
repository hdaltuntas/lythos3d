# SPDX-License-Identifier: AGPL-3.0-only
"""Groundwater: hydrostatic pore pressure in a drained, effective-stress analysis."""

import numpy as np
import pytest

from lythos3d.core.materials import LinearElastic
from lythos3d.core.mesh import box_mesh, graded
from lythos3d.core.problem import Problem, Stage
from lythos3d.core.solver import Solver
from lythos3d.core.water import GAMMA_WATER, Drawdown, WaterTable

GW = GAMMA_WATER
SOIL = LinearElastic("sand", E=2.0e4, nu=0.3, gamma=18.0, gamma_sat=20.0)


def _column(H=10.0, breaks=(), material=SOIL, water=None):
    from lythos3d.core.model import Model, Stratum

    mesh = box_mesh(graded(0, 2, 1.0), graded(0, 2, 1.0), graded(-H, 0, 1.0, breaks))
    mesh.region = np.zeros(mesh.n_elements, dtype=np.int64)
    strata = Model("column", (0, 2), (0, 2), -H, [Stratum("sand", material, 0.0)])
    return Problem(mesh, [material], vertical_stress=strata.overburden, water=water)


def _expected_vertical(z, level, top=0.0):
    """Effective vertical stress (compression positive) under a level water table."""
    depth = top - z
    wet = np.clip(level - z, 0.0, depth)
    return SOIL.gamma * (depth - wet) + (SOIL.saturated_weight - GW) * wet


def test_water_table_head_wells_and_drawdowns():
    assert WaterTable.dry().is_dry
    assert np.all(WaterTable.dry().pressure([[0, 0, -5]]) == 0.0)
    flat = WaterTable(level=-2.0)
    assert np.allclose(flat.pressure([[0, 0, -5], [0, 0, 1]]), [3 * GW, 0.0])
    tilted = WaterTable(wells=[(0, 0, -1.0), (10, 0, -3.0), (0, 10, -1.0)])
    assert np.allclose(tilted.head([[5, 5, 0]]), -2.0)
    assert np.allclose(tilted.head([[20, 5, 0]]), -3.0)          # level beyond the wells
    pit = flat.lowered([(0, 0), (4, 0), (4, 4), (0, 4)], -6.0)
    assert np.allclose(pit.head([[2, 2, 0], [6, 2, 0]]), [-6.0, -2.0])
    # a drawdown never raises the water
    assert np.allclose(WaterTable(level=-8.0, drawdowns=[Drawdown([(0, 0), (4, 0), (4, 4)], -6.0)])
                       .head([[3, 1, 0]]), -8.0)
    with pytest.raises(ValueError):
        WaterTable(level=0.0, wells=[(0, 0, 0.0)])


def test_saturated_weight_defaults_to_the_dry_one():
    assert LinearElastic("x", E=1e4, nu=0.3, gamma=17.0).saturated_weight == 17.0
    assert SOIL.saturated_weight == 20.0


@pytest.mark.parametrize("level", [-3.0, 0.0, 2.5])
def test_k0_stresses_under_water_are_already_in_equilibrium(level):
    """Effective K0 stresses below a level water table (or under standing water) need no correction.

    The skeleton's load - saturated weight, the divergence of the pore
    pressure, and the water standing on the ground - balances the K0 field
    exactly, so the initial stage moves nothing.
    """
    problem = _column(breaks=(level,), water=WaterTable(level=level))
    solver = Solver(problem, tolerance=1e-10)
    r = solver.run([Stage("initial", kind="initial")])[0]
    assert r.converged
    assert np.abs(r.displacement).max() == 0.0
    z = problem.continuum.gauss_xyz.reshape(-1, 3)[:, 2]
    sv = _expected_vertical(z, level)
    assert np.allclose(r.state.stress[:, 2], -sv, atol=1e-9)
    assert np.allclose(r.state.stress[:, 0], -SOIL.k0 * sv, atol=1e-9)
    assert np.allclose(r.pore_pressure, GW * np.maximum(level - z, 0.0))
    # the residual of the K0 field under its loads is round-off
    f_int = problem.continuum.internal_forces(r.state.stress)
    from lythos3d.core.assembly import assemble_vector

    residual = (problem.gravity(problem_active(problem), problem.water)
                + problem.water_loads(problem.water, problem_active(problem))
                - assemble_vector(problem.n_dof, problem.dofs, f_int))
    residual[problem.fixed] = 0.0
    assert np.abs(residual).max() < 1e-9 * np.abs(f_int).max()


def problem_active(problem):
    return np.ones(problem.mesh.n_elements, bool)


def test_gravity_loading_under_water_gives_buoyant_effective_stress():
    level = -4.0
    problem = _column(breaks=(level,), water=WaterTable(level=level))
    r = Solver(problem, tolerance=1e-10).run([Stage("initial", kind="initial", initial_stress="gravity")])[0]
    z = problem.continuum.gauss_xyz.reshape(-1, 3)[:, 2]
    assert np.allclose(r.state.stress[:, 2], -_expected_vertical(z, level), atol=1e-8)
    # total stress is the full saturated weight: sigma' - p
    total = r.state.stress[:, 2] - r.pore_pressure
    expected = -(SOIL.gamma * np.minimum(-z, -level) + SOIL.saturated_weight * np.maximum(level - z, 0.0))
    assert np.allclose(total, expected, atol=1e-8)


def test_lowering_the_water_table_settles_the_ground_by_the_effective_stress_gained():
    """Drawdown from the surface to depth d: sigma_v' grows by gamma_w min(depth, d).

    With gamma = gamma_sat the soil's weight does not change, so the whole
    settlement comes from the pore pressure lost:
    s = gamma_w / M (d^2 / 2 + d (H - d)).
    """
    H, d = 10.0, 4.0
    soil = LinearElastic("sand", E=2.0e4, nu=0.3, gamma=20.0)
    problem = _column(H, breaks=(-d,), material=soil, water=WaterTable(level=0.0))
    results = Solver(problem, tolerance=1e-10).run([
        Stage("initial", kind="initial"),
        Stage("pump", water=WaterTable(level=-d), increments=1),
    ])
    top = np.isclose(problem.mesh.nodes[:, 2], 0.0)
    settlement = -results[-1].displacement[top, 2]
    expected = GW / soil.oedometer_modulus * (d ** 2 / 2 + d * (H - d))
    assert np.allclose(settlement, expected, rtol=1e-9)
    assert np.abs(results[-1].displacement[:, :2]).max() < 1e-12


def test_a_wall_with_interfaces_carries_the_difference_in_water_pressure():
    """Water at 0 on one side of a wall and drawn down to -6 on the other.

    The wall's own nodes touch only interfaces, so the water load on them is
    exactly what the water does to the wall: the difference of the two
    hydrostatic pressures, 1/2 gamma_w (12^2 - 6^2) per metre over a wall
    from the base of the model, -12, to the surface.
    """
    from lythos3d.core.interfaces import InterfaceSpec
    from lythos3d.core.model import Model, Stratum, Wall
    from lythos3d.core.structures import PlateSection

    water = WaterTable(level=0.0).lowered([(5, -1), (20, -1), (20, 2), (5, 2)], -6.0)
    model = Model("wall", (0, 20), (0, 1), -12.0, [Stratum("sand", SOIL, 0.0)],
                  walls=[Wall("wall", (5, 0, -12), (5, 1, 0), PlateSection(E=3e7, nu=0.2, t=0.5),
                              interface=InterfaceSpec())],
                  mesh_size=1.0, water=water)
    problem = model.build()
    f = problem.water_loads(water, np.ones(problem.mesh.n_elements, bool))
    wall = np.unique(problem.interface_elements.wall_faces)
    fx = f[3 * wall].sum()
    expected = 0.5 * GW * (12.0 ** 2 - 6.0 ** 2)
    assert fx == pytest.approx(expected, rel=1e-6)   # the pressure is read just inside the soil
    assert abs(f[3 * wall + 2].sum()) < 1e-9 * expected


def test_a_flooded_pit_bottom_carries_the_water_standing_on_it():
    """Dig out a block with the water table above its floor: the water left in the pit presses on it.

    Digging a volume of saturated soil out from under the water and letting
    water stand in its place is, for the skeleton below, the same as taking
    away the buoyant weight of the soil only.
    """
    from lythos3d.core.model import Model, Stratum, Volume

    def run(water, soil):
        model = Model("pit", (0, 8), (0, 8), -10.0, [Stratum("sand", soil, 0.0)],
                      volumes=[Volume("dig", (0, 0, -3), (8, 8, 0))], water=water,
                      stages=[Stage("initial", kind="initial"), Stage("dig", excavate=("dig",))],
                      mesh_size=2.0, vertical_mesh_size=1.0)
        problem, results = model.run(tolerance=1e-10)
        return problem, results[-1]

    # all of the top 3 m dug out, water at the surface: heave from gamma' 3 m unloading
    problem, r = run(WaterTable(level=0.0), SOIL)
    # the same with dry soil of the buoyant weight
    buoyant = LinearElastic("b", E=SOIL.E, nu=SOIL.nu, gamma=SOIL.saturated_weight - GW)
    p_dry, r_dry = run(None, buoyant)
    floor = np.isclose(problem.mesh.nodes[:, 2], -3.0)
    assert np.allclose(r.displacement[floor], r_dry.displacement[floor], atol=1e-12)
    assert r.displacement[floor, 2].min() > 0.0


def test_dewatered_excavations_draw_the_water_down_in_the_default_stages():
    from lythos3d.core.model import Site
    from lythos3d.core.site import Borehole, Excavation, Soil, SoilProfile

    profile = SoilProfile([Soil("sand", SOIL)], [Borehole("BH", 0, 0, [("sand", 0.0)])], -10.0)
    pit = Excavation("pit", [(2, 2), (6, 2), (6, 6), (2, 6)], [-2.0, -4.0], dewatered=True)
    site = Site("s", profile, (0, 10), (0, 10), [pit], water=WaterTable(level=-1.0))
    digs = [s for s in site.stages if s.excavate]
    assert [float(s.water.head([[4, 4, 0]])[0]) for s in digs] == [-2.0, -4.0]
    assert all(float(s.water.head([[8, 8, 0]])[0]) == -1.0 for s in digs)
    dry = Site("s", profile, (0, 10), (0, 10), [Excavation("pit", pit.polygon, pit.levels)],
               water=WaterTable(level=-1.0))
    assert all(s.water is None for s in dry.stages)


def test_water_round_trips_through_json(tmp_path):
    from lythos3d.core.model import Site
    from lythos3d.core.site import Borehole, Excavation, Soil, SoilProfile
    from lythos3d.io.site_json import load_site, save_site

    profile = SoilProfile([Soil("sand", SOIL)], [Borehole("BH", 0, 0, [("sand", 0.0)])], -10.0)
    pit = Excavation("pit", [(2, 2), (6, 2), (6, 6), (2, 6)], [-2.0, -4.0], dewatered=True)
    site = Site("s", profile, (0, 10), (0, 10), [pit],
                water=WaterTable(wells=[(0, 0, -1.0), (10, 0, -1.5), (0, 10, -1.2)]))
    back = load_site(save_site(site, tmp_path / "s.json"))
    assert back.water == site.water
    assert back.excavations[0].dewatered
    assert back.profile.soils[0].material.gamma_sat == 20.0
    assert [s.water for s in back.stages] == [s.water for s in site.stages]


def test_pore_pressure_and_total_stress_are_written_for_paraview(tmp_path):
    from lythos3d.io.vtu import read_vtu_array, write_stage

    problem = _column(breaks=(-3.0,), water=WaterTable(level=-3.0))
    r = Solver(problem).run([Stage("initial", kind="initial")])[0]
    text = open(write_stage(tmp_path / "w.vtu", problem, r)).read()
    p = read_vtu_array(text, "pore_pressure")
    z = problem.mesh.nodes[:, 2]
    assert np.allclose(p, GW * np.maximum(-3.0 - z, 0.0), atol=1e-9)
    total = read_vtu_array(text, "total_stress").reshape(-1, 6)
    stress = read_vtu_array(text, "stress").reshape(-1, 6)
    assert np.allclose(total[:, 2], stress[:, 2] - p)

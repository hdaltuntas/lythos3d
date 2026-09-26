"""Constructing ground: embankments on the surface and backfill in a dug pit."""

import numpy as np
import pytest

from lythos3d.core.materials import LinearElastic
from lythos3d.core.model import Fill, Model, Stratum, Volume
from lythos3d.core.problem import Stage

SOIL = LinearElastic("clay", E=2.0e4, nu=0.3, gamma=18.0)
FILL = LinearElastic("fill", E=5.0e4, nu=0.3, gamma=20.0)


def test_a_fill_layer_over_the_whole_site_loads_the_ground_one_dimensionally():
    """Fill t thick over everything: the ground settles gamma_f t H / M, and the fill carries its own weight."""
    H, t = 6.0, 2.0
    model = Model("layer", (0, 2), (0, 2), -H, [Stratum("clay", SOIL, 0.0)],
                  fills=[Fill("fill", (0, 0, 0), (2, 2, t), FILL)],
                  stages=[Stage("initial", kind="initial"), Stage("place", construct=("fill",), increments=1)],
                  mesh_size=1.0)
    problem, results = model.run(tolerance=1e-10)
    assert results[0].active.sum() < problem.mesh.n_elements          # the fill is not there at first
    x = problem.mesh.nodes
    surface = np.isclose(x[:, 2], 0.0)
    assert np.allclose(results[-1].displacement[surface, 2], -FILL.gamma * t * H / SOIL.oedometer_modulus,
                       rtol=1e-9)
    z = problem.continuum.gauss_xyz.reshape(-1, 3)[:, 2]
    in_fill = z > 0
    assert np.allclose(results[-1].state.stress[in_fill, 2], -FILL.gamma * (t - z[in_fill]), atol=1e-8)
    # the ground below carries its own weight and the fill's
    below = ~in_fill
    assert np.allclose(results[-1].state.stress[below, 2], -(SOIL.gamma * -z[below] + FILL.gamma * t), atol=1e-8)


def test_backfill_takes_its_material_and_starts_free_of_stress():
    dig = Volume("dig", (0, 0, -2), (1, 2, 0))
    model = Model("backfill", (0, 4), (0, 2), -6.0, [Stratum("clay", SOIL, 0.0)], volumes=[dig],
                  fills=[Fill("backfill", (0, 0, -2), (1, 2, 0), FILL)],
                  stages=[Stage("initial", kind="initial"), Stage("dig", excavate=("dig",)),
                          Stage("backfill", construct=("backfill",))], mesh_size=0.5)
    problem, results = model.run(tolerance=1e-9)
    assert all(r.converged for r in results)
    fill = problem.groups["backfill"]
    assert not results[1].active[fill].any() and results[2].active[fill].all()
    assert np.all(problem.mesh.region[fill] == 1)
    # the backfill carries its own weight only: nothing of the dug ground's stress
    ngp = problem.continuum.n_gauss
    gp = (np.nonzero(fill)[0][:, None] * ngp + np.arange(ngp)).ravel()
    z = problem.continuum.gauss_xyz.reshape(-1, 3)[gp, 2]
    szz = results[-1].state.stress[gp, 2]
    w = problem.continuum.detJw.ravel()[gp]
    # the fill's mean vertical stress is its own weight at mid-depth, less a
    # little hung on the sides; the dug ground had carried about twice that
    mean = float(np.sum(szz * w) / np.sum(w))
    assert -FILL.gamma * 1.0 < mean < -0.6 * FILL.gamma * 1.0
    # a solver started again begins from the original materials
    problem.reset_regions()
    assert np.all(problem.mesh.region[fill] == 0)


def test_fill_names_must_not_clash():
    with pytest.raises(ValueError):
        Model("x", (0, 1), (0, 1), -1, [Stratum("s", SOIL, 0.0)], volumes=[Volume("a", (0, 0, -1), (1, 1, 0))],
              fills=[Fill("a", (0, 0, 0), (1, 1, 1), FILL)])


def test_an_embankment_drawn_in_plan_is_meshed_above_the_ground_and_raised_in_lifts(tmp_path):
    pytest.importorskip("gmsh")
    from lythos3d.core.model import Site
    from lythos3d.core.site import Borehole, SiteFill, Soil, SoilProfile
    from lythos3d.io.site_json import load_site, save_site

    clay = LinearElastic("clay", E=1e4, nu=0.3, gamma=17.0)
    profile = SoilProfile([Soil("clay", clay)],
                          [Borehole("A", 0, 0, [("clay", 0.0)]), Borehole("B", 30, 0, [("clay", -1.0)]),
                           Borehole("C", 0, 20, [("clay", 0.0)])], -10.0)
    bank = SiteFill("bank", [(8, 0), (22, 0), (22, 20), (8, 20)], [1.5, 3.0], FILL)
    site = Site("embankment", profile, (0, 30), (0, 20), fills=[bank], mesh_size=2.5)
    names = [s.name for s in site.stages]
    assert names[1:3] == ["bank: raise to 1.5", "bank: raise to 3"]
    problem = site.build()
    mesh = problem.mesh
    for k, name in enumerate(bank.lift_names):
        mask = problem.groups[name]
        assert mask.any() and np.all(mesh.region[mask] == 1)
        top = [1.5, 3.0][k]
        # the lift's volume: its footprint times its thickness above the ground
        volume = problem.continuum.detJw[mask].sum()
        xs = np.linspace(8, 22, 141)
        ground = profile.ground(np.column_stack([xs, np.full_like(xs, 10.0), np.zeros_like(xs)]))
        lower = ground if k == 0 else np.full_like(xs, 1.5)
        expected = 20.0 * np.trapezoid(np.clip(top - lower, 0, None), xs)
        assert volume == pytest.approx(expected, rel=0.02)     # the lofted ground rounds the hull's kink
    assert np.all(problem.inactive == (problem.groups["bank 1"] | problem.groups["bank 2"]))
    stages = [s for s in site.stages if s.kind != "ssr"]
    from lythos3d.core.solver import Solver

    results = Solver(problem).run(stages)
    assert all(r.converged for r in results)
    assert results[-1].displacement[:, 2].min() < 0.0
    back = load_site(save_site(site, tmp_path / "bank.json"))
    assert back.fills[0].levels == bank.levels and back.fills[0].material == FILL
    assert back.stages[1].construct == ("bank 1",)

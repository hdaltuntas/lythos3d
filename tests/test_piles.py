# SPDX-License-Identifier: AGPL-3.0-only
"""Embedded piles."""

import numpy as np
import pytest

from lythos3d.core.beams import BeamSection, EmbeddedPile, PileLoad
from lythos3d.core.materials import LinearElastic
from lythos3d.core.model import Model, Stratum
from lythos3d.core.problem import Stage
from lythos3d.core.solver import Solver

CLAY = LinearElastic("clay", E=3.0e4, nu=0.3, gamma=0.0)
SECTION = BeamSection.circular(E=3.0e7, D=0.8)


def _ground(piles, soil=CLAY, size=1.0):
    return Model("ground", (-6, 6), (-6, 6), -16.0, [Stratum("soil", soil, 0.0)], piles=piles, mesh_size=size)


def _pile(**kw):
    kw.setdefault("skin", (20.0, 40.0))
    kw.setdefault("base", 200.0)
    return EmbeddedPile("pile", (0.3, 0.2, 0.0), (0.3, 0.2, -10.0), SECTION, element_size=1.0, **kw)


def test_pile_head_load_is_carried_by_the_shaft_and_the_tip():
    """Below its capacity every kilonewton on the head goes into the skin or the tip."""
    problem = _ground([_pile(skin=None, base=None)]).build()
    solver = Solver(problem, tolerance=1e-8)
    solver.run_stage(Stage("install", install=("pile",)))
    r = solver.run_stage(Stage("load", loads=(PileLoad("pile", (0, 0, -300.0)),)))
    assert r.converged
    f = r.pile_forces["pile"]
    w = problem.embedded.w_skin[problem.skin_pile == 0]
    # the skin resists the push: its traction points back up the pile
    assert np.sum(-f["skin"] * w) + f["base"] == pytest.approx(300.0, rel=1e-6)
    # and the axial force falls from 300 kN at the head towards the tip
    N = f["resultants"][:, 0]
    order = np.argsort(f["s"])
    assert N[order][0] == pytest.approx(-300.0, rel=0.05)
    assert abs(N[order][-1]) < abs(N[order][0])
    assert f["base"] > 0


def test_axial_capacity_is_shaft_plus_base():
    """Capacity (T_top + T_tip) / 2 L + F_max: holds at 97%, not at 103%.

    The capacity does not depend on the springs' stiffness or on the mesh,
    so a coarse one will do; proving that no equilibrium exists is what
    takes the time.
    """
    capacity = 0.5 * (20.0 + 40.0) * 10.0 + 200.0
    problem = Model("ground", (-4, 4), (-4, 4), -14.0, [Stratum("soil", CLAY, 0.0)],
                    piles=[_pile()], mesh_size=2.0).build()
    solver = Solver(problem, tolerance=1e-6, max_iterations=20)
    solver.run_stage(Stage("install", install=("pile",)))
    below = solver.run_stage(Stage("0.97", loads=(PileLoad("pile", (0, 0, -0.97 * capacity)),), increments=10))
    assert below.converged
    above = solver.run_stage(Stage("1.03", loads=(PileLoad("pile", (0, 0, -1.03 * capacity)),), increments=10))
    assert not above.converged


def test_pile_installed_in_moving_ground_starts_free_of_force():
    soil = LinearElastic("clay", E=3.0e4, nu=0.3, gamma=18.0)
    model = _ground([_pile()], soil=soil)
    model.stages = [Stage("gravity", kind="initial", initial_stress="gravity"), Stage("install", install=("pile",))]
    _, results = model.run(tolerance=1e-8)
    f = results[-1].pile_forces["pile"]
    assert np.abs(f["resultants"]).max() < 1e-6 and np.abs(f["skin"]).max() < 1e-6 and f["base"] == 0.0


def test_lateral_load_is_resisted_near_the_top():
    """A horizontal push on the head: the pile bends most near the top, and the soil holds it."""
    problem = _ground([_pile()]).build()
    solver = Solver(problem, tolerance=1e-8)
    solver.run_stage(Stage("install", install=("pile",)))
    r = solver.run_stage(Stage("push", loads=(PileLoad("pile", (50.0, 0, 0)),)))
    assert r.converged
    f = r.pile_forces["pile"]
    moment = np.hypot(f["resultants"][:, 4], f["resultants"][:, 5])
    s = f["s"]
    assert s[np.argmax(moment)] < 4.0                          # the peak is in the upper part
    assert moment[s > 8.0].max() < 0.2 * moment.max()          # and it has died away at depth
    head = problem.pile_node_dofs[0][0]
    free = 50.0 * 10.0 ** 3 / (3 * SECTION.EI2)                # the same pile with no soil at all
    assert 0 < solver._u[head[0]] < 0.1 * free


def test_pile_in_a_gmsh_site_and_its_json(tmp_path):
    """A pile through dipping strata on a site, and the site file carrying it."""
    import json

    from lythos3d.examples import sloping_site
    from lythos3d.io.site_json import load_site, save_site, site_to_dict

    pytest.importorskip("gmsh", exc_type=ImportError)
    site = sloping_site(mesh_size=3.0)
    # the head just below the ground there: the meshed surface is a loft
    # through the interpolated one and may sit a few millimetres off it
    ground = float(site.profile.ground(np.array([[20.0, 12.0, 0.0]]))[0])
    site.piles = [EmbeddedPile("pile", (20.0, 12.0, ground - 0.05), (20.0, 12.0, -9.0), SECTION,
                               skin=(30.0, 60.0), base=300.0)]
    site.excavations = []
    site.stages = [Stage("initial", kind="initial"), Stage("install", install=("pile",)),
                   Stage("load", loads=(PileLoad("pile", (0, 0, -400.0)),))]
    problem, results = site.run(tolerance=1e-6)
    assert all(r.converged for r in results)
    f = results[-1].pile_forces["pile"]
    w = problem.embedded.w_skin[problem.skin_pile == 0]
    assert np.sum(-f["skin"] * w) + f["base"] == pytest.approx(400.0, rel=1e-4)

    path = save_site(site, tmp_path / "site.json")
    again = load_site(path)
    assert site_to_dict(again) == json.load(open(path))
    pile = again.piles[0]
    assert pile.skin == (30.0, 60.0) and pile.base == 300.0
    assert pile.section.EA == pytest.approx(SECTION.EA)


@pytest.mark.slow
def test_embedded_pile_is_as_stiff_as_a_pile_of_solid_elements():
    """The same 10 m, 0.8 m pile in elastic ground, modelled two ways.

    Coupled at its perimeter the embedded pile settles within 6% of the
    solid-element pile and moves within 10% laterally, at every mesh tried
    (2, 1 and 0.7 m); coupled on its axis it had settled twice as much, and
    more the finer the mesh.  The rest is the springs' own give and the
    solid pile's square section.
    """
    from lythos3d.core.analysis import SurfaceLoad
    from lythos3d.core.mesh import box_mesh, graded
    from lythos3d.core.problem import Problem

    h, D, L, P, H = 1.0, 0.8, 10.0, 300.0, 50.0
    concrete = LinearElastic("concrete", 3e7, 0.2, 0.0)
    b = D * np.sqrt(np.pi) / 2
    X, Z = graded(-6, 6, h, [-b / 2, b / 2]), graded(-15, 0, h, [-L])
    solid = box_mesh(X, X, Z)
    c = solid.centroids()
    solid.region = ((np.abs(c[:, 0]) < b / 2) & (np.abs(c[:, 1]) < b / 2) & (c[:, 2] > -L)).astype(np.int64)
    head = np.nonzero(np.all(np.isclose(solid.nodes, 0.0), axis=1))[0][0]

    def on_top(cc):
        return (np.abs(cc[:, 0]) < b / 2) & (np.abs(cc[:, 1]) < b / 2)

    reference = {}
    for name, traction, k in (("axial", (0, 0, -P / b ** 2), 2), ("lateral", (H / b ** 2, 0, 0), 0)):
        r = Solver(Problem(solid, [CLAY, concrete]), tolerance=1e-8).run_stage(
            Stage("load", loads=(SurfaceLoad("z", 0.0, traction, where=on_top),)))
        reference[name] = r.displacement[head, k]

    embedded = {}
    for name, force, k in (("axial", (0, 0, -P), 2), ("lateral", (H, 0, 0), 0)):
        problem = Problem(box_mesh(X, X, Z), [CLAY], piles=[EmbeddedPile(
            "pile", (0, 0, 0), (0, 0, -L), BeamSection.circular(3e7, D), element_size=h)])
        solver = Solver(problem, tolerance=1e-8)
        solver.run_stage(Stage("install", install=("pile",)))
        solver.run_stage(Stage("load", loads=(PileLoad("pile", force),)))
        embedded[name] = solver._u[problem.pile_node_dofs[0][0][k]]

    assert embedded["axial"] / reference["axial"] == pytest.approx(1.0, abs=0.08)
    assert embedded["lateral"] / reference["lateral"] == pytest.approx(1.0, abs=0.12)

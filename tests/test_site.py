# SPDX-License-Identifier: AGPL-3.0-only
"""Boreholes, excavations in plan, and gmsh meshes of a site."""

import json

import numpy as np
import pytest

from lythos3d.core.elements import ContinuumElements
from lythos3d.core.materials import LinearElastic
from lythos3d.core.site import Borehole, Excavation, Soil, SoilProfile


def _soils():
    return [Soil("fill", LinearElastic("fill", 2e4, 0.3, 18.0)),
            Soil("clay", LinearElastic("clay", 3e4, 0.3, 19.0)),
            Soil("sand", LinearElastic("sand", 6e4, 0.3, 20.0))]


def _three_boreholes():
    return [Borehole("A", 0, 0, [("fill", 0), ("clay", -2), ("sand", -6)]),
            Borehole("B", 30, 0, [("fill", 1), ("sand", -4)]),          # no clay here
            Borehole("C", 0, 20, [("fill", -1), ("clay", -3), ("sand", -8)])]


# --------------------------------------------------------------------------- profile
def test_one_borehole_gives_level_strata():
    p = SoilProfile(_soils(), [Borehole("A", 5, 5, [("fill", 0), ("clay", -2), ("sand", -6)])], -15)
    pts = np.array([[0, 0, 0], [100, -40, 0], [5, 5, 0]], float)
    assert np.allclose(p.tops(pts), [0, -2, -6])


def test_interpolation_is_exact_at_the_boreholes_and_linear_between():
    p = SoilProfile(_soils(), _three_boreholes(), -15)
    at_holes = p.tops(np.array([[0, 0, 0], [30, 0, 0], [0, 20, 0]], float))
    assert np.allclose(at_holes, [[0, -2, -6], [1, -4, -4], [-1, -3, -8]])
    # halfway between A and B the clay has thinned to half
    mid = p.tops(np.array([[15, 0, 0]], float))[0]
    assert np.allclose(mid, [0.5, -3.0, -5.0])


def test_a_soil_missing_from_a_borehole_pinches_out_there():
    p = SoilProfile(_soils(), _three_boreholes(), -15)
    tops = p.tops(np.array([[30, 0, 0]], float))[0]
    assert tops[1] == tops[2]                                   # no clay at B
    assert p.soil_of(np.array([[30, 0, -3.0]]))[0] == 0         # fill down to the sand


def test_outside_the_boreholes_the_surfaces_continue_from_the_hull():
    p = SoilProfile(_soils(), _three_boreholes(), -15)
    # the hull edge B-C is nearest (30, 30): the values at the foot of the
    # perpendicular from it, a fraction t of the way from B to C
    b, c = np.array([30.0, 0.0]), np.array([0.0, 20.0])
    t = np.dot(np.array([30.0, 30.0]) - b, c - b) / np.dot(c - b, c - b)
    far = p.tops(np.array([[30, 30, 0]], float))[0]
    assert np.allclose(far, (1 - t) * np.array([1, -4, -4]) + t * np.array([-1, -3, -8]))
    # and the surfaces are continuous across the hull
    eps = 1e-6
    inside = p.tops(np.array([[15, 10 - eps, 0]], float))
    outside = p.tops(np.array([[15, 10 + eps, 0]], float))
    assert np.allclose(inside, outside, atol=1e-5)


def test_overburden_is_the_weight_of_the_soil_column():
    p = SoilProfile(_soils(), _three_boreholes(), -15)
    sv = p.overburden(np.array([[0, 0, -10.0], [30, 0, -10.0], [0, 0, 5.0]]))
    assert np.allclose(sv, [18 * 2 + 19 * 4 + 20 * 4, 18 * 5 + 20 * 6, 0.0])


def test_profile_validation():
    soils = _soils()
    with pytest.raises(ValueError, match="unknown soil"):
        SoilProfile(soils, [Borehole("A", 0, 0, [("peat", 0)])], -10)
    with pytest.raises(ValueError, match="out of order"):
        SoilProfile(soils, [Borehole("A", 0, 0, [("clay", 0), ("fill", -2)])], -10)
    with pytest.raises(ValueError, match="fall with depth"):
        SoilProfile(soils, [Borehole("A", 0, 0, [("fill", 0), ("clay", 1)])], -10)
    with pytest.raises(ValueError, match="below the bottom"):
        SoilProfile(soils, [Borehole("A", 0, 0, [("fill", 0), ("clay", -12)])], -10)


def test_excavation_lifts():
    e = Excavation("pit", [(0, 0), (4, 0), (4, 4), (0, 4)], [-1.5, -3.0])
    assert e.lift_names == ["pit 1", "pit 2"]
    lift = e.lift_of(np.array([[1, 1, -1.0], [1, 1, -2.0], [1, 1, -4.0], [5, 1, -1.0]]))
    assert lift.tolist() == [0, 1, -1, -1]
    with pytest.raises(ValueError, match="fall lift by lift"):
        Excavation("pit", [(0, 0), (4, 0), (4, 4)], [-2.0, -1.0])


# --------------------------------------------------------------------------- meshing
gmsh = pytest.importorskip("gmsh", exc_type=ImportError)


def _reference_volumes(profile, x, y, n=400):
    xs = np.linspace(*x, n + 1)
    ys = np.linspace(*y, n + 1)
    X, Y = np.meshgrid(0.5 * (xs[1:] + xs[:-1]), 0.5 * (ys[1:] + ys[:-1]))
    pts = np.column_stack([X.ravel(), Y.ravel()])
    tops = profile.tops(pts)
    bases = np.column_stack([tops[:, 1:], np.full(len(pts), profile.bottom)])
    return ((tops - bases) * (x[1] - x[0]) * (y[1] - y[0]) / n ** 2).sum(axis=0)


def test_level_site_mesh_has_exact_layer_and_lift_volumes():
    from lythos3d.core.gmsh_mesh import mesh_site

    p = SoilProfile(_soils(), [Borehole("A", 10, 10, [("fill", 0), ("clay", -2), ("sand", -6)])], -15)
    pit = Excavation("pit", [(4, 4), (12, 4), (12, 10), (4, 10)], [-1.5, -3.0])
    mesh, groups, _, _ = mesh_site(p, (0, 20), (0, 16), [pit], mesh_size=2.0)
    v = ContinuumElements(mesh.nodes, mesh.elements).volumes()
    assert v.sum() == pytest.approx(20 * 16 * 15)
    for soil, thickness in enumerate([2, 4, 9]):
        assert v[mesh.region == soil].sum() == pytest.approx(20 * 16 * thickness)
    for name in pit.lift_names:
        assert v[groups[name]].sum() == pytest.approx(8 * 6 * 1.5)
    assert mesh.quality().min() > 0.2


def test_sloping_site_mesh_follows_the_interpolated_strata():
    """Three boreholes, dipping strata, a layer pinching out and a pit.

    The meshed layer volumes must match a quadrature of the interpolated
    surfaces; the small difference is the lofted surfaces smoothing the
    kinks of the piecewise-linear interpolation.  The pit's second lift is
    below the ground everywhere, so its volume is exact.
    """
    from lythos3d.core.gmsh_mesh import mesh_site

    p = SoilProfile(_soils(), _three_boreholes(), -15)
    pit = Excavation("pit", [(4, 4), (12, 4), (12, 10), (4, 10)], [-1.5, -3.0])
    mesh, groups, _, _ = mesh_site(p, (0, 30), (0, 20), [pit], mesh_size=2.0)
    v = ContinuumElements(mesh.nodes, mesh.elements).volumes()
    reference = _reference_volumes(p, (0, 30), (0, 20))
    for soil in range(3):
        assert v[mesh.region == soil].sum() == pytest.approx(reference[soil], rel=2e-3)
    assert v[groups["pit 2"]].sum() == pytest.approx(8 * 6 * 1.5)
    # the first lift runs from the (sloping) ground down to -1.5
    inside = np.array([[x, y] for x in np.linspace(4.05, 11.95, 80) for y in np.linspace(4.05, 9.95, 60)])
    expected = 48.0 * (p.ground(inside).mean() + 1.5)
    assert v[groups["pit 1"]].sum() == pytest.approx(expected, rel=1e-3)
    # the clay surface grazes the pit floor at a corner; the grading round
    # the short edges it leaves keeps the elements there from going flat
    assert mesh.quality().min() > 0.005


def test_site_description_round_trips_through_json(tmp_path):
    from lythos3d.examples import sloping_site
    from lythos3d.io.site_json import load_site, save_site, site_to_dict

    site = sloping_site()
    path = save_site(site, tmp_path / "site.json")
    again = load_site(path)
    assert site_to_dict(again) == json.load(open(path))
    assert [s.name for s in again.stages] == ["initial stresses", "pit: dig to -1.5",
                                              "pit: dig to -3", "factor of safety"]
    assert again.profile.tops(np.array([[15.0, 0.0, 0.0]])) == pytest.approx(
        site.profile.tops(np.array([[15.0, 0.0, 0.0]])))


def test_bad_site_descriptions_are_explained():
    from lythos3d.io.site_json import site_from_dict

    with pytest.raises(ValueError, match="missing 'soils'"):
        site_from_dict({"extent": {"x": [0, 1], "y": [0, 1]}, "bottom": -1})
    with pytest.raises(ValueError, match="unknown parameter"):
        site_from_dict({"soils": [{"name": "s", "model": "linear-elastic", "E": 1, "nu": 0.3, "c": 5}],
                        "boreholes": [], "bottom": -1, "extent": {"x": [0, 1], "y": [0, 1]}})


def test_k0_stage_on_sloping_strata_starts_close_to_equilibrium():
    """K0 stresses are exact only for level ground; on gently dipping strata
    the solver must remove a small imbalance and leave little plasticity."""
    from lythos3d.core.model import Site
    from lythos3d.core.problem import Stage

    site = Site("slope", SoilProfile(_soils(), _three_boreholes(), -15), (0, 30), (0, 20),
                stages=[Stage("initial", kind="initial", initial_stress="k0", increments=2)],
                mesh_size=3.0)
    problem, (result,) = site.run()
    assert result.converged
    z = problem.continuum.gauss_xyz.reshape(-1, 3)
    sv = site.profile.overburden(z)
    deep = sv > 100.0
    assert np.allclose(-result.state.stress[deep, 2], sv[deep], rtol=0.1)

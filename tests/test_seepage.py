"""Steady seepage against closed-form flows, and its coupling to the soil."""

import numpy as np
import pytest

from lythos3d.core.materials import LinearElastic
from lythos3d.core.mesh import box_mesh, graded
from lythos3d.core.problem import Problem, Stage
from lythos3d.core.seepage import solve_seepage
from lythos3d.core.solver import Solver
from lythos3d.core.water import GAMMA_WATER, Seepage, WaterTable


def _box(x, z, ks, split_axis=2, splits=(), h=0.5, y=1.0):
    """A box with one material per band between ``splits`` along ``split_axis``, permeabilities ``ks``."""
    breaks = list(splits)
    gx = graded(x[0], x[1], h, breaks if split_axis == 0 else [])
    gz = graded(z[0], z[1], h, breaks if split_axis == 2 else [])
    mesh = box_mesh(gx, graded(0, y, y), gz)
    c = mesh.centroids()[:, split_axis]
    mesh.region = np.searchsorted(np.asarray(breaks), c).astype(np.int64)
    mats = [LinearElastic(f"s{i}", 1e4, 0.3, 20.0, k=k) for i, k in enumerate(ks)]
    return Problem(mesh, mats)


def _wells(points, y=1.0):
    return WaterTable(wells=[(x, yy, h) for x, h in points for yy in (0.0, y)])


def test_flow_along_layers_is_the_sum_of_their_transmissivities():
    """Horizontal flow through two layers under a sloping head: Q = W (k1 t1 + k2 t2) dH / L."""
    L, dH = 6.0, 2.0
    p = _box((0, L), (-4, 0), ks=[4.0, 1.0], splits=[-2.0])        # lower layer 4, upper 1
    field = solve_seepage(p, Seepage(_wells([(0, 3.0), (L, 1.0)]), closed=("ymin", "ymax")),
                          np.ones(p.mesh.n_elements, bool))
    expected = 1.0 * (1.0 * 2.0 + 4.0 * 2.0) * dH / L
    assert field.flows["in"] == pytest.approx(expected, rel=1e-9)
    assert field.flows["out"] == pytest.approx(expected, rel=1e-9)
    x = p.mesh.nodes
    assert np.allclose(field.head, 3.0 - dH * x[:, 0] / L, atol=1e-9)


def test_flow_across_layers_follows_their_resistances_in_series():
    """Flow along x through k = 1 then k = 4: q = dH / (L1/k1 + L2/k2), the head bending at the joint."""
    ha = 3.0 - 2.0 / (2.0 / 1.0 + 4.0 / 4.0) * 2.0
    p = _box((0, 6), (-4, 0), ks=[1.0, 4.0], split_axis=0, splits=[2.0])
    field = solve_seepage(p, Seepage(_wells([(0, 3.0), (2, ha), (6, 1.0)]), closed=("ymin", "ymax")),
                          np.ones(p.mesh.n_elements, bool))
    assert field.flows["in"] == pytest.approx(4.0 * 2.0 / 3.0, rel=1e-9)
    x = p.mesh.nodes
    assert np.allclose(field.head, np.interp(x[:, 0], [0, 2, 6], [3.0, ha, 1.0]), atol=1e-9)


@pytest.mark.parametrize("psi_k, rel", [(0.7, 0.04), (0.2, 0.012)])
def test_rectangular_dam_discharges_as_charny_proved(psi_k, rel):
    """Unconfined flow through a rectangular dam, heads 5 and 1 over 6 m: q = k (H1^2 - H2^2) / 2L exactly.

    The free surface and the seepage face on the downstream side are found
    by the solution.  The unsaturated zone above the surface still carries
    a little water, more the wider its transition ``psi_k``: 2.067 at 0.7 m,
    2.016 at 0.2 m, 2.002 at 0.05 m.
    """
    L, H1, H2 = 6.0, 5.0, 1.0
    p = _box((0, L), (0, 6), ks=[1.0], splits=[], h=0.5)
    field = solve_seepage(p, Seepage(_wells([(0, H1), (L, H2)]), closed=("ymin", "ymax"), psi_k=psi_k),
                          np.ones(p.mesh.n_elements, bool))
    assert field.flows["in"] == pytest.approx((H1 ** 2 - H2 ** 2) / (2 * L), rel=rel)
    # water leaves the downstream face above the tail water: a seepage face
    assert field.flows["seepage_face"] > 0.0
    x = p.mesh.nodes
    face = np.isclose(x[:, 0], L)
    assert np.all(field.head[face] <= np.maximum(x[face, 2], H2) + 1e-6)


def test_still_water_gives_the_hydrostatic_pressure():
    p = _box((0, 4), (-6, 0), ks=[1.0], splits=[], h=1.0)
    table = WaterTable(level=-2.0)
    active = np.ones(p.mesh.n_elements, bool)
    flow = p.pore_field(Seepage(table), active)
    still = p.pore_field(table, active)
    assert np.allclose(flow.gauss, still.gauss, atol=1e-6 * GAMMA_WATER)
    assert flow.flows["in"] < 1e-9


def _pit(interface=True, toe=-8.0, water=None, install=True):
    from lythos3d.core.interfaces import InterfaceSpec
    from lythos3d.core.materials import MohrCoulomb
    from lythos3d.core.model import Model, Stratum, Volume, Wall
    from lythos3d.core.structures import PlateSection

    soil = MohrCoulomb("sand", E=4e4, nu=0.3, gamma=18.0, gamma_sat=20.0, c=5.0, phi=32.0, k=1e-5)
    wall = Wall("wall", (5, 0, toe), (5, 1, 0), PlateSection(E=3e7, nu=0.2, t=0.6),
                interface=InterfaceSpec() if interface else None)
    table = WaterTable(level=0.0)
    pumped = Seepage(table.lowered([(-1, -1), (5, -1), (5, 2), (-1, 2)], -3.0), closed=("xmin", "ymin", "ymax"))
    stages = [Stage("initial", kind="initial"), Stage("wall", install=("wall",) if install else ()),
              Stage("dig and pump", excavate=("dig",), water=pumped if water is None else water)]
    return Model("pit", (0, 20), (0, 1), -14.0, [Stratum("sand", soil, 0.0)],
                 volumes=[Volume("dig", (0, 0, -3), (5, 1, 0))],
                 walls=[wall], water=table, stages=stages, mesh_size=1.0)


def test_water_flows_under_an_impermeable_wall_into_a_pumped_pit():
    problem, results = _pit().run()
    assert all(r.converged for r in results)
    flows = results[-1].flows
    # everything that comes in from the far side is pumped out of the pit
    assert flows["pumped"] == pytest.approx(flows["in"], rel=1e-6)
    assert flows["pumped"] > 0.0
    # the head drops across the wall, not below its toe: the soil either
    # side of the wall at mid-depth sees different heads...
    head = results[-1].head
    x = problem.mesh.nodes
    mid = np.isclose(x[:, 2], -5.0) & np.isclose(x[:, 1], 0.0)
    left = head[mid & np.isclose(x[:, 0], 4.0)][0]
    right = head[mid & np.isclose(x[:, 0], 6.0)][0]
    assert right - left > 1.0
    # ...and the pore pressure is continuous below the toe, where the
    # hydrostatic drawdown would have made it jump
    deep = np.isclose(x[:, 2], -11.0) & np.isclose(x[:, 1], 0.0)
    below = head[deep & np.isclose(x[:, 0], 4.0)][0] - head[deep & np.isclose(x[:, 0], 6.0)][0]
    assert abs(below) < 0.3


def test_a_deeper_wall_lets_less_water_in_and_a_permeable_one_more():
    active = None

    def pumped(**kw):
        nonlocal active
        problem = _pit(**kw).build()
        active = ~problem.groups["dig"]
        spec = _pit().stages[-1].water
        installed = (0,) if kw.get("install", True) else ()
        return solve_seepage(problem, spec, active, installed=installed).flows["pumped"]

    base = pumped()
    assert pumped(toe=-11.0) < base
    assert pumped(interface=False) > 1.5 * base
    # a wall with interfaces is no barrier until it is built
    assert pumped(install=False) == pytest.approx(pumped(interface=False), rel=1e-6)


def test_seepage_round_trips_through_json(tmp_path):
    from lythos3d.core.model import Site
    from lythos3d.core.site import Borehole, Excavation, Soil, SoilProfile
    from lythos3d.io.site_json import load_site, save_site

    soil = LinearElastic("sand", 3e4, 0.3, 18.0, gamma_sat=20.0, k=2e-5, k_v=1e-5)
    profile = SoilProfile([Soil("sand", soil)], [Borehole("BH", 0, 0, [("sand", 0.0)])], -10.0)
    pit = Excavation("pit", [(2, 2), (6, 2), (6, 6), (2, 6)], [-2.0, -4.0], dewatered=True)
    site = Site("s", profile, (0, 10), (0, 10), [pit],
                water=Seepage(WaterTable(level=-1.0), closed=("xmin",), psi_k=0.5))
    back = load_site(save_site(site, tmp_path / "s.json"))
    assert back.water == site.water
    assert [s.water for s in back.stages] == [s.water for s in site.stages]
    assert isinstance(back.stages[2].water, Seepage)
    assert back.profile.soils[0].material.k_v == 1e-5

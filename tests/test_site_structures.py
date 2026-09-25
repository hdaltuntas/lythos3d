"""Walls and anchors drawn in plan, meshed by gmsh."""

import json

import numpy as np
import pytest

from lythos3d.core.materials import LinearElastic
from lythos3d.core.site import Borehole, Excavation, SiteAnchor, SiteWall, Soil, SoilProfile
from lythos3d.core.structures import PlateElements, PlateSection

gmsh = pytest.importorskip("gmsh", exc_type=ImportError)
from lythos3d.core.gmsh_mesh import mesh_site  # noqa: E402

SECTION = PlateSection(E=3e7, nu=0.2, t=0.6)


def _soils():
    return [Soil("fill", LinearElastic("fill", 2e4, 0.3, 18.0)),
            Soil("clay", LinearElastic("clay", 3e4, 0.3, 19.0)),
            Soil("sand", LinearElastic("sand", 6e4, 0.3, 20.0))]


def _sloping():
    return SoilProfile(_soils(), [Borehole("A", 0, 0, [("fill", 0), ("clay", -2), ("sand", -6)]),
                                  Borehole("B", 30, 0, [("fill", 1), ("sand", -4)]),
                                  Borehole("C", 0, 20, [("fill", -1), ("clay", -3), ("sand", -8)])], -15)


PIT = Excavation("pit", [(4, 4), (12, 4), (12, 10), (4, 10)], [-1.5, -3.0])
RING = SiteWall("wall", [(4, 4), (12, 4), (12, 10), (4, 10), (4, 4)], toe=-9.0, section=SECTION)


def _area_below_ground(profile, wall):
    path = np.asarray(wall.path, float)
    total = 0.0
    for a, b in zip(path[:-1], path[1:]):
        t = np.linspace(0, 1, 4001)
        top = profile.ground(a + (b - a) * t[:, None]) if wall.top is None else np.full(len(t), wall.top)
        total += np.trapezoid(top - wall.toe, t) * np.linalg.norm(b - a)
    return total


@pytest.mark.parametrize("sloping", [False, True])
def test_wall_round_a_pit_is_meshed_up_to_the_ground(sloping):
    profile = _sloping() if sloping else SoilProfile(
        _soils(), [Borehole("A", 10, 10, [("fill", 0), ("clay", -2), ("sand", -6)])], -15)
    mesh, _, walls, _ = mesh_site(profile, (0, 30), (0, 20), [PIT], mesh_size=2.0, walls=[RING])
    faces = walls["wall"]
    area = PlateElements(mesh.nodes, faces, SECTION).area.sum()
    assert area == pytest.approx(_area_below_ground(profile, RING), rel=1e-4)
    # every wall face is a face of the tetrahedra: the wall is conforming
    tet_faces = {tuple(sorted(f)) for f in mesh.elements[:, [[0, 1, 2], [0, 1, 3], [1, 2, 3], [0, 2, 3]]]
                 .reshape(-1, 3).tolist()}
    assert all(tuple(sorted(f)) in tet_faces for f in faces[:, :3].tolist())


def test_a_wall_stopping_short_of_the_base_is_kept():
    """A wall whose toe is in the soil is embedded in a volume rather than bounding one."""
    profile = SoilProfile(_soils(), [Borehole("A", 10, 10, [("fill", 0), ("clay", -2), ("sand", -6)])], -15)
    wall = SiteWall("sheet", [(10, 2), (10, 18)], toe=-5.0, top=-0.5, section=SECTION)
    mesh, _, walls, _ = mesh_site(profile, (0, 20), (0, 20), [], mesh_size=2.0, walls=[wall])
    assert PlateElements(mesh.nodes, walls["sheet"], SECTION).area.sum() == pytest.approx(16 * 4.5)


def test_anchor_points_become_nodes():
    profile = _sloping()
    points = [(12, 7, -1.0), (20.3, 13.7, -6.1)]
    mesh, _, _, nodes = mesh_site(profile, (0, 30), (0, 20), [PIT], mesh_size=2.0, walls=[RING], points=points)
    assert np.allclose(mesh.nodes[nodes], points)


def test_default_stages_install_walls_first_and_anchors_once_exposed():
    from lythos3d.core.model import Site

    strut = SiteAnchor("strut", (4, 7, -1.0), (12, 7, -1.0), EA=1e6)
    deep = SiteAnchor("deep strut", (4, 7, -2.5), (12, 7, -2.5), EA=1e6)
    site = Site("s", _sloping(), (0, 30), (0, 20), [PIT], walls=[RING], anchors=[strut, deep])
    assert [(s.name, s.excavate, s.install) for s in site.stages] == [
        ("initial stresses", (), ()),
        ("install walls", (), ("wall",)),
        ("pit: dig to -1.5", ("pit 1",), ()),
        ("stress strut", (), ("strut",)),
        ("pit: dig to -3", ("pit 2",), ()),
        ("stress deep strut", (), ("deep strut",)),
        ("factor of safety", (), ()),
    ]


def test_walls_and_anchors_round_trip_through_json(tmp_path):
    from lythos3d.examples import walled_pit
    from lythos3d.io.site_json import load_site, save_site, site_from_dict, site_to_dict

    path = save_site(walled_pit(), tmp_path / "site.json")
    again = load_site(path)
    assert site_to_dict(again) == json.load(open(path))
    assert again.walls[0].section.t == pytest.approx(0.6)
    assert again.anchors[0].prestress == -100.0
    d = json.load(open(path))
    d["walls"][0] = {"name": "w", "path": [[0, 0], [1, 0]], "toe": -3, "EA": 1.8e7, "EI": 5.4e5}
    wall = site_from_dict(d).walls[0]
    assert wall.section.EA == pytest.approx(1.8e7) and wall.section.EI == pytest.approx(5.4e5)


def test_strutted_pit_carries_the_strut_in_compression():
    from lythos3d.core.model import Site

    profile = SoilProfile([Soil("sand", LinearElastic("sand", 3e4, 0.3, 18.0))],
                          [Borehole("A", 0, 0, [("sand", 0.0)])], -10)
    pit = Excavation("pit", [(4, 4), (12, 4), (12, 10), (4, 10)], [-1.5, -3.0])
    strut = SiteAnchor("strut", (4, 7, -1.0), (12, 7, -1.0), EA=2e6, prestress=-100.0)
    site = Site("strutted", profile, (0, 16), (0, 14), [pit], walls=[RING], anchors=[strut], mesh_size=2.0)
    site.stages = [s for s in site.stages if s.kind != "ssr"]
    problem, results = site.run()
    assert all(r.converged for r in results)
    names = [s.name for s in site.stages]
    stressed = names.index("stress strut")
    assert results[stressed].bar_forces["strut"] == pytest.approx(-100.0)
    # after lock-off the strut follows the walls elastically: the change in
    # its force is EA / L times the change in distance between its ends
    bar = problem.bars[0]
    axis = (problem.mesh.nodes[bar.b] - problem.mesh.nodes[bar.a]) / 8.0
    moved = results[-1].displacement                     # the last stage's movement only
    extension = axis @ (moved[bar.b] - moved[bar.a])
    assert results[-1].bar_forces["strut"] - results[-2].bar_forces["strut"] == pytest.approx(
        2e6 / 8.0 * extension, rel=1e-6)
    assert results[-1].bar_forces["strut"] < 0.0
    _, M, _ = results[-1].plate_forces["wall"]
    assert np.abs(M).max() > 0.0


@pytest.mark.slow
def test_walled_pit_agrees_between_the_gmsh_site_and_the_structured_box():
    """A quarter of a walled pit, meshed two independent ways.

    The structured box model and the gmsh site describe the same ground,
    wall and dig; at 1 m elements they gave 0.34 and 0.34 mm at the top of
    the wall, and 30.5 and 32.2 kNm/m as the largest moment (at the corner).
    """
    from lythos3d.core.materials import MohrCoulomb
    from lythos3d.core.model import Model, Site, Stratum, Volume, Wall
    from lythos3d.core.problem import Stage

    soil = MohrCoulomb("sand", E=3.0e4, nu=0.3, gamma=18.0, c=5.0, phi=30.0)
    section = PlateSection(E=3e7, nu=0.2, t=0.5)

    def summary(problem, result, names):
        top, moment = 0.0, 0.0
        for name in names:
            k = [p.name for p in problem.plates].index(name)
            faces, el = problem.plates[k].faces, problem.plate_elements[k]
            nodes = np.unique(faces.ravel())
            nodes = nodes[np.isclose(problem.mesh.nodes[nodes, 2], 0.0)]
            top = max(top, np.linalg.norm(result.displacement[nodes, :2], axis=1).max())
            _, M, _ = result.plate_forces[name]
            vertical_is_x = np.abs(el.R[:, 0, 2]) > np.abs(el.R[:, 1, 2])
            moment = max(moment, np.abs(np.where(vertical_is_x[:, None], M[:, :, 0], M[:, :, 1])).max())
        return top, moment

    box = Model("box", (0, 16), (0, 16), -14.0, [Stratum("sand", soil, 0.0)],
                volumes=[Volume("pit 1", (0, 0, -3), (4, 4, 0))],
                walls=[Wall("wx", (4, 0, -8), (4, 4, 0), section), Wall("wy", (0, 4, -8), (4, 4, 0), section)],
                stages=[Stage("initial", kind="initial"), Stage("walls", install=("wx", "wy")),
                        Stage("dig", excavate=("pit 1",))], mesh_size=1.0)
    problem, results = box.run()
    box_top, box_moment = summary(problem, results[-1], ["wx", "wy"])

    profile = SoilProfile([Soil("sand", soil)], [Borehole("BH", 0, 0, [("sand", 0.0)])], -14.0)
    site = Site("site", profile, (0, 16), (0, 16),
                [Excavation("pit", [(0, 0), (4, 0), (4, 4), (0, 4)], [-3.0], mesh_size=0.5)],
                walls=[SiteWall("wall", [(4, 0), (4, 4), (0, 4)], toe=-8.0, section=section)],
                stages=[Stage("initial", kind="initial"), Stage("walls", install=("wall",)),
                        Stage("dig", excavate=("pit 1",))], mesh_size=2.0)
    problem, results = site.run()
    site_top, site_moment = summary(problem, results[-1], ["wall"])

    assert site_top == pytest.approx(box_top, rel=0.1)
    assert site_moment == pytest.approx(box_moment, rel=0.1)

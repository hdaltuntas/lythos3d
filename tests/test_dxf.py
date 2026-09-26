# SPDX-License-Identifier: AGPL-3.0-only
"""Plan outlines from DXF drawings."""

import json

import pytest

from lythos3d.io.dxf import Polyline, parse, read_plan, write_plan


def _dxf(*entities):
    return "\n".join(["0", "SECTION", "2", "ENTITIES", *entities, "0", "ENDSEC", "0", "EOF"]) + "\n"


R12 = _dxf(
    "0", "POLYLINE", "8", "WALL", "66", "1", "70", "0",
    "0", "VERTEX", "8", "WALL", "10", "4.0", "20", "4.0",
    "0", "VERTEX", "8", "WALL", "10", "12.0", "20", "4.0",
    "0", "SEQEND",
    "0", "LINE", "8", "AXIS", "10", "0", "20", "0", "11", "5", "21", "5",
    "0", "POINT", "8", "BOREHOLES", "10", "1.5", "20", "2.5", "30", "0.0",
)


def test_old_style_polylines_lines_and_points():
    plan = parse(R12)
    assert plan.layers == ["WALL", "AXIS", "BOREHOLES"]
    assert plan.polyline("wall").points == [(4.0, 4.0), (12.0, 4.0)]
    assert plan.polyline("AXIS").points == [(0.0, 0.0), (5.0, 5.0)]
    assert plan.points == [("BOREHOLES", (1.5, 2.5, 0.0))]
    with pytest.raises(ValueError):
        plan.polyline("PIT")


def test_lightweight_polylines_round_trip(tmp_path):
    pit = Polyline("PIT", [(4, 4), (12, 4), (12, 10), (4, 10)], closed=True)
    path = write_plan(tmp_path / "plan.dxf", [pit, Polyline("PIT", [(0, 0), (1, 1)])])
    plan = read_plan(path)
    assert plan.polyline("PIT").closed and plan.polyline("PIT").points == pit.points
    assert len(plan.on("PIT")) == 2


def test_a_site_file_takes_its_pit_and_wall_from_a_drawing(tmp_path):
    from lythos3d.io.site_json import load_site

    ring = [(4, 4), (12, 4), (12, 10), (4, 10)]
    write_plan(tmp_path / "plan.dxf", [Polyline("PIT", ring, closed=True),
                                        Polyline("WALL", ring + ring[:1])])
    site = {
        "extent": {"x": [0, 16], "y": [0, 14]}, "bottom": -10,
        "soils": [{"name": "sand", "model": "linear-elastic", "E": 3e4, "nu": 0.3, "gamma": 18}],
        "boreholes": [{"name": "A", "x": 0, "y": 0, "tops": [["sand", 0.0]]}],
        "excavations": [{"name": "pit", "polygon": {"dxf": "plan.dxf", "layer": "PIT"}, "levels": [-2]}],
        "walls": [{"name": "w", "path": {"dxf": "plan.dxf", "layer": "WALL"}, "toe": -6, "E": 3e7, "t": 0.5}],
    }
    (tmp_path / "site.json").write_text(json.dumps(site))
    loaded = load_site(tmp_path / "site.json")
    assert [tuple(p) for p in loaded.excavations[0].polygon] == [tuple(map(float, p)) for p in ring]
    # a wall drawn round the pit comes back closed
    assert len(loaded.walls[0].path) == 5 and tuple(loaded.walls[0].path[0]) == tuple(loaded.walls[0].path[-1])

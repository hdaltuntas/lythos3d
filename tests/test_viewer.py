# SPDX-License-Identifier: AGPL-3.0-only
"""The browser viewer, the HTML report and the plan editor."""

import base64
import json
import os
import re

import numpy as np
import pytest

from lythos3d.core.materials import MohrCoulomb
from lythos3d.core.model import Model, Stratum, Volume, Wall
from lythos3d.core.problem import Stage
from lythos3d.core.structures import PlateSection
from lythos3d.io.viewer import viewer_data, write_report, write_viewer


@pytest.fixture(scope="module")
def analysed():
    soil = MohrCoulomb("clay", E=2e4, nu=0.3, gamma=18.0, c=10.0, phi=25.0)
    model = Model("pit", (0, 8), (0, 1), -6.0, [Stratum("clay", soil, 0.0)],
                  volumes=[Volume("dig", (0, 0, -2), (3, 1, 0))],
                  walls=[Wall("wall", (3, 0, -4), (3, 1, 0), PlateSection(E=3e7, nu=0.2, t=0.4))],
                  stages=[Stage("initial", kind="initial"), Stage("wall", install=("wall",)),
                          Stage("dig", excavate=("dig",)), Stage("fos", kind="ssr", srf_max=1.6)],
                  mesh_size=1.0)
    return model.run()


def _data(html):
    return json.loads(re.search(r'<script type="application/json" id="l3v-data">(.*?)</script>', html, re.S)
                      .group(1).replace("<\\/", "</"))


def test_viewer_data_holds_every_stage_on_the_corner_nodes(analysed):
    problem, results = analysed
    data = viewer_data(problem, results)
    n = data["n_nodes"]
    nodes = np.frombuffer(base64.b64decode(data["nodes"]), np.float32).reshape(-1, 3)
    tets = np.frombuffer(base64.b64decode(data["tets"]), np.uint32).reshape(-1, 4)
    assert len(nodes) == n and len(tets) == problem.mesh.n_elements and tets.max() == n - 1
    assert [s["name"] for s in data["stages"]] == [r.name for r in results]
    dig = data["stages"][2]
    active = np.frombuffer(base64.b64decode(dig["active"]), np.uint8)
    assert active.sum() == results[2].active.sum()
    u = np.frombuffer(base64.b64decode(dig["fields"]["displacement |u| (mm)"]), np.float32)
    assert u.max() == pytest.approx(1000 * results[2].max_displacement, rel=1e-4)
    assert "wall" in dig["plates"] and data["stages"][3]["fos"] == results[3].srf


def test_report_and_viewer_are_self_contained_pages(analysed, tmp_path):
    problem, results = analysed
    report = open(write_report(tmp_path / "report.html", problem, results, title="a pit")).read()
    assert "<h2>Stages</h2>" in report and "Strength reduction, fos" in report
    assert "wall: vertical bending moment" in report
    assert "data:text/javascript;base64," in report                  # three.js inlined: opens offline
    assert len(_data(report)["stages"]) == len(results)
    small = open(write_viewer(tmp_path / "view.html", problem, results, offline=False)).read()
    assert "cdn.jsdelivr.net/npm/three@" in small and len(small) < len(report)


def test_the_plan_editor_ships_with_the_package(tmp_path):
    from lythos3d.cli import main

    out = tmp_path / "editor.html"
    assert main(["editor", "-o", str(out)]) == 0
    text = out.read_text()
    for needle in ("function toJSON", "function parseDXF", "dewatered", "seepage", "toggle3d"):
        assert needle in text
    assert "data:text/javascript;base64," in text                  # three.js inlined for the 3D view


def _chromium():
    try:
        from playwright.sync_api import sync_playwright  # noqa: F401
    except ImportError:
        return None
    for path in ("/opt/pw-browsers/chromium-1194/chrome-linux/chrome",):
        if os.path.exists(path):
            return path
    return None


@pytest.mark.skipif(_chromium() is None, reason="needs playwright and a chromium")
def test_in_a_browser_the_editor_draws_a_site_that_lythos_can_read(tmp_path):
    from playwright.sync_api import sync_playwright

    from lythos3d.cli import main
    from lythos3d.io.site_json import site_from_dict

    main(["editor", "-o", str(tmp_path / "editor.html")])
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=_chromium(), args=["--use-gl=swiftshader", "--ignore-gpu-blocklist"])
        page = browser.new_page(viewport={"width": 1300, "height": 800}, accept_downloads=True)
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(f"file://{tmp_path / 'editor.html'}")
        box = page.locator("#svg").bounding_box()

        def click(x, y):
            sx, sy = page.evaluate(f"toScreen({x}, {y})")
            page.mouse.click(box["x"] + sx, box["y"] + sy)

        page.click("button[data-tool=hole]")
        click(2, 2)
        page.click("button[data-tool=pit]")
        for q in [(10, 10), (20, 10), (20, 18), (10, 18)]:
            click(*q)
        page.keyboard.press("Enter")
        page.click("button[data-tool=wall]")
        for q in [(10, 10), (20, 10), (20, 18), (10, 18), (10, 10)]:
            click(*q)
        # the 3D view of what has been drawn, with three.js inlined in the file
        page.click("#toggle3d")
        page.wait_for_timeout(1500)
        legend = page.inner_text("#legend3d")
        page.click("#toggle3d")
        with page.expect_download() as download:
            page.click("#save")
        site = site_from_dict(json.load(open(download.value.path())))
        browser.close()
    assert not errors
    assert "clay" in legend and "pits" in legend and "walls" in legend
    assert [tuple(p) for p in site.excavations[0].polygon] == [(10, 10), (20, 10), (20, 18), (10, 18)]
    assert len(site.walls[0].path) == 5 and site.profile.boreholes[0].x == 2.0
    assert [s.name for s in site.stages][-1] == "factor of safety"


@pytest.mark.skipif(_chromium() is None, reason="needs playwright and a chromium")
def test_in_a_browser_the_report_draws_without_errors(analysed, tmp_path):
    from playwright.sync_api import sync_playwright

    problem, results = analysed
    path = write_report(tmp_path / "report.html", problem, results)
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=_chromium(), args=["--use-gl=swiftshader", "--ignore-gpu-blocklist"])
        page = browser.new_page()
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(f"file://{path}")
        page.wait_for_timeout(2500)
        page.select_option(".l3v-panel select >> nth=2", "2")       # a cut along y
        page.wait_for_timeout(500)
        legend = page.inner_text(".l3v-legend")
        browser.close()
    assert not errors
    assert "displacement" in legend


def test_the_3d_view_gets_the_soil_surfaces_meshing_uses():
    from lythos3d.examples import walled_pit
    from lythos3d.gui import profile_grid
    from lythos3d.io.site_json import site_to_dict

    site = walled_pit()
    g = profile_grid(site_to_dict(site))
    xs, ys, tops = np.array(g["xs"]), np.array(g["ys"]), np.array(g["tops"])
    assert tops.shape == (site.profile.n_soils, len(ys), len(xs))
    X, Y = np.meshgrid(xs, ys)
    exact = site.profile.tops(np.column_stack([X.ravel(), Y.ravel()]))
    assert np.allclose(tops.reshape(site.profile.n_soils, -1).T, exact)

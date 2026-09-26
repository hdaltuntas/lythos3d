"""Consolidation in time against Terzaghi."""

import numpy as np
import pytest

from lythos3d.core.analysis import SurfaceLoad
from lythos3d.core.consolidation import time_steps
from lythos3d.core.materials import LinearElastic, MohrCoulomb
from lythos3d.core.mesh import box_mesh, graded
from lythos3d.core.problem import Problem, Stage
from lythos3d.core.solver import Solver
from lythos3d.core.water import GAMMA_WATER

CLAY = LinearElastic("clay", E=1e4, nu=0.3, drainage="undrained", k=1e-3)


def _terzaghi_degree(Tv):
    m = np.arange(400)
    M = np.pi * (2 * m + 1) / 2
    return 1.0 - np.sum(2.0 / M ** 2 * np.exp(-M ** 2 * Tv))


def test_time_steps_grow_and_add_up():
    dt = time_steps(10.0, 20)
    assert dt.sum() == pytest.approx(10.0)
    assert np.all(np.diff(dt) > 0) and dt[-1] < 0.15 * 10.0


@pytest.mark.parametrize("two_way", [False, True])
def test_one_dimensional_consolidation_follows_terzaghi(two_way):
    """A column loaded undrained, then left to drain through its top (and base).

    With the water compressible as the undrained Poisson's ratio makes it,
    c_v = (k / gamma_w) / (1/M + n/K_w), the excess starts at
    q (Kw/n) / (M + Kw/n), and the settlement is
    w = q H / M - p0 H (1 - U(Tv)) / M, with the drainage path H (or H/2).
    """
    H, q = 4.0, 100.0
    mesh = box_mesh(graded(0, 1, 1.0), graded(0, 1, 1.0), graded(-H, 0, 0.25))
    load = SurfaceLoad("z", 0.0, (0, 0, -q))
    solver = Solver(Problem(mesh, [CLAY]), tolerance=1e-8)
    solver.run_stage(Stage("load", loads=(load,), increments=1))
    M, kw = CLAY.oedometer_modulus, CLAY.water_bulk_modulus
    cv = CLAY.k / GAMMA_WATER / (1 / M + 1 / kw)
    p0 = q * kw / (M + kw)
    path = H / 2 if two_way else H
    top = np.isclose(mesh.nodes[:, 2], 0.0)
    elapsed = 0.0
    for Tv in (0.05, 0.3, 1.0):
        t = Tv * path ** 2 / cv
        r = solver.run_stage(Stage(f"to Tv {Tv}", kind="consolidation", time=t - elapsed, loads=(load,),
                                   increments=40, reset_displacements=False,
                                   drained_sides=("base",) if two_way else ()))
        elapsed = t
        assert r.converged
        assert r.time == pytest.approx(t)
        w = -r.displacement[top, 2].mean()
        exact = q * H / M - p0 * H * (1 - _terzaghi_degree(Tv)) / M
        assert w == pytest.approx(exact, rel=5e-3)
    # the excess left where the water has furthest to go; backward Euler
    # lags a little behind here, by 5% at 40 steps and 2.5% at 160
    i = int(np.argmax(r.excess_pore_pressure))
    z = solver.p.continuum.gauss_xyz.reshape(-1, 3)[i, 2]
    d = -z if not two_way else min(-z, H + z)
    m = np.arange(400)
    Mm = np.pi * (2 * m + 1) / 2
    exact_p = p0 * np.sum(2 / Mm * np.sin(Mm * d / path) * np.exp(-Mm ** 2 * 1.0))
    assert r.excess_pore_pressure[i] == pytest.approx(exact_p, rel=0.06)
    assert len(r.consolidation) == 40 and r.consolidation[-1][0] == pytest.approx(elapsed)


def test_plastic_soil_consolidates_under_a_footing():
    soil = MohrCoulomb("clay", E=8e3, nu=0.3, gamma=17.0, c=8.0, phi=24.0, drainage="undrained", k=1e-4)
    mesh = box_mesh(graded(0, 6, 0.75, [1.0]), graded(0, 1, 1.0), graded(-5, 0, 0.5))
    load = SurfaceLoad("z", 0.0, (0, 0, -60.0), where=lambda c: c[:, 0] < 1.0)
    solver = Solver(Problem(mesh, [soil]))
    solver.run_stage(Stage("initial", kind="initial", initial_stress="gravity"))
    undrained = solver.run_stage(Stage("load", loads=(load,)))
    later = solver.run_stage(Stage("wait", kind="consolidation", time=50.0, loads=(load,),
                                   drained_sides=("xmax",), reset_displacements=False))
    assert undrained.converged and later.converged
    corner = np.nonzero(np.all(np.isclose(mesh.nodes, [0, 0, 0]), axis=1))[0][0]
    # the footing keeps settling as the water leaves, and the excess falls
    assert later.displacement[corner, 2] < undrained.displacement[corner, 2] < 0
    assert later.excess_pore_pressure.max() < undrained.excess_pore_pressure.max()


def test_consolidation_stage_needs_a_time():
    with pytest.raises(ValueError):
        Stage("wait", kind="consolidation")

# SPDX-License-Identifier: AGPL-3.0-only
"""Plates and bars acting with the ground."""

import numpy as np
import pytest

from lythos3d.core.analysis import SurfaceLoad
from lythos3d.core.materials import LinearElastic, MohrCoulomb
from lythos3d.core.model import Anchor, Model, Stratum, Volume, Wall
from lythos3d.core.problem import Stage
from lythos3d.core.structures import PlateSection

CLAY = LinearElastic("clay", E=1.0e4, nu=0.3, gamma=0.0)
CONCRETE = PlateSection(E=3.0e7, nu=0.2, t=0.5)


def _column(**kw):
    return Model("column", (0, 4), (0, 3), -8.0, [Stratum("clay", CLAY, 0.0)], mesh_size=1.0, **kw)


def test_raft_under_uniform_pressure_settles_as_the_ground_would():
    """A plate on the surface of a laterally restrained column: 1D compression.

    The raft moves down as a whole, so it adds nothing but its weight: the
    settlement is (q + w) H / M exactly and it carries no moment.
    """
    q, w = 50.0, 12.0
    raft = Wall("raft", (0, 0, 0), (4, 3, 0), PlateSection(E=3e7, nu=0.2, t=0.5, weight=w))
    model = _column(walls=[raft], stages=[Stage("load", install=("raft",),
                                                loads=(SurfaceLoad("z", 0.0, (0, 0, -q)),))])
    problem, (result,) = model.run()
    top = np.isclose(problem.mesh.nodes[:, 2], 0.0)
    expected = -(q + w) * 8.0 / CLAY.oedometer_modulus
    assert np.allclose(result.displacement[top, 2], expected, rtol=1e-8)
    N, M, Q = result.plate_forces["raft"]
    assert np.abs(M).max() < 1e-6 * q and np.abs(Q).max() < 1e-6 * q


def test_a_plate_installed_in_deformed_ground_starts_free_of_force():
    soil = LinearElastic("clay", E=1.0e4, nu=0.3, gamma=20.0)
    model = Model("x", (0, 4), (0, 3), -8.0, [Stratum("clay", soil, 0.0)], mesh_size=1.0,
                  walls=[Wall("wall", (2, 0, -6), (2, 3, 0), CONCRETE)],
                  stages=[Stage("gravity", kind="initial", initial_stress="gravity"),
                          Stage("wall", install=("wall",))])
    _, results = model.run()
    N, M, Q = results[1].plate_forces["wall"]
    assert max(np.abs(N).max(), np.abs(M).max(), np.abs(Q).max()) < 1e-8
    assert "wall" not in results[0].plate_forces


def test_anchor_is_stressed_to_its_lock_off_load_and_then_responds_elastically():
    EA, P, q = 2.0e5, 150.0, 200.0
    footing = SurfaceLoad("z", 0.0, (0, 0, -q),
                          where=lambda c: (np.abs(c[:, 0] - 2.0) < 0.5) & (np.abs(c[:, 1] - 1.5) < 0.5))
    model = _column(anchors=[Anchor("tie", (1, 1, 0), (3, 1, 0), EA, prestress=P)],
                    stages=[Stage("stress", install=("tie",)),
                            Stage("hold"),
                            Stage("load", loads=(footing,))])
    problem, results = model.run()
    assert results[0].bar_forces["tie"] == pytest.approx(P)
    # the jack's pull has shortened the ground between the ends; locking off
    # keeps the force at P when nothing else changes
    assert results[1].bar_forces["tie"] == pytest.approx(P, rel=1e-9)
    # the footing between the ends moves them, and the tie answers elastically
    a, b = problem.bars[0].a, problem.bars[0].b
    u = results[2].displacement                    # this stage's displacement only
    stretch = u[b, 0] - u[a, 0]
    assert abs(stretch) > 1e-5
    assert results[2].bar_forces["tie"] == pytest.approx(P + EA / 2.0 * stretch, rel=1e-6)


def test_a_stiff_strut_holds_the_top_of_a_wall():
    """An excavation beside a wall in a plane-strain slice, with and without a prop."""
    soil = MohrCoulomb("sand", E=3e4, nu=0.3, gamma=18.0, c=5.0, phi=30.0)

    def run(with_strut):
        anchors = [Anchor("strut", (5, 0, 0), (0, 0, 0), EA=1e9, fixed_end=True)] if with_strut else []
        stages = [Stage("initial", kind="initial"),
                  Stage("wall", install=("wall",) + (("strut",) if with_strut else ()))]
        stages.append(Stage("dig", excavate=("dig",)))
        model = Model("slice", (0, 20), (0, 1), -12.0, [Stratum("sand", soil, 0.0)],
                      volumes=[Volume("dig", (0, 0, -3), (5, 1, 0))],
                      walls=[Wall("wall", (5, 0, -8), (5, 1, 0), CONCRETE)],
                      anchors=anchors, stages=stages, mesh_size=1.0)
        problem, results = model.run()
        assert all(r.converged for r in results)
        top = np.nonzero(np.all(np.isclose(problem.mesh.nodes, [5, 0, 0]), axis=1))[0][0]
        return results[-1].displacement[top, 0], results[-1]

    free, _ = run(False)
    propped, result = run(True)
    assert free < -1e-3                              # the wall leans into the dig
    assert abs(propped) < 1e-3 * abs(free)
    assert result.bar_forces["strut"] < 0            # a strut is in compression


def test_wall_carries_its_moment_into_the_soil():
    """Equilibrium of a plate loaded at its free end: the soil reacts, the wall bends."""
    soil = LinearElastic("clay", E=2.0e4, nu=0.3, gamma=0.0)
    model = Model("x", (0, 8), (0, 1), -10.0, [Stratum("clay", soil, 0.0)], mesh_size=0.5,
                  walls=[Wall("wall", (4, 0, -6), (4, 1, 0), CONCRETE)],
                  stages=[Stage("push", install=("wall",), loads=(
                      SurfaceLoad("z", 0.0, (0, 0, -100.0), where=lambda c: c[:, 0] < 4.0),))])
    _, (result,) = model.run()
    N, M, Q = result.plate_forces["wall"]
    assert np.abs(M).max() > 1.0                      # the one-sided surcharge bends the wall
    assert np.all(np.isfinite(M))


@pytest.mark.slow
def test_cantilever_wall_matches_2d_lythos_in_plane_strain():
    """A wall retaining a 3 m cut, as a 3D slice against 2D Lythos.

    Sand (c' = 5 kPa, phi' = 30 deg), K0 initial stresses, the wall wished
    in place and the cut dug in one go, no interfaces.  2D Lythos, with the
    beam given the plane-strain plate stiffness E t^3 / 12 (1 - nu^2), gives
    a largest moment of 9.83 kNm/m and a top displacement of 1.22 mm at 0.5 m
    elements (10.62 and 1.32 at 0.25 m, where the 3D slice gives 10.40 and
    1.28).
    """
    h = 0.5
    soil = MohrCoulomb("sand", E=3.0e4, nu=0.3, gamma=18.0, c=5.0, phi=30.0, psi=0.0)
    model = Model("slice", (0, 20), (0, h), -12.0, [Stratum("sand", soil, 0.0)],
                  volumes=[Volume("dig", (0, 0, -3), (5, h, 0))],
                  walls=[Wall("wall", (5, 0, -8), (5, h, 0), PlateSection(E=3e7, nu=0.2, t=0.5))],
                  stages=[Stage("initial", kind="initial"), Stage("wall", install=("wall",)),
                          Stage("dig", excavate=("dig",))], mesh_size=h)
    problem, results = model.run(tolerance=1e-4)
    assert all(r.converged for r in results)
    N, M, Q = results[-1].plate_forces["wall"]
    R = problem.plate_elements[0].R
    vertical_is_x = np.abs(R[:, 0, 2]) > np.abs(R[:, 1, 2])
    moment = np.abs(np.where(vertical_is_x[:, None], M[:, :, 0], M[:, :, 1])).max()
    top = np.nonzero(np.all(np.isclose(problem.mesh.nodes, [5, 0, 0]), axis=1))[0][0]
    assert moment == pytest.approx(9.83, rel=0.05)
    assert -results[-1].displacement[top, 0] == pytest.approx(1.22e-3, rel=0.05)

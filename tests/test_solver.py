# SPDX-License-Identifier: AGPL-3.0-only
"""Staged construction and strength reduction."""

import numpy as np
import pytest

from lythos3d.core.analysis import SurfaceLoad, linear_static
from lythos3d.core.materials import LinearElastic, MohrCoulomb
from lythos3d.core.mesh import box_mesh, graded
from lythos3d.core.model import Model, Stratum, Volume
from lythos3d.core.problem import Problem, Stage
from lythos3d.core.solver import Solver

FILL = LinearElastic("fill", E=2.0e4, nu=0.3, gamma=18.0)
CLAY = LinearElastic("clay", E=1.0e4, nu=0.3, gamma=20.0)


def _column():
    return box_mesh(graded(0, 4, 1), graded(0, 3, 1), graded(-10, 0, 1))


def test_gravity_loading_reproduces_the_linear_solution():
    mesh = _column()
    reference = linear_static(mesh, [CLAY])
    solver = Solver(Problem(mesh, [CLAY]))
    result = solver.run([Stage("initial", kind="initial", initial_stress="gravity")])[0]
    assert result.converged
    assert np.allclose(result.state.stress, reference.stress, atol=1e-9)
    # the initial stage leaves no displacement to report
    assert np.abs(result.displacement).max() == 0.0


def test_k0_procedure_sets_geostatic_stresses_in_equilibrium():
    soil = MohrCoulomb("sand", E=3e4, nu=0.3, gamma=19.0, c=1.0, phi=30.0)
    model = Model("level ground", (0, 4), (0, 3), -8.0,
                  [Stratum("fill", FILL, 0.0), Stratum("sand", soil, -3.0)], mesh_size=1.0,
                  stages=[Stage("initial", kind="initial", initial_stress="k0")])
    problem, (result,) = model.run()
    assert result.converged
    z = problem.continuum.gauss_xyz.reshape(-1, 3)[:, 2]
    sv = np.where(z > -3.0, FILL.gamma * -z, FILL.gamma * 3.0 + soil.gamma * (-3.0 - z))
    k0 = np.where(z > -3.0, FILL.k0, soil.k0)
    s = result.state.stress
    assert np.allclose(s[:, 2], -sv, atol=1e-9)
    assert np.allclose(s[:, 0], -k0 * sv, atol=1e-9) and np.allclose(s[:, 1], -k0 * sv, atol=1e-9)
    # level ground and level strata: the K0 field is already in equilibrium
    assert all(log.iteration == 0 for log in result.iterations)
    assert not result.state.yielding.any()


def test_k0_needs_an_overburden():
    solver = Solver(Problem(_column(), [CLAY]))
    with pytest.raises(ValueError, match="K0 procedure needs"):
        solver.run([Stage("initial", kind="initial", initial_stress="k0")])


def _two_layer_model(**kw):
    return Model("dig", (0, 4), (0, 3), -10.0,
                 [Stratum("fill", FILL, 0.0), Stratum("clay", CLAY, -2.0)],
                 volumes=[Volume("top", (0, 0, -2), (4, 3, 0))], mesh_size=1.0, **kw)


def test_excavating_a_layer_unloads_the_ground_below_it():
    """Removing 2 m of fill over the whole area heaves the clay by gamma h H / M."""
    model = _two_layer_model(stages=[Stage("initial", kind="initial"),
                                     Stage("excavate", excavate=("top",))])
    problem, results = model.run()
    assert all(r.converged for r in results)
    nodes = problem.mesh.nodes
    formation = np.isclose(nodes[:, 2], -2.0)
    heave = FILL.gamma * 2.0 * 8.0 / CLAY.oedometer_modulus
    assert np.allclose(results[1].displacement[formation, 2], heave, rtol=1e-9)
    # the nodes left with no soil around them are held, not left to float
    orphan = nodes[:, 2] > -2.0 + 1e-9
    assert np.abs(results[1].displacement[orphan]).max() == 0.0
    assert not results[1].active[problem.groups["top"]].any()


def test_unknown_group_is_refused_before_anything_runs():
    model = _two_layer_model(stages=[Stage("initial", kind="initial"),
                                     Stage("excavate", excavate=("pit",))])
    with pytest.raises(ValueError, match="unknown group 'pit'"):
        model.run()


def test_model_validation():
    with pytest.raises(ValueError, match="top down"):
        Model("bad", (0, 1), (0, 1), -5, [Stratum("a", FILL, -2), Stratum("b", CLAY, 0)])
    with pytest.raises(ValueError, match="contains no part"):
        Model("bad", (0, 1), (0, 1), -5, [Stratum("a", FILL, 0)],
              volumes=[Volume("v", (5, 5, -1), (6, 6, 0))]).build()


def test_footing_on_mohr_coulomb_yields_and_settles_more_than_elastic():
    """A footing loaded past first yield but well short of collapse.

    The plastic zone must spread and the settlement exceed the elastic one,
    and Newton must still converge in every increment.
    """
    soil = MohrCoulomb("clay", E=2e4, nu=0.3, gamma=0.0, c=30.0, phi=0.0, tension_cutoff=None)
    mesh = box_mesh(graded(0, 5, 0.5, [1.0]), graded(0, 5, 0.5, [1.0]), graded(-5, 0, 0.5))
    footing = SurfaceLoad("z", 0.0, (0, 0, -100.0), where=lambda c: (c[:, 0] < 1) & (c[:, 1] < 1))
    stage = Stage("load", loads=(footing,), increments=4)
    plastic = Solver(Problem(mesh, [soil])).run([stage])[0]
    elastic = Solver(Problem(mesh, [LinearElastic("e", 2e4, 0.3)])).run([stage])[0]
    assert plastic.converged
    assert plastic.plastic_fraction > 0.0
    corner = np.nonzero(np.all(np.isclose(mesh.nodes, 0.0), axis=1))[0][0]
    assert plastic.displacement[corner, 2] < elastic.displacement[corner, 2] < 0
    iterations_per_increment = len(plastic.iterations) / 4
    assert iterations_per_increment < 10


def test_strength_reduction_of_elastic_ground_reaches_the_upper_limit():
    solver = Solver(Problem(_column(), [CLAY]))
    results = solver.run([Stage("initial", kind="initial", initial_stress="gravity"),
                          Stage("fos", kind="ssr", srf_max=1.5)])
    assert results[-1].srf == pytest.approx(1.5)
    assert "largest factor" in results[-1].message


def slope_mesh(h: float, width: float | None = None):
    """The 2:1 benchmark slope of Lythos: 10 m high, toe at x = 5, crest at x = 25."""
    y = graded(0, width, h) if width else np.array([0.0, h])
    base = box_mesh(graded(0, 1, h / 35), y, graded(0, 10, h))

    def to_slope(p):
        x0 = 5 + 2 * p[:, 2]
        return np.column_stack([x0 + p[:, 0] * (40 - x0), p[:, 1], p[:, 2]])

    return base.mapped(to_slope)


SLOPE_SOIL = MohrCoulomb("silty clay", E=1e5, nu=0.3, c=10.0, phi=20.0, psi=0.0, gamma=20.0)


@pytest.mark.slow
def test_plane_strain_slice_reproduces_the_2d_slope_factor_of_safety():
    """The Lythos 2:1 slope as a 3D slice one element thick, held in plane strain.

    Every node lies on one of the two faces normal to y, where the box
    restraints hold v = 0, so this is plane strain.  2D Lythos gives 1.430 at
    this element size (and 1.381 on its finest mesh, against 1.377 from an
    independent Bishop search); the 3D element must land beside it.
    """
    problem = Problem(slope_mesh(2.5), [SLOPE_SOIL])
    results = Solver(problem, tolerance=2e-3).run([
        Stage("gravity", kind="initial", initial_stress="gravity"),
        Stage("fos", kind="ssr", srf_min=0.8, srf_max=2.5)])
    fos = results[-1].srf
    assert fos == pytest.approx(1.430, abs=0.02), fos
    curve = results[-1].srf_curve
    assert curve[-1][1] > 5 * max(curve[0][1], 1e-6)
